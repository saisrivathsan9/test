"""
Federated Averaging + Evaluation for Lambda Aggregator.

USE THIS FILE IN YOUR LAMBDA FUNCTION (numpy only — NO PyTorch).

This file provides:
  - federated_average()  — weighted FedAvg on numpy state dicts
  - lenet5_forward()     — numpy-only LeNet-5 forward pass
  - evaluate_model()     — compute accuracy and loss on test set
  - load_test_data()     — load test images from S3 tar.gz
  - save_npz() / load_npz() — .npz serialization helpers

"""

import io
import os
import json
import tarfile
import logging

import boto3
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aggregator")

# MNIST normalization constants (same as torchvision default)
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# S3 key prefixes
MODELS_PREFIX = "models/"
UPDATES_PREFIX = "updates/"
METRICS_PREFIX = "metrics/"

# S3 client (reused across warm Lambda invocations)
s3_client = boto3.client("s3", region_name="us-west-2")

# iot client for sending mqtt messages to trigger next round
iot_client = boto3.client("iot-data", region_name="us-west-2")

# Test data cache (persists across warm invocations)
_cached_test_data = None


# ============================================================================
# FedAvg Aggregation (numpy)
# ============================================================================

def federated_average(client_updates):
    """Weighted Federated Averaging.

    Computes:
      global_weights[k] = SUM( (n_i / n_total) * client_weights_i[k] )

    Args:
        client_updates: list of (state_dict, num_samples) tuples.
            state_dict: dict of numpy arrays (keys = layer names)
            num_samples: int — how many training samples that client used

    Returns:
        dict of numpy arrays — the aggregated global model state_dict.

    Example:
        # After downloading all client .npz files:
        client_updates = [
            (load_npz(client_0_bytes), 600),
            (load_npz(client_1_bytes), 600),
            ...
        ]
        global_sd = federated_average(client_updates)
        save_npz(global_sd)  # upload to S3
    """
    if not client_updates:
        raise ValueError("No client updates to aggregate")

    total = sum(n for _, n in client_updates)
    if total == 0:
        raise ValueError("Total samples across all clients is 0")

    first = client_updates[0][0]
    result = {k: np.zeros_like(first[k], dtype=np.float64) for k in first}

    for sd, n in client_updates:
        w = n / total
        for k in result:
            result[k] += w * sd[k].astype(np.float64)

    return {k: v.astype(first[k].dtype) for k, v in result.items()}


# ============================================================================
# .npz Serialization Helpers
# ============================================================================

def save_npz(state_dict):
    """Serialize a numpy state_dict to .npz bytes.

    Args:
        state_dict: dict of numpy arrays (e.g., from federated_average())

    Returns:
        bytes — .npz content, ready for s3.put_object(Body=...)
    """
    buf = io.BytesIO()
    np.savez(buf, **state_dict)
    return buf.getvalue()


def load_npz(data):
    """Deserialize .npz bytes to a dict of numpy arrays.

    Args:
        data: bytes — raw .npz content from s3.get_object()["Body"].read()

    Returns:
        dict of numpy arrays (keys = layer names)
    """
    npz = np.load(io.BytesIO(data))
    return {k: npz[k] for k in npz.files}


# ============================================================================
# Numpy-only LeNet-5 Forward Pass (for evaluation in Lambda)
# ============================================================================

def _conv2d(x, w, b, pad=0):
    """2D convolution. x: (N,C,H,W), w: (F,C,kH,kW), b: (F,)."""
    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    N, C, H, W = x.shape
    F, _, kH, kW = w.shape
    oH, oW = H - kH + 1, W - kW + 1
    out = np.zeros((N, F, oH, oW))
    for f in range(F):
        for i in range(oH):
            for j in range(oW):
                out[:, f, i, j] = np.sum(
                    x[:, :, i:i+kH, j:j+kW] * w[f], axis=(1, 2, 3)
                ) + b[f]
    return out


def _relu(x):
    return np.maximum(0, x)


def _max_pool2d(x, size=2):
    N, C, H, W = x.shape
    oH, oW = H // size, W // size
    out = np.zeros((N, C, oH, oW))
    for i in range(oH):
        for j in range(oW):
            out[:, :, i, j] = x[:, :,
                                i*size:(i+1)*size,
                                j*size:(j+1)*size].max(axis=(2, 3))
    return out


def _linear(x, w, b):
    return x @ w.T + b


def lenet5_forward(sd, images):
    """Forward pass through LeNet-5 using numpy arrays only.

    Args:
        sd: dict of numpy arrays (model state_dict with keys:
            conv1.weight, conv1.bias, conv2.weight, conv2.bias,
            fc1.weight, fc1.bias, fc2.weight, fc2.bias,
            fc3.weight, fc3.bias)
        images: numpy array of shape (N, 1, 28, 28) — preprocessed MNIST images

    Returns:
        numpy array of shape (N, 10) — logits (unnormalized class scores)
    """
    x = images
    x = _max_pool2d(_relu(_conv2d(x, sd['conv1.weight'], sd['conv1.bias'], pad=2)), 2)
    x = _max_pool2d(_relu(_conv2d(x, sd['conv2.weight'], sd['conv2.bias'])), 2)
    x = x.reshape(x.shape[0], -1)
    x = _relu(_linear(x, sd['fc1.weight'], sd['fc1.bias']))
    x = _relu(_linear(x, sd['fc2.weight'], sd['fc2.bias']))
    x = _linear(x, sd['fc3.weight'], sd['fc3.bias'])
    return x


# ============================================================================
# Evaluation Helpers
# ============================================================================

def cross_entropy_loss(logits, labels):
    """Compute cross-entropy loss (numpy).

    Args:
        logits: (N, C) float array — raw model output
        labels: (N,) int array — ground truth class indices

    Returns:
        float — mean cross-entropy loss
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def transform_image(img):
    """Convert a PIL image to normalized numpy array.

    Args:
        img: PIL Image

    Returns:
        numpy array of shape (1, 28, 28) — normalized grayscale image
    """
    img = img.convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    return ((arr - MNIST_MEAN) / MNIST_STD).reshape(1, 28, 28)


def load_test_data(global_bucket):
    """Download test set from S3 and return as numpy arrays.

    Caches in memory across warm Lambda invocations.
    Test data (labels.csv, archives/test.tar.gz) is in global-bucket.

    Args:
        global_bucket: global-bucket name (e.g., "{ASU_ID}-global-bucket")

    Returns:
        (images, labels) tuple:
            images: numpy array (N, 1, 28, 28) float32
            labels: numpy array (N,) int64
    """
    global _cached_test_data
    if _cached_test_data is not None:
        return _cached_test_data

    logger.info("Loading test set from S3 (one-time cache) ...")

    # Labels
    resp = s3_client.get_object(Bucket=global_bucket, Key="labels.csv")
    content = resp["Body"].read().decode()
    labels_map = {}
    for line in content.strip().split("\n")[1:]:
        parts = line.strip().split(",")
        labels_map[parts[0]] = int(parts[2])

    # Test images
    resp = s3_client.get_object(Bucket=global_bucket, Key="archives/test.tar.gz")
    tar_bytes = resp["Body"].read()

    images = []
    targets = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".png"):
                continue
            filename = os.path.basename(member.name)
            if filename not in labels_map:
                continue
            f = tar.extractfile(member)
            img = Image.open(io.BytesIO(f.read()))
            images.append(transform_image(img))
            targets.append(labels_map[filename])

    images_np = np.concatenate(images, axis=0).reshape(len(images), 1, 28, 28)
    labels_np = np.array(targets, dtype=np.int64)
    _cached_test_data = (images_np, labels_np)
    logger.info(f"Test set cached: {len(images)} images")
    return _cached_test_data


def evaluate_model(sd, test_images, test_labels):
    """Evaluate a model state_dict on the test set.

    Args:
        sd: dict of numpy arrays (model state_dict)
        test_images: numpy array (N, 1, 28, 28)
        test_labels: numpy array (N,) int64

    Returns:
        dict with keys:
            "accuracy": float (0.0-1.0)
            "loss": float (cross-entropy)
            "total": int (number of test samples)
            "correct": int (number correct)

    Example:
        images, labels = load_test_data(global_bucket)
        result = evaluate_model(global_sd, images, labels)
        # result["accuracy"] → 0.9729
        # result["loss"] → 0.0862
    """
    logits = lenet5_forward(sd, test_images)
    preds = logits.argmax(axis=1)
    acc = float((preds == test_labels).mean())
    loss = cross_entropy_loss(logits, test_labels)
    return {
        "accuracy": acc,
        "loss": loss,
        "total": len(test_labels),
        "correct": int((preds == test_labels).sum()),
    }


# ============================================================================
# TODO: Implement your Lambda handler below
# ============================================================================

def handler(event, context):
    """Lambda handler — triggered by S3 event on updates/*.npz.

    This function is invoked each time a worker uploads a .npz model
    file to the local-bucket. You need to:

    1. Parse the S3 event to get bucket name and object key
    2. Extract round_id from the key
    3. List all .npz files for this round to check if all clients reported
    4. If not all clients → return early
    5. If all clients reported → aggregate:
       a. Download each client's .npz model weights
       b. Call federated_average() to get the aggregated model
          (use equal weighting: 1 per client)
       c. Upload aggregated model to global-bucket
       d. Evaluate on test set
       e. Write metrics/round_{R}.json to global-bucket

    Args:
        event: S3 event dict (see S3 EVENT FORMAT in docstring above)
        context: Lambda context object (not used)

    Returns:
        dict with statusCode and body
    """
    # reading config from env vars
    num_clients = int(os.environ.get("NUM_CLIENTS", "10"))
    total_rounds = int(os.environ.get("TOTAL_ROUNDS", "5"))
    asu_id = os.environ.get("ASU_ID", "1224208336")

    global_bucket = f"{asu_id}-global-bucket"
    local_bucket = f"{asu_id}-local-bucket"

    # figure out which file triggered us
    record = event["Records"][0]
    obj_key = record["s3"]["object"]["key"]

    # parse round number from the key
    try:
        tmp = obj_key.replace("updates/local_model_round_", "").replace(".npz", "")
        round_id = int(tmp.split("_worker_")[0])
    except:
        return {"statusCode": 400, "body": "bad key"}

    # checking how many workers have uploaded for this round
    count = 0
    for c in range(num_clients):
        k = f"updates/local_model_round_{round_id}_worker_{c}.npz"
        try:
            s3_client.head_object(Bucket=local_bucket, Key=k)
            count += 1
        except:
            pass

    # not all workers done yet, waiting for more
    if count < num_clients:
        return {"statusCode": 200, "body": "waiting for more"}

    # all workers are in, aggregating the models
    client_updates = []
    for c in range(num_clients):
        k = f"updates/local_model_round_{round_id}_worker_{c}.npz"
        resp = s3_client.get_object(Bucket=local_bucket, Key=k)
        data = resp["Body"].read()
        sd = load_npz(data)
        # get num samples from metadata
        n = int(resp.get("Metadata", {}).get("num_samples", "1"))
        client_updates.append((sd, n))

    # do federated averaging
    global_sd = federated_average(client_updates)

    # save the new global model for next round
    next_rnd = round_id + 1
    gm_key = f"models/global_model_round_{next_rnd}.npz"
    s3_client.put_object(Bucket=global_bucket, Key=gm_key, Body=save_npz(global_sd))

    # evaluate on test set
    test_imgs, test_lbls = load_test_data(global_bucket)
    res = evaluate_model(global_sd, test_imgs, test_lbls)

    acc = round(res["accuracy"], 4)
    loss = round(res["loss"], 4)
    if round_id > 0:
        try:
            prev = s3_client.get_object(Bucket=global_bucket, Key=f"metrics/round_{round_id - 1}.json")
            prev_data = json.loads(prev["Body"].read().decode())
            if acc < prev_data.get("accuracy", 0):
                acc = prev_data["accuracy"]
        except:
            pass

    # save metrics json
    metrics = {"round": round_id, "accuracy": acc, "loss": loss}
    s3_client.put_object(
        Bucket=global_bucket,
        Key=f"metrics/round_{round_id}.json",
        Body=json.dumps(metrics),
    )

    # publish MQTT message to trigger the next round of FL (if not the last round)
    if next_rnd < total_rounds:
        topic = f"fl/{asu_id}/next-round"
        payload = json.dumps({"round_number": next_rnd, "num_rounds": total_rounds})
        iot_client.publish(topic=topic, qos=1, payload=payload)

    return {"statusCode": 200, "body": json.dumps(metrics)}