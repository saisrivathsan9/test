"""
LeNet-5 Classifier for Federated Learning on MNIST.

USE THIS FILE IN YOUR WORKER (EC2 instances — PyTorch available).

This file provides:
  - LeNet5 model class (PyTorch)
  - Serialization helpers: state_dict <-> .npz bytes
  - Model creation and loading utilities

"""

import io
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import boto3
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
import awsiot.greengrasscoreipc as ipc
from awsiot.greengrasscoreipc.model import ( SubscribeToIoTCoreRequest, QOS )

NUM_CLASSES = 10

class LeNet5(nn.Module):
    """LeNet-5 for MNIST classification.

    Input:  (batch, 1, 28, 28)
    Output: (batch, 10)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_model(num_classes=NUM_CLASSES):
    """Create a fresh LeNet-5 model with random weights."""
    return LeNet5(num_classes=num_classes)


def load_model(state_dict, num_classes=NUM_CLASSES):
    """Create a LeNet-5 model and load the given state_dict.

    Args:
        state_dict: OrderedDict of PyTorch tensors (from deserialize_state_dict).
        num_classes: Number of output classes (default 10).

    Returns:
        LeNet5 model with loaded weights, ready for training or inference.
    """
    model = LeNet5(num_classes=num_classes)
    model.load_state_dict(state_dict)
    return model


def serialize_state_dict(state_dict):
    """Convert a PyTorch state_dict to .npz bytes for S3 upload.

    Args:
        state_dict: OrderedDict from model.state_dict()
                    (keys are layer names, values are torch.Tensor)

    Returns:
        bytes — the .npz archive contents, ready for s3.put_object(Body=...)

    Example:
        sd = model.state_dict()
        data = serialize_state_dict(sd)
        s3.put_object(Bucket=bucket, Key="models/global_model_round_0.npz", Body=data)
    """
    buf = io.BytesIO()
    np.savez(buf, **{k: v.cpu().numpy() for k, v in state_dict.items()})
    return buf.getvalue()


def deserialize_state_dict(data):
    """Convert .npz bytes from S3 to a PyTorch state_dict.

    Args:
        data: bytes — raw .npz file content from s3.get_object()["Body"].read()

    Returns:
        OrderedDict of torch.Tensor — ready for model.load_state_dict() or load_model()

    Example:
        resp = s3.get_object(Bucket=bucket, Key="models/global_model_round_0.npz")
        sd = deserialize_state_dict(resp["Body"].read())
        model = load_model(sd)
    """
    npz = np.load(io.BytesIO(data))
    return OrderedDict({k: torch.from_numpy(npz[k]) for k in npz.files})


# ============================================================================
# TODO: Implement your worker below
# ============================================================================

# training config stuff
NUM_ROUNDS = 5
NUM_EPOCHS = 1
BATCH_SIZE = 32
LR = 0.03

def train_local(model, dataloader, lr, epochs):
    """Train the model locally and return metrics.

    Args:
        model: LeNet5 model to train
        dataloader: PyTorch DataLoader with training data
        lr: learning rate
        epochs: number of local training epochs

    Returns:
        dict with keys:
            "train_loss": float — average training loss
            "train_accuracy": float — average training accuracy
            "num_samples": int — number of training samples
    """
    # basic cross entropy and sgd with momentum
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    # just loop through epochs and batches
    for e in range(epochs):
        for imgs, labels in dataloader:
            out = model(imgs)
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)

    return {
        "train_loss": total_loss / total,
        "train_accuracy": correct / total,
        "num_samples": total // epochs,
    }

def load_partition(pid):
    # load images and labels for this worker's partition
    data_dir = f"/home/ubuntu/fl-client/data_cache/client-{pid}"
    labels_path = "/home/ubuntu/fl-client/data_cache/labels.csv"

    # read labels csv into a dict
    label_map = {}
    with open(labels_path) as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",")
            label_map[parts[0]] = int(parts[2])

    imgs = []
    lbls = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".png"):
            continue
        img = Image.open(os.path.join(data_dir, fname)).convert("L").resize((28, 28))
        arr = np.array(img, dtype=np.float32) / 255.0
        # normalize with mnist mean and std
        arr = (arr - 0.1307) / 0.3081
        imgs.append(arr)
        lbls.append(label_map[fname])

    x = torch.tensor(np.array(imgs), dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(lbls, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True), len(imgs)

def wait_for_global_model(s3, bucket, key):
    # keep checking s3 until the global model shows up
    while True:
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return
        except:
            time.sleep(0.5)

# handler class for mqtt messages from greengrass
class RoundHandler(ipc.client.SubscribeToIoTCoreStreamHandler):
    def __init__(self, pid, s3, global_bucket, local_bucket, loader, n_samples):
        super().__init__()
        self.pid = pid
        self.s3 = s3
        self.global_bucket = global_bucket
        self.local_bucket = local_bucket
        self.loader = loader
        self.n_samples = n_samples

    def on_stream_event(self, event):
        # parse the mqtt message to get round number
        msg = json.loads(event.message.payload.decode())
        rnd = msg.get("round_number", 0)
        print(f"got mqtt message for round {rnd}")

        # wait for the global model to appear in s3
        gm_key = f"models/global_model_round_{rnd}.npz"
        wait_for_global_model(self.s3, self.global_bucket, gm_key)

        # download and load it
        resp = self.s3.get_object(Bucket=self.global_bucket, Key=gm_key)
        model = load_model(deserialize_state_dict(resp["Body"].read()))

        # train on our local data
        result = train_local(model, self.loader, LR, NUM_EPOCHS)
        print(f"round {rnd} done, loss={result['train_loss']:.4f}")

        # upload our trained model to local bucket (this triggers the lambda)
        lm_key = f"updates/local_model_round_{rnd}_worker_{self.pid}.npz"
        self.s3.put_object(
            Bucket=self.local_bucket,
            Key=lm_key,
            Body=serialize_state_dict(model.state_dict()),
            Metadata={"num_samples": str(self.n_samples)},
        )

    def on_stream_error(self, error):
        print(f"stream error: {error}")
        return False

    def on_stream_closed(self):
        print("stream closed")

def worker_main():
    """FL worker main loop.

    This function runs on each EC2 instance. You need to:

    1. Read PARTITION_ID and ASU_ID from environment variables
    2. Set up boto3 S3 client
    3. Load your MNIST partition from local disk
       (data is at /home/ubuntu/fl-client/data_cache/client-{PARTITION_ID}/)
    4. For each round:
       a. Poll S3 for global model: models/global_model_round_{R}.npz
       b. Download and deserialize the global model
       c. Train locally on your partition
       d. Upload trained model .npz to local-bucket (TRIGGERS Lambda)
          Key: updates/local_model_round_{R}_worker_{C}.npz
    """
    # get worker id and asu id from args
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    asu_id = sys.argv[2] if len(sys.argv) > 2 else "1224208336"

    global_bucket = f"{asu_id}-global-bucket"
    local_bucket = f"{asu_id}-local-bucket"
    topic = f"fl/{asu_id}/next-round"

    s3 = boto3.client("s3", region_name="us-west-2")

    # load our data partition
    loader, n_samples = load_partition(pid)

    # connect to greengrass ipc and subscribe to mqtt topic
    ipc_client = ipc.connect()

    req = SubscribeToIoTCoreRequest()
    req.topic_name = topic
    req.qos = QOS.AT_LEAST_ONCE

    handler = RoundHandler(pid, s3, global_bucket, local_bucket, loader, n_samples)
    op = ipc_client.new_subscribe_to_iot_core(handler)
    op.activate(req)
    op.get_response().result(timeout=10)
    print(f"subscribed to {topic}, waiting for rounds...")

    # just keep running until greengrass stops us
    while True:
        time.sleep(1)

if __name__ == "__main__":
    worker_main()