#!/bin/bash

# =================================================================
# Raspberry Pi Greengrass + FL Worker Setup Script
# =================================================================
# This script sets up a Raspberry Pi as a Greengrass Core device
# and configures it to run the Federated Learning worker.
#
# BEFORE RUNNING: Export temporary AWS credentials (one-time only):
#   export AWS_ACCESS_KEY_ID='your_access_key'
#   export AWS_SECRET_ACCESS_KEY='your_secret_key'
#
# These credentials are ONLY used to provision the Greengrass
# certificates AND to configure boto3 access for ggc_user at runtime.
# After setup, the Pi authenticates via X.509 certs.
# =================================================================

# =================================================================
# 1. CONFIGURATION
# =================================================================
ASU_ID="1225850355"
THING_NAME="${ASU_ID}-fl-worker-pi-gg"
PARTITION_ID="9"

# =================================================================
# 2. Pre-flight: Verify AWS credentials are available
# =================================================================
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "================================================================================="
    echo "ERROR: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set."
    echo ""
    echo "Unlike EC2, a Raspberry Pi cannot have an IAM Role attached directly."
    echo "You need temporary credentials ONLY for this one-time Greengrass installation."
    echo "After setup, the Pi uses X.509 certificates for authentication."
    echo ""
    echo "Steps:"
    echo "  1. Go to IAM Console -> Users -> Create user (e.g., 'pi-installer')"
    echo "  2. Attach AdministratorAccess (or AWSIoTFullAccess + IAMFullAccess + S3FullAccess)"
    echo "  3. Security credentials -> Create access key -> Copy the keys"
    echo "  4. Run:"
    echo "     export AWS_ACCESS_KEY_ID='your_access_key'"
    echo "     export AWS_SECRET_ACCESS_KEY='your_secret_key'"
    echo "  5. Re-run this script"
    echo "  6. You can delete the IAM user after setup completes"
    echo "================================================================================="
    exit 1
fi

echo "AWS credentials detected. Proceeding with setup..."

# =================================================================
# 3. System Dependencies
# =================================================================
echo "Installing system dependencies..."
sudo apt update && sudo apt install -y default-jdk unzip python3-pip

sudo useradd --system --create-home ggc_user || echo "User ggc_user already exists, skipping."
sudo groupadd --system ggc_group || echo "Group ggc_group already exists, skipping."

# =================================================================
# 4. Python Packages
# =================================================================
# IMPORTANT: Install in TWO separate steps to avoid disk exhaustion.
#
# Step A: Install lightweight packages first (boto3, awsiotsdk, Pillow, numpy)
# Step B: Install CPU-only PyTorch (NOT the default which pulls 500MB+ of CUDA/GPU libs)
#
# Lesson learned: Installing torch without --index-url tries to download
# nvidia_cublas (542MB) and fills up the SD card with [Errno 28] No space left.
echo ""
echo "Installing Python packages (Step A: boto3, awsiotsdk, Pillow, numpy)..."
sudo pip3 install --no-cache-dir --break-system-packages \
    boto3 awsiotsdk Pillow numpy

echo ""
echo "Installing Python packages (Step B: CPU-only PyTorch — this may take a few minutes)..."
sudo pip3 install --no-cache-dir --break-system-packages \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# =================================================================
# 5. Configure AWS credentials for ggc_user
# =================================================================
# The Greengrass worker runs as ggc_user. boto3 needs to find credentials
# so it can access S3. We configure them directly in ~/.aws/credentials
# for ggc_user. This is the most reliable approach for a Pi environment
# where the Greengrass Token Exchange Service (TES) may not be available.
echo ""
echo "Configuring AWS credentials for ggc_user..."
sudo mkdir -p /home/ggc_user/.aws
sudo tee /home/ggc_user/.aws/credentials > /dev/null <<EOF
[default]
aws_access_key_id = ${AWS_ACCESS_KEY_ID}
aws_secret_access_key = ${AWS_SECRET_ACCESS_KEY}
EOF
sudo tee /home/ggc_user/.aws/config > /dev/null <<EOF
[default]
region = us-west-2
EOF
sudo chown -R ggc_user:ggc_group /home/ggc_user/.aws
sudo chmod 600 /home/ggc_user/.aws/credentials

# =================================================================
# 6. Path Compatibility Symlink
# =================================================================
# worker.py hardcodes /home/ubuntu/fl-client/data_cache/ as the
# data path. Instead of modifying the code, we create a symlink
# so /home/ubuntu/fl-client -> /home/sai/fl-client
echo ""
echo "Creating /home/ubuntu symlink for worker.py compatibility..."
sudo mkdir -p /home/ubuntu
sudo ln -sfn /home/sai/fl-client /home/ubuntu/fl-client

# =================================================================
# 7. Clean up macOS resource fork files (._*) from fl-client data
# =================================================================
# When data is transferred from a Mac via tar/scp, macOS silently embeds
# hidden "._" resource fork files alongside every image file. These are
# NOT valid PNG files and will cause PIL.UnidentifiedImageError at runtime.
echo ""
echo "Cleaning up macOS resource fork files from fl-client..."
JUNK_COUNT=$(find /home/sai/fl-client -name '._*' 2>/dev/null | wc -l)
if [ "$JUNK_COUNT" -gt 0 ]; then
    sudo find /home/sai/fl-client -name '._*' -delete
    echo "  Deleted $JUNK_COUNT macOS resource fork files."
else
    echo "  No macOS resource fork files found. All clean!"
fi

# =================================================================
# 8. Greengrass Core Installation
# =================================================================
echo ""
echo "Installing Greengrass Core..."
cd ~
curl -s https://d2s8p88vqu9w66.cloudfront.net/releases/greengrass-nucleus-latest.zip > gg.zip
unzip -o gg.zip -d GreengrassInstaller && rm gg.zip

# The --tes-role-name is set to 'worker-role' so the Pi inherits
# the same S3/IoT permissions as your EC2 instances at runtime.
sudo -E java -Droot="/greengrass/v2" -Dlog.store=FILE \
  -jar ./GreengrassInstaller/lib/Greengrass.jar \
  --aws-region us-west-2 \
  --thing-name ${THING_NAME} \
  --thing-group-name MyGreengrassCoreGroup \
  --thing-policy-name GreengrassV2IoTThingPolicy \
  --tes-role-name worker-role \
  --tes-role-alias-name worker-roleAlias \
  --component-default-user ggc_user:ggc_group \
  --provision true \
  --setup-system-service true \
  --deploy-dev-tools true

# =================================================================
# 9. Permissions & Directory Setup
# =================================================================
echo ""
echo "Setting up directories and permissions..."
USER_HOME=$HOME
mkdir -p $USER_HOME/greengrassv2/recipes
mkdir -p $USER_HOME/greengrassv2/artifacts/com.fl.Worker/1.0.0
sudo chmod 755 $USER_HOME
sudo chmod -R 755 $USER_HOME/.local || true
sudo chmod -R 755 $USER_HOME/greengrassv2
sudo chown -R $USER:$USER $USER_HOME/greengrassv2

# Give ggc_user read access to the data and worker script
sudo chmod -R 755 $USER_HOME/fl-client
sudo chmod 755 $USER_HOME/worker.py 2>/dev/null || true

echo ""
echo "================================================================================="
echo "Setup Complete!"
echo "  Thing Name:   ${THING_NAME}"
echo "  Partition:    ${PARTITION_ID} (client-${PARTITION_ID})"
echo "  Data Path:    /home/ubuntu/fl-client/data_cache/ (symlink -> /home/sai/fl-client/data_cache/)"
echo "  TES Role:     worker-role (same permissions as your EC2 instances)"
echo "  boto3 creds:  /home/ggc_user/.aws/credentials (for runtime S3 access)"
echo ""
echo "Next steps:"
echo "  1. Copy your recipe to:   ~/greengrassv2/recipes/com.fl.Worker-1.0.0.json"
echo "  2. Copy worker.py to:     ~/greengrassv2/artifacts/com.fl.Worker/1.0.0/worker.py"
echo "  3. Deploy:"
echo "     sudo /greengrass/v2/bin/greengrass-cli deployment create \\"
echo "       --recipeDir ~/greengrassv2/recipes \\"
echo "       --artifactDir ~/greengrassv2/artifacts \\"
echo "       --merge 'com.fl.Worker=1.0.0'"
echo "  4. Monitor logs:"
echo "     sudo tail -f /greengrass/v2/logs/com.fl.Worker.log"
echo ""
echo "To verify Greengrass is running:"
echo "  sudo systemctl status greengrass.service"
echo "================================================================================="
