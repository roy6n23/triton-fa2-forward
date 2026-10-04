#!/usr/bin/env bash
# The GPU session on a cloud VM or bare-metal box where we have sudo, which is what Nsight Compute needs:
# it runs bench/gpu_session.sh inside the same vLLM 0.30.0 image as the RunPod runs, with CAP_SYS_ADMIN,
# so ncu may read the GPU's performance counters under the driver's default (admin-only) setting.
#
#   bash bench/vm_session.sh           # from the repo root on the VM; results land in bench/results/<date>/
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90   # v0.30.0

# 1. The image is built for CUDA 13.0, which needs a host driver >= 580.
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), driver $driver"
if (( ${driver%%.*} < 580 )); then
    echo "driver $driver < 580: the CUDA 13 image cannot run here; install a 580+ driver (reboot) first" >&2
    exit 1
fi

# 2. Docker and the NVIDIA container toolkit, if the VM image lacks them.
if ! command -v docker > /dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
fi
if ! sudo docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L > /dev/null 2>&1; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
fi

# 3. Inside the image: Nsight Compute from NVIDIA's devtools repo, pytest in a venv, then the session.
sudo docker pull -q "$IMAGE"
sudo docker run --rm --gpus all --cap-add SYS_ADMIN --ipc=host \
    -v "$PWD":/root/triton-fa2-forward -w /root/triton-fa2-forward --entrypoint bash "$IMAGE" -c '
set -e
apt-get update -qq > /dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends gnupg2 wget > /dev/null
. /etc/lsb-release
REPO=https://developer.download.nvidia.com/devtools/repos/ubuntu${DISTRIB_RELEASE//./}/amd64
wget -qO- $REPO/nvidia.pub | gpg --dearmor > /usr/share/keyrings/nvidia-devtools.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] $REPO/ /" > /etc/apt/sources.list.d/nvidia-devtools.list
apt-get update -qq > /dev/null
PKG=$(apt-cache search "^nsight-compute-20" | awk "{print \$1}" | sort -V | tail -1)
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$PKG" > /dev/null
ln -sf "$(ls -d /opt/nvidia/nsight-compute/*/ncu | sort -V | tail -1)" /usr/local/bin/ncu
python3 -m venv --system-site-packages /root/venv && /root/venv/bin/pip install -q pytest
. /root/venv/bin/activate
ncu --version | tail -1
bash bench/gpu_session.sh
'
sudo chown -R "$(id -u):$(id -g)" bench/results
