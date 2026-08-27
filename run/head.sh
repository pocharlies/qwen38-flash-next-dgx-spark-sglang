#!/usr/bin/env bash
# Rank 0 (HEAD) — plain `docker run` path for a DGX Spark pair, no Kubernetes.
# Run this on the head Spark. Assumptions:
#   - Direct 200G DAC link between the two Sparks, RoCEv2, static IPs
#     10.0.0.2 (head, this box) / 10.0.0.1 (worker).
#   - The checkpoint snapshot is reachable at $HF_CACHE (the worker box holds
#     the single copy; mount it here over NFS).
#   - Image built from image/Dockerfile (+ Dockerfile.baked-patches -> v0.2.0).
# Boot order: start worker.sh on the other box FIRST (or within seconds of
# this one). A lone rank waits 601 s for the rendezvous and dies.
set -euo pipefail

IMAGE="${IMAGE:-REGISTRY/sglang-qwen38next-arm64:v0.2.0}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
SNAPSHOT="${SNAPSHOT:-$HF_CACHE/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594}"

docker run --rm --name qwen38-head \
  --gpus all --network host --ipc host \
  --cap-add IPC_LOCK --device /dev/infiniband \
  --shm-size 16g \
  -v "$HF_CACHE":/cache/huggingface:ro \
  -v "$HOME/.cache/sglang-flashinfer":/cache/flashinfer \
  -v "$HOME/.cache/sglang-tmp":/tmp \
  -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCHINDUCTOR_COMPILE_THREADS=4 -e MAX_JOBS=1 \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$SNAPSHOT" \
    --served-model-name qwen38-flash-next \
    --host 0.0.0.0 --port 8888 \
    --tp 2 --nnodes 2 --node-rank 0 --dist-init-addr 10.0.0.2:25100 \
    --trust-remote-code --quantization modelopt_fp4 \
    --fp4-gemm-backend flashinfer_cutlass \
    --page-size 64 \
    --mamba-scheduler-strategy extra_buffer --mamba-track-interval 64 \
    --max-mamba-cache-size 30 \
    --chunked-prefill-size 2048 \
    --max-running-requests 6 \
    --context-length 262144 \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 32 \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --enable-metrics
# NOTE on the FIRST EVER boot (cold JIT cache): add
#   --disable-flashinfer-autotune
# (autotune hangs with a cold cache) and expect up to ~60 min of weight load
# + CUDA graph capture. Warm boots take ~6-9 min.
# NOTE on tuning: mem-fraction 0.85 + mamba 30 = 6 concurrent / ~1M-token KV
# pool (long-agent profile). See the README table for the 8cc and 16cc
# trade-offs. Both ranks must use IDENTICAL engine parameters.
