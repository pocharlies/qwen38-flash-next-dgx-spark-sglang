# sglang-qwen38next-arm64 — SGLang for Qwen3.8-Flash-Next NVFP4 on NVIDIA DGX
# Spark (GB10 = sm_121a, aarch64). Target:
#   REGISTRY/sglang-qwen38next-arm64:v0.1.0
# which is the tag referenced by k8s/qwen38-flash-next-nvfp4-sglang.yaml.
#
# UPSTREAM STATE (checked 2026-08-26 with gh):
#   - Model support is PR sgl-project/sglang#36497 "Introduce Qwen 3.8 Flash
#     Next" — OPEN, in-repo branch `qwen4-main-squashed`, head 73a2552. NOT in
#     main yet: build from that branch and RE-PIN to the merge SHA once it
#     lands.
#   - The official cookbook is already merged (PR #36496, docs) pointing at
#     that PR.
#   - OPEN LANDMINES that affect this recipe directly (check their state
#     before the first boot and on every re-pin):
#       #36531  QSA picks an incompatible pip FA4 path on SM120
#               (direct neighbour of our sm_121: risk #1 — this recipe's
#               launch.sh works around it, see the QSA gate patch)
#       #36532  GDN state dtype conflict prefill/decode on SM120
#       #36514  feature: n-gram PLE offload — this IS the memory go/no-go
#               (PLE FP8 vs BF16 dequant); track it
#       #36537  thinking + qwen3_coder tool parser loops on token 0
#               -> the reason TOOL_CALL_PARSER is EMPTY in the manifest
#
# REQUIREMENTS THIS IMAGE MUST MEET:
#   1. The `qwen4_exp` architecture loads (PR #36497).
#   2. Quantization `modelopt_fp4` with the `flashinfer_cutlass` backend on
#      sm_121a. The RadixArk checkpoint is validated upstream only on
#      GB300/B300; sm_121 is OUR verification.
#   3. aarch64 (Grace). Build on a Spark or an arm64 runner; do not use
#      buildx+qemu for CUDA kernels.
#
# TODO before considering it final:
#   - Confirm the base image (candidates: the NGC DGX Spark playbook images,
#     or nvcr.io/nvidia/pytorch:<25.xx>-py3). Docker Hub lmsysorg/sglang has
#     NO aarch64 tags (checked 2026-08-26).
#   - flashinfer: wheel/build with NVFP4 CUTLASS kernels for sm_121a, and
#     mind #36531/#36532 (the QSA/GDN paths on SM12x).
#   - Run the smoke test at the end on a GB10 and keep the output.
#
# Build (on a GB10; it is CPU-bound but clones/downloads several GB):
#   docker build -t REGISTRY/sglang-qwen38next-arm64:v0.1.0 .
#   docker push REGISTRY/sglang-qwen38next-arm64:v0.1.0

# TODO: confirm the base tag (25.08-py3 is a reasonable placeholder: CUDA 13 +
# aarch64 torch; the Spark playbooks may ship a better one).
FROM nvcr.io/nvidia/pytorch:25.08-py3

# Head of PR #36497 (branch qwen4-main-squashed) as of 2026-08-26. RE-PIN to
# the merge SHA once the PR lands in main.
ARG SGLANG_COMMIT=73a255206f916366c8d26d4022f82ddfb0ab558d
ARG FLASHINFER_REF=TODO-pin

ENV TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_DISABLE_VERSION_CHECK=1

# The qwen4-main-squashed branch ships Rust extensions (/opt/sglang/rust) and
# its setup.py requires cargo to discover and build them (measured failure:
# "RuntimeError: cargo is required to discover the Rust extension modules").
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

RUN git clone https://github.com/sgl-project/sglang.git /opt/sglang \
    && cd /opt/sglang \
    && git checkout "${SGLANG_COMMIT}" \
    && pip install --no-cache-dir -e "python[all]"

# TODO: flashinfer with NVFP4 for sm_121a if the one sglang pulls in does not
# ship it.
# RUN pip install --no-cache-dir "flashinfer-python @ git+https://github.com/flashinfer-ai/flashinfer@${FLASHINFER_REF}"

# Build smoke test: if the architecture or the quant method is missing, the
# build dies here and a broken tag never reaches the registry. (The real FP4
# kernel smoke test needs a GPU: run it separately on a GB10.)
RUN python3 - <<'PY'
import sglang
print("sglang", sglang.__version__)
from sglang.srt.layers.quantization import QUANTIZATION_METHODS
assert "modelopt_fp4" in QUANTIZATION_METHODS, sorted(QUANTIZATION_METHODS)
print("modelopt_fp4 OK")
import importlib
# The checkpoint architecture (config.json architectures) must resolve in
# this build's model registry.
from sglang.srt.models.registry import ModelRegistry
archs = getattr(ModelRegistry, "models", None) or {}
assert any("qwen" in a.lower() and ("4" in a or "next" in a.lower()) for a in archs), sorted(archs)[:40]
print("qwen4/next model registry OK")
PY
