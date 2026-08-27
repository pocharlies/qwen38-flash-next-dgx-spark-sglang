# sglang-qwen38next-arm64 — SGLang para Qwen3.8-Flash-Next NVFP4 en DGX Spark
# (GB10 = sm_121a, aarch64). Destino:
#   REGISTRY/sglang-qwen38next-arm64:v0.1.0
# que es el tag que referencia k8s-ai-pocharlies/k8s/qwen38-flash-next-nvfp4-sglang.yaml.
#
# ESTADO UPSTREAM (comprobado 26-08-2026 con gh):
#   - El soporte del modelo es el PR sgl-project/sglang#36497 "Introduce
#     Qwen 3.8 Flash Next" — ABIERTO, rama in-repo `qwen4-main-squashed`,
#     head 73a2552. NO esta en main: se construye desde esa rama y se
#     RE-PINNEA al SHA de merge cuando entre.
#   - Cookbook oficial ya mergeado (PR #36496, docs) apuntando a ese PR.
#   - MINAS ABIERTAS que nos afectan directamente (revisar estado antes del
#     primer arranque y en cada re-pin):
#       #36531  QSA elige un camino FA4 de pip incompatible en SM120
#               (vecino directo de nuestro sm_121: es el riesgo nº1)
#       #36532  conflicto de dtype del estado GDN prefill/decode en SM120
#       #36514  feature: offload del PLE n-gram — ES nuestro go/no-go de
#               memoria (PLE FP8 vs dequant a BF16); si avanza, seguirlo
#       #36537  thinking + tool parser qwen3_coder entra en bucle con token 0
#               -> razon por la que TOOL_CALL_PARSER va VACIO en el manifiesto
#
# REQUISITOS QUE ESTA IMAGEN TIENE QUE CUMPLIR (gate de la Fase 1a):
#   1. Arquitectura `qwen4_exp` cargando (PR #36497).
#   2. Quantization `modelopt_fp4` con backend `flashinfer_cutlass` en sm_121a.
#      El checkpoint RadixArk esta validado solo en GB300/B300; sm_121 es
#      NUESTRA verificacion. Precedente: el build de vLLM de DeepSeek ya
#      compila flashinfer/CuTeDSL para 12.1a en estos nodos.
#   3. aarch64 (Grace). Construir en un Spark o runner ARC arm64; nada de
#      buildx+qemu para kernels CUDA.
#
# TODO(fase 1a) antes de considerarla lista:
#   - Confirmar imagen base (candidatos: la de los playbooks DGX Spark de NGC,
#     o nvcr.io/nvidia/pytorch:<25.xx>-py3). En Docker Hub lmsysorg/sglang NO
#     hay tags aarch64 (comprobado 26-08).
#   - flashinfer: wheel/build con kernels NVFP4 CUTLASS para sm_121a, y ojo a
#     #36531/#36532 (los caminos QSA/GDN en SM12x).
#   - Ejecutar el smoke del final en un GB10 y guardar la salida junto al PR.
#
# Build (en un GB10; es CPU-bound pero clona/descarga varios GB — misma
# disciplina de ventana que cualquier descarga en esos nodos):
#   docker build -t REGISTRY/sglang-qwen38next-arm64:v0.1.0 .
#   docker push REGISTRY/sglang-qwen38next-arm64:v0.1.0

# TODO(fase 1a): confirmar tag base (25.08-py3 es placeholder razonable:
# CUDA 13 + torch aarch64; los playbooks de Spark pueden traer uno mejor).
FROM nvcr.io/nvidia/pytorch:25.08-py3

# Head del PR #36497 (rama qwen4-main-squashed) a 26-08-2026. RE-PINNEAR al
# SHA de merge cuando el PR entre en main.
ARG SGLANG_COMMIT=73a255206f916366c8d26d4022f82ddfb0ab558d
ARG FLASHINFER_REF=TODO-pinnear

ENV TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_DISABLE_VERSION_CHECK=1

# La rama qwen4-main-squashed trae extensiones Rust (/opt/sglang/rust) y su
# setup.py exige cargo para descubrirlas y compilarlas (fallo medido 26-08:
# "RuntimeError: cargo is required to discover the Rust extension modules").
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

RUN git clone https://github.com/sgl-project/sglang.git /opt/sglang \
    && cd /opt/sglang \
    && git checkout "${SGLANG_COMMIT}" \
    && pip install --no-cache-dir -e "python[all]"

# TODO(fase 1a): flashinfer con NVFP4 para sm_121a si el que arrastra sglang
# no lo trae (mismo camino que el build de vLLM de DeepSeek).
# RUN pip install --no-cache-dir "flashinfer-python @ git+https://github.com/flashinfer-ai/flashinfer@${FLASHINFER_REF}"

# Smoke de build: si la arquitectura o el quant method no estan, el build
# muere aqui y no llega un tag roto a harbor. (El smoke de kernels FP4 reales
# necesita GPU: ejecutarlo aparte en un GB10.)
RUN python3 - <<'PY'
import sglang
print("sglang", sglang.__version__)
from sglang.srt.layers.quantization import QUANTIZATION_METHODS
assert "modelopt_fp4" in QUANTIZATION_METHODS, sorted(QUANTIZATION_METHODS)
print("modelopt_fp4 OK")
import importlib
# La arquitectura del checkpoint (config.json architectures) tiene que
# resolver en el registry de modelos de esta build.
from sglang.srt.models.registry import ModelRegistry
archs = getattr(ModelRegistry, "models", None) or {}
assert any("qwen" in a.lower() and ("4" in a or "next" in a.lower()) for a in archs), sorted(archs)[:40]
print("registro de modelos con qwen4/next OK")
PY
