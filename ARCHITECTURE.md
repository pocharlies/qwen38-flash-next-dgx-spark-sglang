# ARCHITECTURE.md — qwen38-flash-next-dgx-spark-sglang

Receta pública (verificada el 27-08-2026) para servir `RadixArk/Qwen3.8-Flash-Next-NVFP4` en dos NVIDIA DGX Spark (GB10, `sm_121a`) con SGLang, TP=2 sobre RoCEv2. Contiene Dockerfiles, un manifiesto de Kubernetes de referencia, scripts `docker run` y un catálogo de fallos. Tronco: **`main`**. No es un servicio desplegable tal cual.

## Clientes y versiones
- Sin clientes propios. Licencia Apache-2.0 la receta; los pesos, Qwen Community License 1.0.
- Artefactos: `image/Dockerfile` (v0.1.0) y `image/Dockerfile.baked-patches` (v0.2.0, con los dos parches en caliente ya horneados), `k8s/qwen38-flash-next-nvfp4-sglang.yaml` (ConfigMap `qwen38-flash-next-sglang-launch` + Deployments `qwen38-flash-next-head`/`-worker` + Service `qwen38-flash-next`, ns `llm`), `run/head.sh` y `run/worker.sh`, `docs/TROUBLESHOOTING.md`.
- Mismo modelo y hardware que `pocharlies/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark-k8s` (vLLM).

## Dependencias (ambos sentidos)
- **De** la rama abierta de SGLang `qwen4-main-squashed` (PR sgl-project/sglang#36497, commit `73a2552`; hay que re-pinear al SHA del merge cuando entre), flashinfer con kernels NVFP4 para `sm_121a`, `modelopt_fp4`, rustup (la rama compila extensiones Rust) y el checkpoint de Hugging Face.
- **Producción real**: `k8s-ai-pocharlies/k8s/qwen38-flash-next-nvfp4-sglang.yaml` (imagen de Harbor `…/sglang-qwen38next-arm64:<tag>`, no el placeholder `REGISTRY/…` de aquí). Este repo es la versión pública saneada; ante diferencia manda `k8s-ai-pocharlies`.
- **Quién depende de este repo**: nadie en código.

## Stack
SGLang (build `qwen4_exp`), NEXTN/MTP (steps=3, topk=1, draft=4), `--page-size 64`, `--mamba-scheduler-strategy extra_buffer`, `--chunked-prefill-size 2048`, parsers `qwen3`/`qwen3_coder`, métricas con prefijo `sglang:*`. Python y bash; arm64. No se usa: vLLM (se evaluó; ver «Why not vLLM», fechado 27-08), YaRN (se corre a 262 144 nativo).

## Componentes compartidos (canónicos)
Ninguno aquí. Manifiestos y pines: `k8s-ai-pocharlies`; enrutado: `k8s-litellm-pocharlies`; perfil de cómputo y árbitro: `dgx-infra` (`services/dashboard/compute_mode.py`, `gpu_arbiter.py`).

## Cómo se construye aquí
- Imagen: `docker build` de `image/Dockerfile` y, encima, `Dockerfile.baked-patches` (v0.2.0). Los parches `QWEN_FA_STUB` y `QWEN_QSA_GATE` ya van en la imagen; el `launch.sh` queda limpio.
- Parámetros críticos y su razón están comentados en el YAML y tabulados en el README: `--mem-fraction-static` (0.90 es el techo con autotune), `MAX_JOBS=1` (un solo `cicc` de cutlass FP4 llega a 7,7 GB; en paralelo tumba el nodo), límite de memoria del pod 112Gi de ~120 GiB. Tras tocar cualquier perilla, comprobar `max_total_num_tokens` en el log de arranque.
- **Antes de crear cualquier carga en los Sparks (`gx10-ec3d`, `nvidia-dgx`) se lee el ConfigMap `gpu-arbiter-state` (ns `comfyui`)**: `kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}'`. Con `llm-tp` los Sparks son enteros del LLM; con `phase: switching` no se toca nada. Un build de imagen o una descarga de pesos (~135 GB) en esos nodos con el residente cargado ya tumbó un nodo por page cache (19-08): el límite del cgroup no protege. Las réplicas las posee `compute_mode.py`; nunca `kubectl scale` a mano.
- `run/head.sh` y `run/worker.sh` (`docker run`) son para quien replica la receta SIN Kubernetes; en el clúster de Dani los Sparks no tienen docker y todo va por Deployments.

## Tests y validaciones
No hay tests ni CI. Validación documentada: arranque limpio (~8 min), `max_total_num_tokens` por encima del contexto (262 144), probes con thinking/tools/streaming, y el soak de 30 min del README (cifras fechadas, no el estado actual).

## CI/CD y despliegue
Ninguno aquí. El pin de imagen y el manifiesto se despliegan desde `k8s-ai-pocharlies`; este repo no dispara nada.

## Decisiones y trampas
- Reiniciar ambos rangos a la vez (Recreate coordinado): un rango suelto espera 601 s al rendezvous de TCPStore y muere.
- `--disable-flashinfer-autotune` solo en el primer arranque con caché JIT fría.
- Dos «landmines» de la rama (FA2 con ABI roto; ruta FA4 cute que falla) están parcheadas en la imagen; el arreglo definitivo es upstream (#36531).
- **Posible obsolescencia**: el README concluye quedarse en SGLang (27-08), pero la descripción del LiteLLM de producción presenta el residente `qwen38-flash-next-head` con imagen `vllm/vllm-openai`. Qué motor corre hoy se mira en el clúster (`kubectl get pods -n llm`, imagen del head); este documento no lo afirma. Propuesta para el architect: marcar el README como histórico o fusionar con la receta vLLM.
