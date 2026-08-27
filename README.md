---
license: apache-2.0
base_model: RadixArk/Qwen3.8-Flash-Next-NVFP4
tags:
  - sglang
  - nvfp4
  - dgx-spark
  - gb10
  - tensor-parallel
  - speculative-decoding
  - deployment-recipe
language:
  - en
  - es
---

# Qwen3.8-Flash-Next NVFP4 en 2× NVIDIA DGX Spark (GB10) con SGLang — receta completa

Receta verificada en producción (27-08-2026) para servir
[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
(125B MoE / 6B activos + 51B PLE n-gram + 4B MTP, 135 GB) en **dos NVIDIA DGX
Spark** con **TP=2 sobre RoCEv2** y SGLang. El checkpoint NVFP4 de RadixArk
está validado por ellos solo en GB300/B300; **esta receta es la verificación
en sm_121a (GB10)**.

## Rendimiento medido (soak de 30 min)

| Métrica | Valor |
|---|---|
| Single-stream | ~41–42 tok/s |
| 8 streams concurrentes (agregado) | **153 tok/s de media** (139–166), sin degradación |
| Accept-length del especulativo NEXTN | ~2,3 sostenido (2,0–2,7) |
| Térmica bajo carga sostenida | meseta 83–85 °C en ambos nodos, recuperación <60 °C |
| Contexto | 262.144 tokens (256K nativo, sin YaRN) |
| Concurrencia máxima | 8 (capada por la mamba cache con mem-fraction 0.78) |

## Hardware

- 2× NVIDIA DGX Spark (GB10, 128 GB de memoria unificada LPDDR5x cada uno, arm64).
- Enlace directo DAC 200G entre los dos, RoCEv2 (`NCCL_IB_*` en el manifiesto).
  Sin switch: cable directo puerto a puerto y IPs estáticas (10.0.0.1/10.0.0.2).
- Los pesos viven en uno de los nodos; el otro los lee por NFS (opcionalmente
  NFS-RDMA sobre el mismo enlace).

## Software

- **Imagen**: `image/Dockerfile` — SGLang con la build `qwen4_exp`
  (PR [sgl-project/sglang#36497](https://github.com/sgl-project/sglang/pull/36497),
  rama `qwen4-main-squashed@73a2552`), quant `modelopt_fp4`, flashinfer con
  kernels NVFP4 cutlass para `sm_121a`, arm64. La rama exige `cargo` (extensiones
  Rust) — el Dockerfile instala rustup.
- **Despliegue**: `k8s/qwen38-flash-next-nvfp4-sglang.yaml` — dos Deployments
  (head rank 0 + worker rank 1), `hostNetwork` (los ranks se hablan por las IPs
  del fabric RoCE), ConfigMap con el `launch.sh` completo. Todos los parámetros
  van comentados en el propio yaml con el porqué.

## Parámetros críticos (los que costaron sangre)

| Parámetro | Valor | Por qué |
|---|---|---|
| `--mem-fraction-static` | **0.78** | Con 0.85 el arranque muere OOM en la captura de CUDA graphs. En GB10 la memoria de GPU ES la del sistema: el pool estático pinneado + el pico de compilación host-side cruzan el límite del cgroup (y con menos margen, tumban el nodo entero). |
| `MAX_JOBS` | **1** | El JIT de flashinfer compila los fused_moe FP4 con nvcc vía ninja, que por defecto usa todos los cores. **Un solo `cicc` de cutlass FP4 llega a 7,7 GB de RSS**: 20 en paralelo = OOM global del nodo. Con 1, el pico queda en ~8 GB. Solo afecta a la primera captura: la caché JIT persiste. |
| `TORCHINDUCTOR_COMPILE_THREADS` | 4 | Acota los workers de compilación de inductor durante la captura. |
| `--page-size` | 64 | Obligatorio: QSA (atención dispersa) trabaja en micro-bloques de 64. |
| `--mamba-scheduler-strategy extra_buffer` + `--mamba-track-interval 64` | — | Obligatorios para el radix cache sobre el estado GDN híbrido. |
| MTP NEXTN | steps=3, topk=1, draft=4 | Accept-length medido ~2,3. Conmutable con `ENABLE_MTP=false` sin tocar la imagen. |
| `--chunked-prefill-size` | 2048 | Un prefill largo hunde los decodes concurrentes en este hardware. |
| `--disable-flashinfer-autotune` | solo el PRIMER arranque | Con la JIT fría, el autotune cuelga el arranque (GPU 0%, spin). Con la caché templada se puede re-activar (no medimos diferencia de rendimiento, pero arranca limpio). |
| Límite de memoria del pod | 112Gi de ~120 GiB | Deja aire al sistema. El OOM que importa aquí es el GLOBAL del nodo, no el del cgroup — ningún cgroup te protege del page cache + memoria unificada. |

## Los dos parches en caliente (launch.sh)

La imagen construida desde la rama abierta trae dos minas que el `launch.sh`
del ConfigMap parchea antes del `exec` (buscar `QWEN_FA_STUB` y
`QWEN_QSA_GATE` en el yaml):

1. **flash-attn FA2 con ABI rota** (símbolo `c10_cuda_check_implementation` sin
   resolver contra el torch de la imagen) y su `__init__.py` lo importa
   incondicionalmente, tumbando también el path cute de FA4. El parche stubea
   el `__init__` para que solo viva el path cute.
2. **El path cute de FA4 revienta igualmente** (MLIRError de congruencia en
   nvidia_cutlass_dsl al construir el kernel varlen). La salida real: el decode
   disperso de QSA tiene un tercer path, **trtllm-gen vía flashinfer**, capado
   por `is_sm100_supported()` que excluye a los GB10 (sm_121). El parche abre
   el gate a sm120+ — con trtllm resuelto, el backend no llama nunca al varlen
   roto. (Relacionado: sgl-project/sglang#36531.)

Cuando la imagen se reconstruya con un flash-attn compilado contra su torch y
el gate corregido upstream, ambos parches sobran.

## Go/no-go de memoria: el PLE

Las tablas PLE n-gram (51 GB) van FP8 en disco. La ficha de RadixArk dice que
se decuantizan a BF16 al cargar (lo que no cabría); **medido en sm_121a: se
quedan en FP8** — ~74 GiB de pesos por rank, y sobra pool para 262K de
contexto. Verifícalo en tu arranque: `Load weight end ... avail mem` debe
dejar ~33-35 GB.

## Gotchas operativos

- **Los dos ranks siempre juntos**: si uno reinicia descompasado, el otro se
  queda 601 s esperando el rendezvous del TCPStore y muere. Reinicia ambos a
  la vez (Recreate coordinado).
- La primera petición tras un arranque tarda ~10-60 s (warmup). No está caído.
- Si el nodo que sirve los pesos por NFS hace además I/O pesado durante el
  arranque, el otro rank puede quedarse clavado en `folio_wait_bit_common`
  (tormenta de reclaim del page cache). Evita descargas grandes concurrentes.
- Las métricas salen con prefijo `sglang:*` (incluye `sglang:spec_accept_length`
  para vigilar el especulativo).

## Troubleshooting completo

El catálogo de todos los modos de fallo que nos comimos (con su diagnóstico y
fix) está en [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).
