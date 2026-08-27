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
---

# Qwen3.8-Flash-Next NVFP4 on 2× NVIDIA DGX Spark (GB10) with SGLang — full recipe

A production-verified recipe (2026-08-27) for serving
[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
(125B MoE / 6B active + 51B n-gram PLE + 4B MTP, 135 GB) on **two NVIDIA DGX
Spark** machines with **TP=2 over RoCEv2** and SGLang. The RadixArk NVFP4
checkpoint is validated upstream only on GB300/B300; **this recipe is the
sm_121a (GB10) verification**.

## Measured performance (30-minute soak)

| Metric | Value |
|---|---|
| Single-stream | ~41–42 tok/s |
| 8 concurrent streams (aggregate) | **153 tok/s average** (139–166), no degradation |
| NEXTN speculative accept length | ~2.3 sustained (2.0–2.7) |
| Thermals under sustained load | 83–85 °C plateau on both nodes, recovery to <60 °C |
| Context | 262,144 tokens (native 256K, no YaRN) |
| Max concurrency | 8 (capped by the mamba cache at mem-fraction 0.78) |

## Hardware

- 2× NVIDIA DGX Spark (GB10, 128 GB unified LPDDR5x each, arm64).
- Direct 200G DAC link between the two, RoCEv2 (`NCCL_IB_*` in the manifest).
  No switch: a straight port-to-port cable with static IPs (10.0.0.1/10.0.0.2).
- The weights live on one node; the other reads them over NFS (optionally
  NFS-RDMA over the same link — we measured 5.5 GB/s).

## Software

- **Image**: `image/Dockerfile` — SGLang with the `qwen4_exp` build
  (PR [sgl-project/sglang#36497](https://github.com/sgl-project/sglang/pull/36497),
  branch `qwen4-main-squashed@73a2552`), `modelopt_fp4` quantization,
  flashinfer with NVFP4 cutlass kernels for `sm_121a`, arm64. The branch
  requires `cargo` (Rust extensions) — the Dockerfile installs rustup.
- **Deployment**: `k8s/qwen38-flash-next-nvfp4-sglang.yaml` — two Deployments
  (head rank 0 + worker rank 1), `hostNetwork` (the ranks talk over the RoCE
  fabric IPs), and a ConfigMap with the full `launch.sh`. Every parameter is
  commented in the yaml itself with the reasoning behind it.

## Critical parameters (the ones that cost blood)

| Parameter | Value | Why |
|---|---|---|
| `--mem-fraction-static` | **0.78** | With 0.85 the boot dies OOM during CUDA graph capture. On GB10 the GPU memory IS system memory: the pinned static pool plus the host-side compilation spike cross the cgroup limit (and with less margin, they take down the whole node). |
| `MAX_JOBS` | **1** | The flashinfer JIT compiles the FP4 fused_moe kernels with nvcc via ninja, which defaults to one job per core. **A single cutlass FP4 `cicc` reaches 7.7 GB of RSS**: 20 in parallel = global node OOM. With 1, the spike stays around ~8 GB. Only the first capture is affected: the JIT cache persists. |
| `TORCHINDUCTOR_COMPILE_THREADS` | 4 | Bounds inductor's compile workers during capture. |
| `--page-size` | 64 | Mandatory: QSA (sparse attention) works in 64-token micro-blocks. |
| `--mamba-scheduler-strategy extra_buffer` + `--mamba-track-interval 64` | — | Mandatory for the radix cache over the hybrid GDN state. |
| NEXTN MTP | steps=3, topk=1, draft=4 | Measured accept length ~2.3. Switchable with `ENABLE_MTP=false` without touching the image. |
| `--chunked-prefill-size` | 2048 | A long prefill sinks concurrent decodes on this hardware. |
| `--disable-flashinfer-autotune` | FIRST boot only | With a cold JIT cache the autotune hangs the startup (GPU at 0%, spinning). Once the cache is warm it can be re-enabled (we measured no performance difference, but it boots cleanly). |
| Pod memory limit | 112Gi out of ~120 GiB | Leaves air for the system. The OOM that matters here is the GLOBAL node one, not the cgroup one — no cgroup protects you from page cache plus unified memory. |

## The two hot patches (launch.sh)

The image built from the open branch ships two landmines that the ConfigMap's
`launch.sh` patches before `exec` (search for `QWEN_FA_STUB` and
`QWEN_QSA_GATE` in the yaml):

1. **flash-attn FA2 with a broken ABI** (unresolved symbol
   `c10_cuda_check_implementation` against the image's torch) whose
   `__init__.py` imports it unconditionally, killing the FA4 cute path too.
   The patch stubs the `__init__` so only the cute path survives.
2. **The FA4 cute path crashes anyway** (an MLIRError about shape congruence
   in nvidia_cutlass_dsl while building the varlen kernel). The real way out:
   QSA's sparse decode has a third path, **trtllm-gen via flashinfer**, gated
   behind `is_sm100_supported()`, which excludes GB10 (sm_121). The patch
   opens the gate to sm120+ — with trtllm resolved, the backend never calls
   the broken varlen. (Related: sgl-project/sglang#36531.)

Once the image is rebuilt with a flash-attn compiled against its torch and the
gate is fixed upstream, both patches become unnecessary.

## Memory go/no-go: the PLE

The n-gram PLE tables (51 GB) are FP8 on disk. The RadixArk model card says
they dequantize to BF16 on load (which would not fit); **measured on sm_121a:
they stay FP8** — ~74 GiB of weights per rank, with plenty of pool left for
the 262K context. Verify it on your boot: `Load weight end ... avail mem`
should leave ~33–35 GB.

## Operational gotchas

- **Both ranks always together**: if one restarts out of step, the other
  waits 601 s for the TCPStore rendezvous and dies. Restart both at once
  (coordinated Recreate).
- The first request after a boot takes ~10–60 s (warmup). It is not down.
- If the node serving the weights over NFS also does heavy I/O during boot,
  the other rank can get stuck in `folio_wait_bit_common` (a page-cache
  reclaim storm). Avoid large concurrent downloads.
- Metrics come with the `sglang:*` prefix (including
  `sglang:spec_accept_length` to watch the speculative decoding).

## Full troubleshooting

The catalogue of every failure mode we hit (with diagnosis and fix) is in
[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).
