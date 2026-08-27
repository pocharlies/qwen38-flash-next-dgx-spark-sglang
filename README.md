---
# The recipe itself (docs, manifests, Dockerfile) is Apache-2.0. The model
# weights it deploys are under the Qwen Community License 1.0 — see the base
# model card.
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
| Concurrency / KV | tunable trade-off; shipping 8 concurrent / ~1.37M-token KV pool (see the config table) |

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
| `--mem-fraction-static` | **0.90** (shipping) | Steers the KV pool (see the config table). Early on, 0.85 OOM'd during CUDA graph capture — but that was the nvcc/inductor spike, now capped by `MAX_JOBS=1`; with it capped, 0.90 boots clean and 0.94 is the hard edge. On GB10 the GPU memory IS system memory: the pinned static pool plus the host-side compilation spike cross the cgroup limit (and with less margin, they take down the whole node). |
| `MAX_JOBS` | **1** | The flashinfer JIT compiles the FP4 fused_moe kernels with nvcc via ninja, which defaults to one job per core. **A single cutlass FP4 `cicc` reaches 7.7 GB of RSS**: 20 in parallel = global node OOM. With 1, the spike stays around ~8 GB. Only the first capture is affected: the JIT cache persists. |
| `TORCHINDUCTOR_COMPILE_THREADS` | 4 | Bounds inductor's compile workers during capture. |
| `--page-size` | 64 | Mandatory: QSA (sparse attention) selects at micro-block level, 64-token blocks with a budget of 512 blocks / 2048 tokens per query (model card). |
| `--mamba-scheduler-strategy extra_buffer` + `--mamba-track-interval 64` | — | Mandatory for the radix cache over the hybrid GDN state. |
| NEXTN MTP | steps=3, topk=1, draft=4 | Measured accept length ~2.3. Switchable with `ENABLE_MTP=false` without touching the image. |
| `--chunked-prefill-size` | 2048 | A long prefill sinks concurrent decodes on this hardware. |
| `--disable-flashinfer-autotune` | FIRST boot only | With a cold JIT cache the autotune hangs the startup (GPU at 0%, spinning). Once the cache is warm it can be re-enabled (we measured no performance difference, but it boots cleanly). |
| Pod memory limit | 112Gi out of ~120 GiB | Leaves air for the system. The OOM that matters here is the GLOBAL node one, not the cgroup one — no cgroup protects you from page cache plus unified memory. |

## Concurrency vs context: the real tuning axis (measured)

The KV cache of this model is almost free (12 KB/token/rank: only 12 of 48
layers carry KV -- the whole pool at fraction 0.78 was ~4.4 GB), while the
mamba/GDN state costs ~0.5 GB per concurrent request (5 slots/request with
NEXTN x ~0.1 GB/slot). The KV pool has no direct knob: it is the remainder
`fraction x 120 - weights(74) - mamba - graphs`, so you steer it with
`--mem-fraction-static` and `--max-mamba-cache-size` (always identical on
both ranks). Three configurations measured on the same pair of Sparks:

| Config | Concurrency | KV pool | Aggregate | Per stream | Node headroom |
|---|---|---|---|---|---|
| 0.78 / mamba auto | 8 | 365K tok | 153 tok/s | 19.7 | ~24 GB |
| 0.86 / mamba 80 | 16 | 737K tok | 160 tok/s | 10.2 | ~11 GB |
| 0.85 / mamba 30 | 6 | 1.02M tok | 124 tok/s | 22.8 | ~15 GB |
| **0.90 / mamba 40 (SHIPPING)** | **8** | **1.37M tok** | -- | -- | ~7 GB |
| 0.94 / mamba 40 (the edge, not recommended) | 8 | 1.70M tok | -- | -- | ~2 GB |

The **shipping config is 0.90 / 8 concurrent / ~1.37M-token KV pool** (~5 full
262K sessions in parallel) with flashinfer autotune ON and ~7 GB of node
headroom. It boots clean in ~8 min.

**0.90 is the ceiling with autotune enabled.** 0.94 does reach a ~1.7M KV pool
but leaves the node with 2 GB, which hangs the flashinfer autotune (no buffer
memory) for ~27 min and drags CUDA graph capture out to ~25 min; it only boots
with `--disable-flashinfer-autotune`. Not worth the fragility.

**Always check `max_total_num_tokens` in the boot log after touching either
knob**: one intermediate config left the KV pool at 67K tokens -- below the
262K context -- and long agent sessions silently stopped fitting.

Large-context concurrency on the 6cc/1M profile (prefix-cache-friendly
corpus, so treat prefill figures as upper bounds):

| Load | Total prompt tokens | Wall | Aggregate prefill |
|---|---|---|---|
| 6 x ~30K | 198K | 25.9 s | 7.7K tok/s |
| 4 x ~60K | 264K | 22.4 s | **11.8K tok/s** |
| 2 x ~120K | 264K | 34.9 s | 7.6K tok/s |

A cold 72K-token single prompt (no prefix reuse) prefills in ~29 s
(~2.5K tok/s).

## Why not vLLM (yet)

There is an [official vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)
for this model. We evaluated it and stayed on SGLang for this hardware:

- **The FP8 checkpoint is 172.8 GiB** (~86.4 GiB/rank on TP2). Our NVFP4
  checkpoint is ~74 GiB/rank -- and on 128 GB unified-memory boxes that
  12 GiB difference is the whole safety margin. vLLM does not load the
  RadixArk experts-only NVFP4 layout, so with vLLM you are forced onto FP8.
- **The PLE CPU-offload flag buys nothing on GB10**: `VLLM_PLE_CPU_OFFLOAD`
  moves the 51 GB n-gram table "to host RAM" -- on unified memory that IS
  the same LPDDR. It only helps on discrete-GPU systems (GB300/H200).
- **The recipe is validated single-node only** (GB300, 8x H200, MI355X) and
  n-gram embedding does not support pipeline parallelism; two-Spark TP over
  RoCE with the PLE would be unexplored territory again.
- Independent validation is welcome though: the vLLM recipe also disables
  flashinfer autotune, uses 3 speculative tokens, and hits the same
  mamba-cache capacity wall -- three of our hard-won conclusions, reproduced.

If an official NVFP4 checkpoint that vLLM loads ever ships, it is worth
re-evaluating: vLLM on these Sparks is otherwise well-trodden ground.

## Running without Kubernetes (plain docker run)

`run/head.sh` and `run/worker.sh` reproduce the whole deployment with plain
`docker run` on the two Sparks -- same image, same NCCL/RoCE environment,
same engine flags, with the boot-order and GID-resolution notes inline.

## The two former hot patches -- now baked into the image

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

Both patches are now BAKED into the image as build layers (v0.2.0, see
`image/Dockerfile.baked-patches`): the launch script is clean again. The
definitive fix is still upstream (a flash-attn wheel built against the
image's torch, and sglang accepting sm120 in the gate).

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
- Parsers verified in production: `--reasoning-parser qwen3` and
  `--tool-call-parser qwen3_coder` (the template emits XML tool calls).
  Reasoning arrives in `reasoning_content`, tool calls come structured, and
  sglang#36537 (thinking + qwen3_coder looping) did not reproduce on this
  build — tested with thinking, tools, both combined, and streaming.

## Sampling and thinking controls (from the official model card)

The model **thinks by default**, emitting `<think>\n...</think>\n\n` before
the final answer. Recommended sampling, per mode:

| Mode | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty |
|---|---|---|---|---|---|---|
| Thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Instruct (non-thinking) | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |

`presence_penalty` can be raised between 0 and 2 to curb endless repetition,
at the cost of occasional language mixing and a slight quality drop.

Thinking behaviour is controlled through the chat template (works through any
OpenAI-compatible gateway via `chat_template_kwargs`):

- `enable_thinking: false` — direct answers, no `<think>` block.
- `preserve_thinking` (default **true**) — keeps the thinking blocks of ALL
  previous turns in the prompt. This is deliberate: it maximises decision
  consistency in agents and radix/KV-cache reuse. Set it to `false` to keep
  only the latest turn's thinking.
- `reasoning_effort` — supported levels: `xhigh`, `medium`, `low`. Note from
  the card, confirmed by our agent experience: in multi-turn agentic tasks a
  LOWER effort does not necessarily lower total task time — weaker analysis
  causes failures and retries that cost more than the faster turns save.

For agentic workloads the card recommends generous output budgets (reasoning
up to 262,144 tokens and final response up to 131,072 where the serving stack
splits the two). Size your gateway's `max_output_tokens` accordingly — our
deployment uses a much smaller cap and it is a deliberate trade-off, not a
model limit.

## Extending context beyond 256K (YaRN)

We deliberately run at the native 262,144 (static YaRN penalises short
contexts, and on 2× GB10 there is no memory for the 1M KV pool anyway). If
you need more, the official recipe is to change `rope_parameters` inside
`text_config` in `config.json`:

```json
{
  "mrope_interleaved": true,
  "mrope_section": [11, 11, 10],
  "rope_type": "yarn",
  "rope_theta": 10000000,
  "partial_rotary_factor": 0.25,
  "factor": 4.0,
  "original_max_position_embeddings": 262144
}
```

Set `factor` to what you actually need (e.g. 2.0 for 512K): all open-source
frameworks implement *static* YaRN, so an oversized factor degrades short
prompts all the time.

## Vision and video

Attention/GDN/vision run in BF16 in this checkpoint, so the multimodal path
is intact. For hour-scale video the card recommends raising `longest_edge`
in `video_preprocessor_config.json` to `469762048` (≈224k video tokens);
see sglang PR #18467 for engine-side overrides. We have not benchmarked video
on the 2× Spark setup — budget KV accordingly before trying.

## Alternative: one Spark, GGUF

If you have a single DGX Spark (or want CPU offload of the 51 GB n-gram PLE),
the [Unsloth Dynamic GGUFs](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
run through their [llama.cpp PR ggml-org/llama.cpp#27742](https://github.com/ggml-org/llama.cpp/pull/27742)
(the n-gram embedding is exactly the kind of parameter mass that offloads
well — it is indexed, not scanned). That path trades our TP=2 throughput and
native FP4 experts for a one-box setup; this recipe is the full-quality,
full-context two-node route.

## Full troubleshooting

The catalogue of every failure mode we hit (with diagnosis and fix) is in
[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).
