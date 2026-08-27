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
| `--page-size` | 64 | Mandatory: QSA (sparse attention) selects at micro-block level, 64-token blocks with a budget of 512 blocks / 2048 tokens per query (model card). |
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
