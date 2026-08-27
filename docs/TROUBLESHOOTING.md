# Boot failure catalogue (all of them lived through, with fixes)

The real chronology of one night bringing this up on 2× DGX Spark. Each
failure uncovered the next. If your boot fails, look up the symptom here.

## 1. OOMKilled (exit 137) 30–90 s after "Capture target verify CUDA graph begin"

**Symptom**: the head (or the worker) dies right as CUDA graph capture
starts. The weights loaded fine, the KV cache was allocated.

**Cause**: on GB10 the GPU memory is system memory. The static pool
(`mem-fraction-static` × ~120 GiB) is pinned, and the compilation triggered
by the capture adds a host-side spike that crosses the limit.

**Fix**: `mem-fraction-static` 0.78 (not 0.85) + see failure 2.

## 2. The OOM is GLOBAL to the node, not the cgroup

**Symptom**: `dmesg` shows `cicc invoked oom-killer` and the task dump lists
system processes (systemd, nfsd...). Hundreds of `cicc` processes.

**Cause**: the flashinfer JIT compiles the FP4 fused_moe kernels with nvcc
via ninja **without -j** → one compiler per core. **A single cutlass FP4
`cicc` reaches 7.7 GB of RSS.** 20 in parallel on a node with the pool pinned
= global OOM. The pod memory limit does NOT protect against this.

**Fix**: `MAX_JOBS=1` (honoured by torch cpp_extension/ninja). The first
capture is slow (serial nvcc) but the JIT cache persists on a hostPath and
subsequent boots do not compile. Size the startupProbe for 60 minutes.

## 3. One rank stuck in `folio_wait_bit_common` (D-state, GPU at 0%)

**Symptom**: a rank sits >20 min between "Mamba Cache is allocated" and the
KV allocation (a step that takes 6 s). `ps -eLo ... wchan` shows the
scheduler thread in D state at `folio_wait_bit_common`.

**Cause**: a page-cache reclaim storm — the node was simultaneously serving
the 135 GB of weights over NFS to the other rank and/or doing heavy
concurrent I/O.

**Fix**: do not schedule downloads/massive I/O on the nodes during boot. If
it is already wedged, it does not recover on its own: coordinated restart of
both ranks.

## 4. "Timed out after 601 seconds waiting for clients. 1/2 clients joined"

**Symptom**: the head dies with that TCPStore timeout; the worker shows
`Broken pipe` against the head. Usually happens after any single-rank crash.

**Cause**: the two ranks restarted out of step and the rendezvous broke.

**Fix**: coordinated restart (Recreate of both at once). Design the
deployment so a single rank never restarts alone.

## 5. ImportError: undefined symbol `c10_cuda_check_implementation` (flash_attn)

**Symptom**: the capture dies in `_resolve_flash_attn_varlen_func` with an
undefined symbol while importing `flash_attn_2_cuda*.so`.

**Cause**: the image's flash-attn (FA2) wheel is built against a different
torch (broken ABI), and the package `__init__.py` always imports it — which
also kills the cute path (FA4), the pure-Python one.

**Temporary fix**: stub `flash_attn/__init__.py` (see `QWEN_FA_STUB` in
launch.sh). Real fix: rebuild the image with a flash-attn compiled against
its torch.

## 6. MLIRError: "coord and shape of view are weakly congruent" (cutlass DSL)

**Symptom**: with the stub above, the capture dies building the FA4 cute
varlen kernel in `nvidia_cutlass_dsl`.

**Cause**: the cute kernel is incompatible with the shapes QSA feeds it on
this stack. This is the landmine of sgl-project/sglang#36531 (QSA picks an
incompatible FA4 path on SM120/121).

**Fix**: open QSA's trtllm decode gate to sm120+ (see `QWEN_QSA_GATE` in
launch.sh): `is_sm100_supported()` excludes GB10, but
`flashinfer.decode.trtllm_batch_decode_with_kv_cache` from the sm_121a build
works; with trtllm resolved the backend never calls the varlen.

## 7. The flashinfer autotune hangs the FIRST boot

**Symptom**: 20+ minutes after startup, GPU at 0%, scheduler spinning, no
capture progress.

**Cause**: autotune + cold JIT cache.

**Fix**: `--disable-flashinfer-autotune` on the first boot; once the cache is
warm it can be re-enabled (boots cleanly; we measured no performance gain).

## 8. The first request takes a minute

It is not down: it is warmup (shapes outside the captured graphs, residual
JIT). The second request runs at normal speed. If your healthcheck performs
real inference, give it slack.

## Verifying a good boot

- `Load weight end ... avail mem=33-35 GB` per rank → the PLE stayed FP8 (GO).
- `/health` returns 200 on the head, both ranks stable with no restarts.
- `sglang:spec_accept_length` at ~2.0–2.7 under load (if it is ~1.0, MTP is
  not speculating).
- 8 concurrent streams ≈ 150 tok/s aggregate; single-stream ≈ 40.
