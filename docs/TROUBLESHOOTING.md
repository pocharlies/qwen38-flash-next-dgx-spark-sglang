# Catálogo de fallos de arranque (todos vividos, con fix)

Cronología real de una noche de puesta en producción en 2× DGX Spark. Cada
fallo destapó el siguiente. Si tu arranque falla, busca aquí el síntoma.

## 1. OOMKilled (exit 137) a los 30–90 s de «Capture target verify CUDA graph begin»

**Síntoma**: el head (o el worker) muere justo al empezar la captura de CUDA
graphs. Los pesos cargaron bien, la KV está asignada.

**Causa**: en GB10 la memoria de GPU es memoria del sistema. El pool estático
(`mem-fraction-static` × ~120 GiB) está pinneado, y la compilación que dispara
la captura añade un pico host-side que cruza el límite.

**Fix**: `mem-fraction-static` 0.78 (no 0.85) + ver fallo 2.

## 2. El OOM es GLOBAL del nodo, no del cgroup

**Síntoma**: `dmesg` muestra `cicc invoked oom-killer` y el task dump lista
procesos de sistema (systemd, nfsd...). Cientos de procesos `cicc`.

**Causa**: el JIT de flashinfer compila los kernels fused_moe FP4 con nvcc vía
ninja **sin -j** → un compilador por core. **Un solo `cicc` de cutlass FP4
llega a 7,7 GB de RSS.** 20 en paralelo sobre un nodo con el pool pinneado =
OOM global. El límite de memoria del pod NO protege de esto.

**Fix**: `MAX_JOBS=1` (lo honra torch cpp_extension/ninja). La primera captura
es lenta (nvcc en serie) pero la caché JIT persiste en un hostPath y los
siguientes arranques no compilan. Dimensiona el startupProbe a 60 min.

## 3. Un rank clavado en `folio_wait_bit_common` (D-state, GPU 0 %)

**Síntoma**: un rank se queda >20 min entre «Mamba Cache is allocated» y la KV
(paso que tarda 6 s). `ps -eLo ... wchan` muestra el scheduler en estado D en
`folio_wait_bit_common`.

**Causa**: tormenta de reclaim del page cache — el nodo servía a la vez los
135 GB de pesos por NFS al otro rank y/o tenía I/O pesado concurrente.

**Fix**: no programar descargas/I/O masivo en los nodos durante el arranque.
Si ya está clavado, no se recupera solo: reinicio coordinado de ambos ranks.

## 4. «Timed out after 601 seconds waiting for clients. 1/2 clients joined»

**Síntoma**: el head muere con ese timeout del TCPStore; el worker muestra
`Broken pipe` contra el head. Suele pasar tras cualquier crash de un rank.

**Causa**: los dos ranks reiniciaron descompasados y el rendezvous quedó roto.

**Fix**: reinicio coordinado (Recreate de ambos a la vez). Diseña el deploy
para que nunca reinicie uno solo.

## 5. ImportError: undefined symbol `c10_cuda_check_implementation` (flash_attn)

**Síntoma**: la captura muere en `_resolve_flash_attn_varlen_func` con un
undefined symbol al importar `flash_attn_2_cuda*.so`.

**Causa**: el wheel de flash-attn (FA2) de la imagen está compilado contra
otro torch (ABI rota), y el `__init__.py` del paquete lo importa siempre —
tumba también el path cute (FA4) que es Python puro.

**Fix temporal**: stubear `flash_attn/__init__.py` (ver `QWEN_FA_STUB` en el
launch.sh). Fix real: reconstruir la imagen con flash-attn compilado contra su
torch.

## 6. MLIRError: «coord and shape of view are weakly congruent» (cutlass DSL)

**Síntoma**: con el stub anterior, la captura muere construyendo el kernel
varlen cute de FA4 en `nvidia_cutlass_dsl`.

**Causa**: el kernel cute no es compatible con las shapes que le pasa QSA en
este stack. Es la mina de sgl-project/sglang#36531 (QSA elige camino FA4
incompatible en SM120/121).

**Fix**: abrir el gate del decode trtllm de QSA a sm120+ (ver `QWEN_QSA_GATE`
en el launch.sh): `is_sm100_supported()` excluye a los GB10 pero el
`flashinfer.decode.trtllm_batch_decode_with_kv_cache` de la build sm_121a
funciona; con trtllm resuelto el backend nunca llama al varlen.

## 7. El autotune de flashinfer cuelga el PRIMER arranque

**Síntoma**: 20+ min tras el arranque, GPU al 0 %, scheduler en spin, sin
progreso de captura.

**Causa**: autotune + caché JIT fría.

**Fix**: `--disable-flashinfer-autotune` en el primer arranque; cuando la
caché está templada se puede re-activar (arranca limpio; no medimos mejora).

## 8. La primera petición tarda un minuto

No está caído: es warmup (shapes fuera de los graphs capturados, JIT
residual). La segunda petición ya va a velocidad normal. Si tu healthcheck
hace inferencia real, dale margen.

## Verificación de que todo fue bien

- `Load weight end ... avail mem=33-35 GB` por rank → el PLE quedó en FP8 (GO).
- `/health` 200 en el head, ambos ranks estables sin reinicios.
- `sglang:spec_accept_length` ~2,0-2,7 bajo carga (si es ~1,0 el MTP no
  especula).
- 8 concurrentes ≈ 150 tok/s agregado; single-stream ≈ 40.
