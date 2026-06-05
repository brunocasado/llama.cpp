# TurboQuant fork audit — prefill + architecture gap check (2026-06-05)

Branch: `feature/turboquant-architecture`
Commit under test: `2eae61618`
Repo: `brunocasado/llama.cpp`
Reference repo compared: `AmesianX/TurboQuant`

## Executive summary

The current fork is now stable again for real chat/server workloads and the prefill regression is fixed.

However, the current `--kv-cache-codec turboquant` path is still an experimental compatibility path, not a full TurboQuant backend.

What exists today in this fork:
- codec selection and CLI plumbing
- resolved quantized KV types (`q4_0/q4_0` by default)
- resolved compressor policy (`direct` / `hadamard`)
- selective fast prefill path restored only for the turboquant codec
- legacy path restored to upstream-like behavior for real chat

What does NOT exist yet in this fork:
- runtime TBQ/TBQP encode path
- dedicated GGML TBQ/TBQP/AMX block types
- CUDA attention kernels that consume TBQ/TBQP blocks directly
- QJL correction path
- internal WHT/TBQ query preprocessing path
- Tensor Core / MMA TurboQuant kernels for Windows CUDA
- Metal-native TurboQuant kernels for macOS

Conclusion:
- the prefill regression is fixed
- the fork is usable again
- the current turboquant mode is a compatibility shim around quantized KV, not the full algorithm from the reference implementation

## What I fixed today

Files changed:
- `src/llama-kv-cache.cpp`
- `src/llama-kv-cache.h`
- `src/llama-kv-cache-dsa.cpp`
- `src/llama-kv-cache-dsa.h`
- `src/llama-kv-cache-iswa.cpp`
- `src/llama-kv-cache-iswa.h`
- `src/llama-memory-hybrid.cpp`
- `src/llama-memory-hybrid.h`
- `src/llama-memory-hybrid-iswa.cpp`
- `src/llama-memory-hybrid-iswa.h`
- `src/llama-model.cpp`

Behavioral fixes:
1. Restored the dense contiguous prefill write fast path only when codec=`turboquant`
2. Re-enabled direct prefill attention assembly only for the turboquant experimental path
3. Kept legacy quantized KV behavior on the upstream-safe path
4. Added an explicit runtime warning that current turboquant mode is not the full TurboQuant backend

## Prefill benchmarks

Model:
- `lmstudio-community/Qwen3-4B-GGUF`
- file: `Qwen3-4B-Q4_K_M.gguf`

Command family:
- `./build/bin/llama-bench -m <model> -ngl 999 -t 4 -r 1 -o jsonl ...`

Raw outputs saved at:
- `/tmp/turboquant-prefill-check.jsonl` (before fix)
- `/tmp/turboquant-prefill-check-after.jsonl` (after fix)

### Before the fix

- legacy ub512 p4096: `285.660660 tok/s`
- turbo q4 direct ub512 p4096: `220.637542 tok/s`
- turbo q4 hadamard ub512 p4096: `225.194429 tok/s`
- legacy ub4096 p4096: `285.987498 tok/s`
- turbo q4 direct ub4096 p4096: `226.714867 tok/s`

Turbo path was ~21–23% slower than legacy in prefill.

### After the fix

- legacy ub512 p4096: `285.835078 tok/s`
- turbo q4 direct ub512 p4096: `270.625825 tok/s`
- turbo q4 hadamard ub512 p4096: `270.652093 tok/s`
- legacy ub4096 p4096: `286.922552 tok/s`
- turbo q4 direct ub4096 p4096: `274.901749 tok/s`

Delta vs previous turbo results:
- turbo q4 direct ub512: `+22.66%`
- turbo q4 hadamard ub512: `+20.19%`
- turbo q4 direct ub4096: `+21.25%`

Remaining gap vs legacy after the fix:
- turbo q4 direct ub512: `-5.32%`
- turbo q4 hadamard ub512: `-5.31%`
- turbo q4 direct ub4096: `-4.19%`

Interpretation:
- the regression was real and predominantly in prefill
- the selective fast path recovered almost all of the lost throughput
- there is still a small residual gap, which is expected because this path still writes/reads ordinary quantized KV rather than real TBQ kernels

## Real workload validation

### Legacy real chat regression check

Server command class:
- Gemma 4 12B Instruct, q8_0 KV, flash-attn, real `/v1/chat/completions`

Artifacts:
- log: `/tmp/llama-fork-legacy-regression.log`

Observed:
- `/health` returned `{"status":"ok"}`
- `/v1/chat/completions` returned 200
- no `Compute error`
- log shows:
  - `attn_rot_k = 1`
  - `attn_rot_v = 1`
  - `server is listening`

### Turboquant smoke test

Server command class:
- Qwen3-4B GGUF with `--kv-cache-codec turboquant --turboquant-type-k q4_0 --turboquant-type-v q4_0 --kv-cache-compressor-k direct --kv-cache-compressor-v direct`

Artifacts:
- log: `/tmp/llama-fork-turboquant-qwen.log`

Observed:
- `/health` returned `{"status":"ok"}`
- `/v1/chat/completions` returned 200
- startup warning now says:
  - `KV cache codec turboquant does not provide runtime encode yet; using the experimental compatibility path ... (this is not the full TurboQuant backend)`

## Architecture comparison: current fork vs reference TurboQuant

## 1) What the reference repo has

Evidence from `AmesianX/TurboQuant`:
- `README.md:3` describes full TurboQuant as WHT + Lloyd-Max + QJL
- `README.md:60` states `attn_rot_k = 0 / attn_rot_v = 0` for the internally-WHT TBQ path
- `README.md:85` explains external Hadamard is duplicate rotation once TBQ/TBQP/AMX encoders exist
- `README.md:115-117` documents dedicated CUDA kernel work, including current `fattn-vec` path and future `fattn-mma`
- `turboquant/modified_files/ggml/include/ggml.h:432-453` adds dedicated GGML types:
  - `GGML_TYPE_TBQ3_0 ... GGML_TYPE_TBQP4_4`
  - `GGML_TYPE_AMX3_1`
  - `GGML_TYPE_AMXV3_1`
- `turboquant/modified_files/ggml/src/ggml-cuda/fattn-common.cuh:586+` contains TBQ/TBQP dequant and dot-product logic
- `.../fattn-common.cuh:659+` contains fused QJL score path for TBQP
- `.../ggml-cuda/template-instances/*tbq*.cu` contains many CUDA template instantiations for TBQ/TBQP attention combinations

So the reference implementation is not just "quantized KV + a Hadamard option".
It adds new data types, new encode/decode semantics, and direct attention kernels that understand those formats.

## 2) What this fork currently has

Evidence from this fork:
- `src/llama-kv-codec.cpp:65-92`
  - resolves `turboquant` to ordinary ggml types (`q4_0` by default)
  - resolves compressors (`hadamard` / `direct`)
  - `has_runtime_encode() == false`
- `src/llama-model.cpp:1991-1997`
  - explicitly warns that runtime encode does not exist yet
  - calls the current path an experimental compatibility path
- `src/llama-kv-cache.cpp:295-313`
  - still uses external `attn_rot_k` / `attn_rot_v` policy based on quantized types
- `src/llama-kv-cache.cpp:20-22` and `315-333`
  - generates Hadamard matrices on the host and applies them as generic rotation support
- there are no dedicated GGML TBQ/TBQP/AMX types in this fork
- there are no dedicated CUDA TBQ/TBQP template instances in this fork
- there is no QJL-specific encode/decode implementation in this fork

So today this fork behaves like:
- a codec seam
- a resolved quantized KV storage policy
- optional external Hadamard rotation
- optimized prefill plumbing for that experimental path

That is useful infrastructure, but it is still not the same thing as TurboQuant proper.

## 3) Biggest architecture gaps

### Gap A — no runtime encoder
Current fork:
- resolves storage type only
- writes via normal KV path

Needed for real TurboQuant:
- dedicated runtime encode from fp16/bf16 activations into TBQ/TBQP/AMX blocks
- per-head-dim specialization (64 / 128 / 256 / 576 etc.)

### Gap B — no dedicated GGML storage types
Current fork:
- uses `q4_0`, `q8_0`, etc.

Reference repo:
- uses dedicated `GGML_TYPE_TBQ*`, `GGML_TYPE_TBQP*`, `GGML_TYPE_AMX*`

Needed:
- add real block layouts to ggml
- plumb row-size / type traits / backend allocation support

### Gap C — no direct CUDA attention kernels for TBQ/TBQP
Current fork:
- attention kernels still consume normal ggml types
- no TBQ-specific dot product path

Reference repo:
- has specialized CUDA vector attention instances and QJL-aware score fusion

Needed for Windows CUDA focus:
- first target should be CUDA vec path for correctness
- second target should be CUDA MMA/Tensor Core path for throughput

### Gap D — external Hadamard is still part of the experimental path
Current fork:
- quantized turbo path still reports `attn_rot_k = 1`, `attn_rot_v = 1` in the smoke test

Reference repo:
- internally-WHT TBQ/TBQP/AMX path reports `attn_rot_k = 0`, `attn_rot_v = 0`

Implication:
- our current path is not yet using the encoder/kernel design where the transform is internalized

### Gap E — no Metal-native TBQ backend
Current fork:
- works on Metal because it still uses ordinary ggml quantized KV behavior
- no TBQ-specific Metal kernels

Needed for macOS:
- once semantics are correct on CPU/CUDA, add Metal read/dequant path if we want real TurboQuant on Apple Silicon

## Windows CUDA priority recommendation

If the goal is "mostly Windows CUDA, secondarily macOS", the right order is:

1. Implement real GGML TBQ/TBQP storage types
2. Implement runtime encode on CPU reference path first for correctness
3. Implement CUDA vec attention consumption of TBQ/TBQP blocks
4. Remove duplicate external `attn_rot_*` for internally-WHT types
5. Add QJL-corrected score path where applicable
6. Only after correctness is stable, port to CUDA MMA / Tensor Core path
7. After CUDA is solid, consider Metal-specific acceleration

## Immediate next step recommendation

Recommended next coding step:
- keep the current selective fast prefill fix
- do not pretend the shim is full TurboQuant
- start phase 4 as a real backend effort:
  - add real TBQ storage types to ggml
  - add a CPU reference encoder/decoder
  - validate numerical behavior on small head dimensions
  - then port to CUDA vec kernels

That gives us a clean path toward a real Windows CUDA implementation without destabilizing the working fork again.
