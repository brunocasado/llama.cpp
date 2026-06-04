# TurboQuant fork implementation plan

> For Hermes: execute in phases; verify each phase with real code, real benchmarks, and real diffs.

Goal: add TurboQuant-style KV cache compression to a llama.cpp fork with first-class support for Windows CUDA and a secondary macOS/Metal path.

Architecture:
- introduce a KV codec abstraction inside the runtime memory layer
- build CPU reference first
- ship optimized CUDA path second
- wire benchmarking and source caching into the repo from day one

Tech stack:
- C++
- ggml / llama.cpp runtime
- CUDA C++ for production kernels
- optional Metal for macOS parity

---

## Phase 0: research freeze and fork hygiene

Objective: lock upstream baseline and freeze the requirements sources.

Files:
- existing fork root
- `docs/research/turboquant/tweet-qvac-2026-06-01.md`
- `docs/research/turboquant/architecture-notes.md`
- `docs/research/turboquant/sources.json`

Steps:
1. Keep upstream pinned at tag `b9509` until the codec seam is merged locally.
2. Do not copy GPL source into this fork.
3. Treat `AmesianX/TurboQuant` as the primary diff-mining repo.
4. Record provenance for each imported idea.

Verification:
- `git rev-parse --short HEAD`
- `git status --short`
- confirm docs exist under `docs/research/turboquant/`

## Phase 1: introduce KV codec abstraction

Objective: separate “runtime KV compression algorithm” from raw cache tensor dtype.

Files:
- Create: `src/llama-kv-codec.h`
- Create: `src/llama-kv-codec.cpp`
- Modify: `src/llama-kv-cache.h`
- Modify: `src/llama-kv-cache.cpp`
- Modify: `src/llama-model.cpp`
- Modify: `src/llama-context.cpp`
- Modify: `common/arg.cpp`

Steps:
1. Define codec interface for encode/decode/state/memory-accounting.
2. Implement `legacy` codec as a no-op baseline.
3. Thread codec selection through context creation.
4. Preserve legacy behavior when TurboQuant is disabled.

Verification:
- build CPU target successfully
- existing cache-type CLI behavior unchanged when codec is legacy

## Phase 2: CPU reference TurboQuant

Objective: create a correctness-first implementation independent of CUDA/Metal.

Files:
- Create: `src/llama-kv-codec-turboquant.h`
- Create: `src/llama-kv-codec-turboquant.cpp`
- Modify: `ggml/include/ggml.h` only if absolutely necessary
- Prefer avoiding permanent ggml type explosion in v1 of this fork

Steps:
1. Implement WHT/sign-flip + quantization metadata packing in pure C++.
2. Implement decode path and roundtrip tests.
3. Add session/state serialization versioning.
4. Add unit-style correctness tests for roundtrip and decode stability.

Verification:
- deterministic roundtrip tests pass
- short-context inference works with codec enabled on CPU

## Phase 3: Windows CUDA production backend

Objective: make the feature actually useful on RTX-class devices.

Files:
- Create or modify under `ggml/src/ggml-cuda/`
- likely touch: `convert.cu`, attention kernels, backend registration
- add CUDA-specific benchmark notes under `docs/research/turboquant/`

Steps:
1. Implement packed storage kernels.
2. Implement fused decode or compressed-attention read path where worthwhile.
3. Optimize decode for long-context steady-state generation.
4. Validate on Windows CUDA configurations first.

Verification:
- successful CUDA build on Windows
- benchmark CSV showing memory reduction and tok/s delta
- no correctness regression versus CPU reference beyond accepted tolerance

## Phase 4: benchmark integration

Objective: make claims falsifiable.

Files:
- Modify: `tools/llama-bench/llama-bench.cpp`
- Create: `docs/research/turboquant/benchmarks/README.md`
- Create: `docs/research/turboquant/benchmarks/*.csv`

Steps:
1. Add codec-aware benchmark flags.
2. Record memory footprint, TTFT, throughput, and context ceiling.
3. Add benchmark presets for 16K, 32K, 64K, 128K contexts.

Verification:
- `llama-bench` runs with codec off/on
- benchmark CSVs generated and readable

## Phase 5: macOS / Metal backend

Objective: functional parity on Apple Silicon after CUDA path is stable.

Files:
- touch `ggml/src/ggml-metal/` and related registration code

Steps:
1. implement correctness-first Metal path
2. benchmark on Apple Silicon
3. document limitations if decode overhead is too high

Verification:
- Metal build succeeds
- inference runs on Apple Silicon with codec enabled

## Definition of done
- source tweet cached locally
- architecture notes committed to repo
- codec abstraction exists
- CPU reference passes
- Windows CUDA path works
- benchmark evidence exists
- unsupported cases are documented explicitly
