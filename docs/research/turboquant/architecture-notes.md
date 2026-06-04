# TurboQuant fork architecture notes

Goal: build a serious llama.cpp fork that adds TurboQuant-class KV cache compression while staying maintainable against upstream.

Base upstream:
- repo: ggml-org/llama.cpp
- pinned release/tag: b9509
- pinned commit: 6f3a9f3de

## Executive recommendation
Implement TurboQuant as a KV codec layer in the llama.cpp memory subsystem.
Do not model it as “just another tensor dtype” unless you accept deep coupling and high rebase pain.

Priority order:
1. CPU reference codec for correctness
2. Windows/NVIDIA CUDA production backend
3. llama-bench + server integration
4. macOS/Metal backend

## Verified upstream integration points
- `src/llama-kv-cache.h`
- `src/llama-kv-cache.cpp`
- `src/llama-model.cpp`
- `src/llama-context.cpp`
- `common/arg.cpp`
- `common/common.cpp`
- `tools/llama-bench/llama-bench.cpp`
- `ggml/include/ggml.h`
- `ggml/src/ggml.c`
- `ggml/src/ggml-quants.c`
- `ggml/src/ggml-cuda/`
- future Metal path: `ggml/src/ggml-metal/`

## Why the memory subsystem is the right seam
Upstream creates runtime memory implementations through `llama_model::create_memory()`.
Depending on model architecture, the runtime memory object may be:
- plain KV cache
- SWA split KV cache
- DSA KV cache
- hybrid/recurrent memory

That means TurboQuant must plug into the memory path, not be bolted on after the fact.

## Current upstream constraints already relevant to TurboQuant
- KV cache is already parameterized by `type_k` and `type_v`.
- Quantized V cache requires flash attention.
- Shape divisibility is validated for quantized cache types.
- `llama-bench` already sweeps cache types, which is ideal for benchmark extension.

## Public reference implementations inspected
1. `AmesianX/TurboQuant` (MIT)
   - most directly relevant llama.cpp fork
   - useful for file touch points, CUDA kernel ideas, CLI shape, head-dim routing
   - high value as an architectural reference
2. `mitkox/vllm-turboquant` (Apache-2.0)
   - useful for CUDA serving and benchmark ideas
3. `0xSero/turboquant` (GPL-3.0)
   - algorithmically interesting but license-sensitive
   - use as a research reference, not copy source into an MIT-friendly fork

## Design recommendation: codec layer above raw dtype
Preferred model:
- keep existing `--cache-type-k/--cache-type-v` for legacy storage dtypes
- add a higher-level KV codec concept for algorithms that require transforms, metadata, and specialized decode behavior

Proposed public flags:
- `--kv-codec legacy|turboquant`
- `--kv-codec-k legacy|turboquant3`
- `--kv-codec-v legacy|turboquant2`
- `--kv-group-size 64|128`
- `--kv-prefill-codec on|off`
- `--kv-residual-window N`
- `--kv-importance-metric qjl|norm|none`

## Why Windows CUDA comes first
- upstream already ships Windows CUDA binaries every release
- long-context local usage is strongest on RTX-class devices
- CUDA is where fused encode/decode/attention kernels matter most
- best chance to prove the memory/speed tradeoff credibly

## Why macOS is second
Metal support is important, but it should reuse the same codec contract.
Do not let Metal-specific limitations define the whole abstraction.

## Risk register
- decode overhead may erase memory wins at short context
- prefill and decode likely need different optimization strategies
- session/state save/restore needs explicit versioning
- hybrid/SWA/recurrent memory paths complicate support
- flash-attention codepaths may need backend-specific specialization
- benchmark claims are meaningless without quality + throughput + latency together

## Minimum credible v1
- selectable TurboQuant mode
- CPU reference implementation
- Windows CUDA implementation
- benchmark output for memory, TTFT, throughput, and quality delta
- clear unsupported-model/backends matrix
- cached source material and provenance notes
