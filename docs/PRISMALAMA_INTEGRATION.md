# prismallama.cpp — Prismalama Integration Guide

> This is the GGUF inference engine vendored into [Prismalama](https://github.com/pryzmatpl/prismalama)
> (the Ollama-compatible fork). Prismalama vendors this repo via `Makefile.sync`.
>
> Prismalama docs: see the prismalama repo for `CODEMAP.md`, `docs/DEVELOPER.md`,
> `docs/PRISMALAMA_PRINCIPLE.md`, and `docs/RUNTIME_DISPATCH.md`.

---

## How Prismalama vendors this repo

Prismalama's `Makefile.sync` pins a `FETCH_HEAD` commit SHA from this repo and syncs
the source tree into two locations:

| Source (this repo) | Destination (prismalama) | Purpose |
|--------------------|-------------------------|---------|
| `./` (root) | `llama/llama.cpp/` | C++ source for CGo bindings |
| `ggml/` | `ml/backend/ggml/ggml/` | GGML backend sources |

### Sync flow

```bash
# In the prismalama repo:
make -f Makefile.sync checkout       # Clone this repo at FETCH_HEAD
make -f Makefile.sync apply-patches  # Apply 32+ local patches
make -f Makefile.sync sync           # rsync to llama/llama.cpp/ + ml/backend/ggml/ggml/
```

### Local patches

Prismalama maintains 32+ local patches in `llama/patches/` (audited in `llama/patches/README.md`):

- **Model-specific (keep local):** pretokenizer, clip-unicode, solar-pro, DeepSeek regex, interleave MRoPE
- **Multi-vendor (keep local):** GPU discovery enhancements, NVML fallback, device sorting
- **Robustness (keep local):** harden exception registration, exit vs abort, skip large CUDA batches
- **Safe to upstream:** argsort, batch size hint, no-alloc mode, DXGI+PDH memory detection, MLA flash attention

---

## CGo bindings (in prismalama)

Prismalama wraps this C++ codebase via CGo in `llama/llama.go` (810 lines):

### Key APIs exposed

| API | C++ source | Go wrapper |
|-----|-----------|------------|
| Model loading | `llama_load_model_from_file` | `Model.LoadModelFromFile()` |
| Context creation | `llama_new_context_with_model` | `NewContext()` |
| Token decode | `llama_decode` | `Context.Decode()` |
| Tokenization | `llama_tokenize` / `llama_token_to_piece` | `Model.Tokenize()` / `TokenToPiece()` |
| Sampling | `common_sampler_*` | `SamplingContext.Sample()` / `Accept()` |
| KV cache | `llama_kv_cache_seq_*` | `Context.KvCacheSeq*()` |
| Vision | `mtmd_*` | `MtmdContext.MultimodalTokenize()` |
| Grammar | `llama_grammar_*` | `Grammar.Apply()` / `Accept()` |

### Device priority

CUDA > ROCm/HIP > Vulkan > Metal > CPU

---

## Streaming extensions (in prismalama)

Prismalama adds GGUF weight streaming on top of this engine:

| Component | File (prismalama) | Purpose |
|-----------|-------------------|---------|
| `InferenceStreamer` | `ml/streaming/inference.go` | Eval callback: load next block, evict current |
| `LayerMap` | `ml/streaming/layermap.go` | Parse GGUF tensor names into block boundaries |
| `BudgetTracker` | `ml/streaming/budget.go` | LRU residency budget |
| `Prefetcher` | `ml/streaming/prefetch.go` | Async I/O with concurrency bounding |
| `Streamer` | `ml/streaming/streamer.go` | High-level orchestrator |
| `LoadStreaming` | `ml/backend/ggml/ggml_streaming.go` | Layer-by-layer model loading |

Controlled by `OLLAMA_LAYER_STREAMING=1` and `OLLAMA_STREAMING_BUDGET` (default 4 GiB).

---

## Weight compression (in prismalama)

| Format | Ratio | Method |
|--------|-------|--------|
| BC4 | ~4:1 | 4×4 block quantization (GPU texture codec) |
| DCT | Variable | 2D discrete cosine transform + frequency quantization |

Source: `ml/weightimage/` in prismalama.

---

## Contributing

When contributing to this repo, be aware that changes will be synced into Prismalama
via `Makefile.sync`. Local patches may need rebasing after upstream changes.

See Prismalama's `llama/patches/README.md` for the patch audit table and bisect policy.
