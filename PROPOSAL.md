# Proposal: opt-in startup sizing of the native FlashInfer workspace in vLLM

## Motivation and measured behavior

On an RTX 4060 Laptop (SM89), Qwen2.5-1.5B-Instruct BF16, vLLM 0.28.0 and FlashInfer 0.6.16.post3 with the separately validated numerical fix from FlashInfer PR #4401, a source-locked startup policy selects a 32 MiB float workspace instead of the configured 394 MiB reservation. Native vLLM profiling and automatic KV allocation then increase KV capacity from 62,208 to 75,440 tokens (+21.27%) at the same 65% GPU-memory budget.

For four requests with 16,384 input tokens and exactly 128 output tokens each, two separate ABBA runs reduced batch completion time by 6.25% and 5.38%. In the latter run, pooled medians were 13.577162 s versus 12.846201 s. Admission traces show that the extra capacity admits the fourth request earlier. The first request completes about 2.24 s later, so the latency/throughput trade-off should be explicit.

Sixteen paired concurrent reference-task outputs retained identical text and token sequences. Both configurations scored 4/16 (repetitions of three tasks across four scenarios); the original all-tasks-correct acceptance flag remains false. Fixed-output performance batches and normal-EOS quality batches are recorded separately.

## Reused work and proposed increment

The prototype uses the existing workspace-size configuration, concrete `workspace_size()` queries, and an attributed host port of the range-bound allocator arithmetic from [FlashInfer PR #5236](https://github.com/flashinfer-ai/flashinfer/pull/5236), commit `fdf3c5142b27dc34b8c2cb1aa8fe8910add79486`. That proposal is closed and unmerged as checked on 2026-10-24. Its arithmetic remains explicitly attributed in the local prototype.

The increment is startup integration: size a supported route before memory profiling, return the recovered reservation to native KV allocation, and measure capacity/admission/end-to-end effects. The lifecycle work in [vLLM PR #46883](https://github.com/vllm-project/vllm/pull/46883) and the larger-workspace protection in [vLLM PR #40383](https://github.com/vllm-project/vllm/pull/40383) should inform the integration design.

## Contract for an upstream implementation

1. An explicit user workspace setting takes precedence, including a value equal to the historical default. The prototype currently recognizes nondefault values by number; explicit-default provenance is an integration item.
2. Automatic shrinking is opt-in and requires a supported route with a trustworthy service-range bound. Unsupported routes or unavailable bounds retain native allocation before memory profiling.
3. The sizing implementation should have a single source of truth in FlashInfer. vLLM supplies actual backend/model geometry, query/token/batch limits, dtype, page layout and Graph mode.
4. Apply the decision before memory profiling and Graph capture. Keep the serving workspace stable across capture/replay. A later route/signature change must trigger reinitialization and reprofiling, or stop the unsupported transition.
5. Record configured/selected bytes, source of selection, fallback reason and resulting KV capacity. Propagate allocation, Graph and execution failures normally.

## Prototype scope and reproduction

The measured configuration is SM89, ordinary causal FA2, BF16 Q/KV/output, 12 query heads, 2 KV heads, head dimension 128, page size 16, TP/PP/DCP=1, max sequences 4, max batched tokens 512, max model length 16,512, prefix cache off, chunked prefill on, FULL_DECODE_ONLY Graph sizes [1,4], and automatic KV allocation. The source and scheduler hashes are pinned. The Chinese proposal contains the exact runtime checks and frozen-route assumptions.

The local package includes a standard-library CPU evidence recomputation and a separately budgeted GPU replay using the frozen inputs. The current prototype uses a custom Worker and a process-local getter hook; an upstream patch should integrate with native lifecycle hooks after agreement on the API and supported scope.

Feedback requested: preferred range-sizing API, ownership of the persistent arena, compatibility with ongoing lifecycle work, and an acceptable initial support/test scope.

For the saved-evidence command and GPU environment prerequisites, see [README.md](README.md).
