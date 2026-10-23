# FlashInfer workspace startup sizing: serving evidence

This repository contains a bounded, AI-assisted engineering experiment on selecting a smaller FlashInfer workspace before vLLM memory profiling. It reuses existing workspace configuration, concrete-plan queries and the attributed range-bound arithmetic from FlashInfer PR #5236.

## Measured configuration and results

RTX 4060 Laptop (SM89, 8 GB), Qwen2.5-1.5B-Instruct BF16, vLLM 0.28.0, FlashInfer 0.6.16.post3 with the separately validated numerical fix from PR #4401. Both configurations use the same 65% GPU-memory budget, ordinary FA2, TP/PP/DCP=1, page size 16, max sequences 4, max batched tokens 512, max model length 16,512, prefix caching disabled, chunked prefill enabled and FULL_DECODE_ONLY Graph sizes [1,4].

| Metric | Default | Startup-selected |
| --- | ---: | ---: |
| Float workspace | 394 MiB | 32 MiB |
| Actual KV allocation | 1701 MiB | 2062.8125 MiB |
| KV token capacity | 62,208 | 75,440 (+21.27%) |
| M06 batch completion median | 13.577162 s | 12.846201 s (-5.38%) |
| M06 first-request completion median | 8.915071 s | 11.153288 s (+2.24 s) |
| Concurrent reference-task score | 4/16 | 4/16 |

The workload has four requests, each with 16,384 input tokens and exactly 128 output tokens. Four fresh processes run default → auto → auto → default, with two warmups and five formal batches per process. An earlier, separate round on the same workload saved 6.25%. Native admission metrics show earlier admission of the fourth request. Shorter workloads showed little or no batch-time improvement.

Quality batches use normal EOS and are separate from fixed-output performance batches. All 16 paired reference-task outputs retain identical text and tokens. The denominator includes repeated runs of three tasks in four scenarios. Both configurations have shared long-context errors; the original all-tasks-correct acceptance flag remains false.

## Inspect and recompute the saved evidence

Python 3.11+ and its standard library are sufficient. From the repository root:

```bash
python -B -X utf8 verify_evidence.py --source evidence/M06 --out replay-output
```

This reads frozen source/input hashes, committed results, guardian receipts, raw timing samples and reference outputs. It recomputes the M06 measurements and writes `replay-output/recomputed.json` and a Markdown report. It performs no new inference. The return code reports evidence integrity/recomputation; original experiment acceptance flags are preserved separately.

## GPU experiment reproduction

The measured GPU harness is included under `evidence/M06`. Its model, WSL interpreter, fixed numerical-fix overlay, compiled cache and earlier-stage dependencies are pinned to the original local installation. Re-running it on another machine requires adapting those environment paths and obtaining the same dependencies; the existing launchers describe the original Windows/WSL setup.

Frozen prompts, settings, raw results and the exact startup prototype are included. The current purpose is to discuss the sizing API and the smallest native vLLM integration before preparing a portable GPU integration patch. GPU re-running is optional for the initial discussion.

## Proposed integration and existing work

The policy chooses a supported range before native memory profiling so the recovered budget can flow into automatic KV allocation. A native integration should preserve explicit user settings, report fallback reasons, obtain sizing bounds from FlashInfer, and keep the serving workspace stable through Graph capture/replay.

- [Proposal](PROPOSAL.md): implementation contract, measured costs, and feedback requested.
- [Detailed support and fallback notes](docs/support-and-integration.md): current checks, frozen assumptions and proposed behavior.
- [M06 result report](docs/results-M06.md): full measured results and original acceptance status.
- [M05 result report](docs/results-M05.md): the earlier separate performance round.
- `evidence/M06/startup_worker.py` and `workspace_policy.py`: exact measured prototype.
- `results/recomputed.json`: recomputed data, file list and retained original flags.

The prototype's explicit-size precedence currently recognizes values differing from 394 MiB. An explicitly supplied 394 MiB value still enters automatic selection; provenance-aware handling is a proposed integration item. Unsupported startup selection falls back before native profiling. Allocation, profiling, Graph and execution failures propagate normally.

Related work: [FlashInfer #5236](https://github.com/flashinfer-ai/flashinfer/pull/5236), [vLLM #40383](https://github.com/vllm-project/vllm/pull/40383), and [vLLM #46883](https://github.com/vllm-project/vllm/pull/46883). The broad memory-sizing idea and range arithmetic are existing work. This repository adds serving integration evidence, capacity/admission measurements and explicit trade-offs.

## Attribution

See [NOTICE.md](NOTICE.md). The range-bound port and vendor headers are attributed to FlashInfer PR #5236, commit `fdf3c5142b27dc34b8c2cb1aa8fe8910add79486`; original headers are preserved. The FlashInfer Apache-2.0 license text is included under `LICENSES/FlashInfer-Apache-2.0.txt`. Prototype code, scripts and analysis were prepared with AI assistance; the results come from the saved local runs.
