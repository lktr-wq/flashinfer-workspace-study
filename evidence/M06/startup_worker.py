"""Isolated vLLM worker: choose a bounded FlashInfer arena before profiling.

Only CPU numbers/JSON records survive the startup query.  The native builder
continues to own its arena and wrappers, including the separate profiling and
serving lifetimes.  Import this module only in an explicitly launched worker.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import torch
import vllm.envs as envs
from vllm.config import get_layers_from_vllm_config, set_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.v1.worker.gpu_worker import Worker


MIB = 1 << 20
NATIVE_DEFAULT_BYTES = 394 * MIB
INT_ARENA_BYTES = 8 * MIB
SELECTOR_BYTES = 256


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _save_trace(report):
    target = os.environ.get("FI_STARTUP_WORKSPACE_TRACE")
    if not target:
        return
    path = Path(target)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                        allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    except OSError as error:
        report["worker_health"]["trace_write_error"] = type(error).__name__ + ": " + str(error)


def _dtype_name(value):
    return str(value).removeprefix("torch.")


def _source_record(query_method, policy=None):
    policy = policy or {}
    verification = policy.get("source_verification", {})
    upper_hash = policy.get("upper_bound_source_sha256")
    if not upper_hash and isinstance(verification, dict):
        upper_hash = verification.get("upper_bound_source_sha256")
    return {
        "query_method": query_method,
        "upper_bound_source_sha256": upper_hash,
        "selector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_verification": verification,
    }


def _builder_signature(builder):
    """Fields that must match between profiling and serving builders."""
    return {
        "builder_class": type(builder).__module__ + "." + type(builder).__name__,
        "num_qo_heads": int(builder.num_qo_heads),
        "num_kv_heads": int(builder.num_kv_heads),
        "head_dim": int(builder.head_dim),
        "page_size": int(builder.page_size),
        "max_num_seqs": int(builder.max_num_reqs),
        "max_num_batched_tokens": int(builder.max_num_batched_tokens),
        "max_model_len": int(builder.model_config.max_model_len),
        "q_dtype": _dtype_name(builder.q_data_type_prefill),
        "decode_q_dtype": _dtype_name(builder.q_data_type_decode),
        "kv_dtype": _dtype_name(builder.kv_cache_dtype),
        "o_dtype": _dtype_name(builder.model_config.dtype),
        "prefill_fixed_split_size": int(builder.prefill_fixed_split_size),
        "decode_fixed_split_size": int(builder.decode_fixed_split_size),
        "disable_split_kv": bool(builder.disable_split_kv),
        "has_sinks": bool(builder.has_sinks),
        "sliding_window": int(builder.window_left),
        "logits_soft_cap": float(builder.logits_soft_cap or 0.0),
        "use_dcp": bool(builder.use_dcp),
        "trtllm_decode": bool(builder.use_trtllm_decode_attention),
        "enable_cuda_graph": bool(builder.enable_cuda_graph),
    }


def _native_bytes(builder, configured_bytes):
    if envs.VLLM_BATCH_INVARIANT:
        from vllm.v1.attention.backends.flashinfer import (
            FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT,
        )
        return int(FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT)
    return max(configured_bytes, int(builder.max_num_batched_tokens)
               * int(builder.num_qo_heads) * int(builder.head_dim) * 16)


def _extract_config(worker, builder):
    properties = torch.cuda.get_device_properties(worker.device)
    signature = _builder_signature(builder)
    config = {
        "backend": "FLASHINFER",
        "device_capability": (int(properties.major), int(properties.minor)),
        "num_sms": int(properties.multi_processor_count),
        **{key: signature[key] for key in (
            "num_qo_heads", "num_kv_heads", "head_dim", "page_size",
            "max_num_seqs", "max_num_batched_tokens", "max_model_len",
            "q_dtype", "kv_dtype", "o_dtype", "enable_cuda_graph",
            "disable_split_kv", "has_sinks", "sliding_window", "logits_soft_cap",
        )},
        "tensor_parallel_size": int(worker.parallel_config.tensor_parallel_size),
        "decode_context_parallel_size": int(builder.dcp_world_size),
        "fixed_split_size": int(builder.prefill_fixed_split_size),
        "num_colocated_ctas": 0,
        "use_cascade": False,
        "has_custom_mask": bool(worker.vllm_config.attention_config.use_non_causal),
        "pos_encoding_mode": "NONE",
    }
    if signature["decode_q_dtype"] != signature["q_dtype"]:
        raise ValueError("prefill/decode query dtypes differ")
    if signature["decode_fixed_split_size"] != signature["prefill_fixed_split_size"]:
        raise ValueError("prefill/decode split settings differ")
    if signature["trtllm_decode"] or signature["use_dcp"]:
        raise ValueError("native FA2 without DCP is required")
    if worker.vllm_config.speculative_config is not None:
        raise ValueError("speculative decoding is outside the startup policy")
    if worker.parallel_config.pipeline_parallel_size != 1:
        raise ValueError("pipeline parallelism is outside the startup policy")
    return config, signature


def _metadata(query_lengths, kv_lengths, page_size):
    query_indptr, page_indptr = [0], [0]
    for query_length, kv_length in zip(query_lengths, kv_lengths):
        query_indptr.append(query_indptr[-1] + int(query_length))
        page_indptr.append(page_indptr[-1] + (int(kv_length) + page_size - 1) // page_size)
    return (
        torch.tensor(query_indptr, dtype=torch.int32, device="cpu"),
        torch.tensor(page_indptr, dtype=torch.int32, device="cpu"),
        torch.zeros(page_indptr[-1], dtype=torch.int32, device="cpu"),
        torch.tensor([(length - 1) % page_size + 1 for length in kv_lengths],
                     dtype=torch.int32, device="cpu"),
    )


def _run_startup_queries(builder, policy):
    """Check concrete boundary plans; the policy supplies whole-range coverage."""
    from flashinfer import (BatchDecodeWithPagedKVCacheWrapper,
                            BatchPrefillWithPagedKVCacheWrapper)
    from vllm.v1.attention.backends.flashinfer import get_kv_cache_layout

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("workspace demand queries must run before graph capture")
    max_batch = int(builder.max_num_reqs)
    max_tokens = int(builder.max_num_batched_tokens)
    max_context = int(builder.model_config.max_model_len)
    page_size = int(builder.page_size)
    float_bound, int_bound = int(policy["float_bound_bytes"]), int(policy["int_bound_bytes"])
    if int_bound > INT_ARENA_BYTES:
        raise ValueError("computed integer arena bound exceeds native wrapper's 8 MiB")
    rows = []
    selector = torch.empty(SELECTOR_BYTES, dtype=torch.uint8, device=builder.device)
    layout = get_kv_cache_layout()
    common = {
        "num_qo_heads": int(builder.num_qo_heads),
        "num_kv_heads": int(builder.num_kv_heads),
        "page_size": page_size,
        "pos_encoding_mode": "NONE",
        "window_left": int(builder.window_left),
        "logits_soft_cap": builder.logits_soft_cap,
        "q_data_type": builder.q_data_type_prefill,
        "kv_data_type": builder.kv_cache_dtype,
        "o_data_type": builder.model_config.dtype,
        "sm_scale": builder.sm_scale,
        "fixed_split_size": int(builder.prefill_fixed_split_size),
        "disable_split_kv": bool(builder.disable_split_kv),
    }

    def check(kind, query_lengths, kv_lengths, graph):
        qo, indptr, indices, last = _metadata(query_lengths, kv_lengths, page_size)
        wrapper = None
        try:
            if kind == "prefill":
                wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    selector, layout, use_cuda_graph=False, backend="auto")
                float_bytes, int_bytes = wrapper.workspace_size(
                    qo, indptr, indices, last, head_dim_qk=int(builder.head_dim),
                    causal=True, **common)
            else:
                graph_buffers = {}
                if graph:
                    graph_buffers = {
                        "paged_kv_indptr_buffer": indptr.to(builder.device),
                        "paged_kv_indices_buffer": indices.to(builder.device),
                        "paged_kv_last_page_len_buffer": last.to(builder.device),
                    }
                wrapper = BatchDecodeWithPagedKVCacheWrapper(
                    selector, layout, use_cuda_graph=graph, use_tensor_cores=True,
                    backend="auto", **graph_buffers)
                float_bytes, int_bytes = wrapper.workspace_size(
                    indptr, indices, last, head_dim=int(builder.head_dim), **common)
            float_bytes, int_bytes = int(float_bytes), int(int_bytes)
            row = {
                "kind": kind, "cuda_graph": bool(graph),
                "query_lengths": list(query_lengths), "kv_lengths": list(kv_lengths),
                "float_bytes": float_bytes, "int_bytes": int_bytes,
                "float_bound_bytes": float_bound, "int_bound_bytes": int_bound,
                "within_bound": 0 <= float_bytes <= float_bound
                                and 0 <= int_bytes <= int_bound,
            }
            rows.append(row)
            if not row["within_bound"]:
                raise RuntimeError("official concrete-plan query exceeds range upper bound: "
                                   + json.dumps(row, sort_keys=True))
        finally:
            del wrapper

    try:
        for batch in sorted({1, max_batch}):
            balanced_total = min(max_tokens, batch * max_context)
            base, remainder = divmod(balanced_total, batch)
            balanced = [base + (index < remainder) for index in range(batch)]
            concentrated = [min(max_context, balanced_total - batch + 1)] + [1] * (batch - 1)
            for query_lengths in [balanced] + ([concentrated] if concentrated != balanced else []):
                for context in sorted({max_context, max(1, max_context - page_size + 1)}):
                    kv_lengths = [max(context, query_length) for query_length in query_lengths]
                    check("prefill", query_lengths, kv_lengths, False)
            for context in sorted({1, max_context}):
                for graph in ([False, True] if builder.enable_cuda_graph else [False]):
                    check("decode", [1] * batch, [context] * batch, graph)
        return {"passed": True, "selector_bytes": SELECTOR_BYTES,
                "integer_wrapper_bytes": INT_ARENA_BYTES, "calls": len(rows),
                "coverage_source": "range upper bound; concrete checks are startup cross-checks",
                "rows": rows}
    except Exception as error:
        raise RuntimeError(json.dumps({"error": type(error).__name__ + ": " + str(error),
                                       "completed_query_rows": rows}, sort_keys=True)) from error
    finally:
        del selector


def _make_temporary_builder(worker):
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder
    from vllm.v1.kv_cache_interface import AttentionSpec

    specs = worker.model_runner.get_kv_cache_spec()
    if not specs:
        raise ValueError("no attention KV cache specification")
    keys = {
        (type(spec).__name__, getattr(spec, "num_kv_heads", None),
         getattr(spec, "head_size", None), getattr(spec, "block_size", None),
         str(getattr(spec, "dtype", None)))
        for spec in specs.values()
    }
    if len(keys) != 1 or any(not isinstance(spec, AttentionSpec) for spec in specs.values()):
        raise ValueError("a single uniform attention KV group is required")
    layers = get_layers_from_vllm_config(worker.vllm_config, Attention)
    for name in specs:
        if name not in layers or layers[name].get_attn_backend().get_name() != "FLASHINFER":
            raise ValueError("all KV layers must use the native FLASHINFER backend")
    builder = FlashInferMetadataBuilder(next(iter(specs.values())), list(specs),
                                        worker.vllm_config, worker.device)
    if builder._workspace_buffer is not None:
        raise RuntimeError("temporary builder unexpectedly allocated a float workspace")
    return builder


def _install_cached_getter(report, expected_signature):
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder
    original = FlashInferMetadataBuilder._get_workspace_buffer
    if getattr(original, "_fi_startup_auto_hook", False):
        raise RuntimeError("a startup workspace hook is already installed in this process")

    def cached_getter(builder):
        if builder._workspace_buffer is not None:
            return builder._workspace_buffer
        signature = _builder_signature(builder)
        decision = report["decisions"][0]
        if signature != expected_signature:
            fallback = {
                "status": "fallback", "configured_bytes": decision["configured_bytes"],
                "chosen_bytes": _native_bytes(builder, decision["configured_bytes"]),
                "fallback": "runtime builder configuration differs from startup coverage",
                "observedsource": "native FlashInferMetadataBuilder._get_workspace_buffer",
                "limits": signature, "query": None,
                "source": _source_record("native_default_after_signature_mismatch"),
            }
            report["decisions"].append(fallback)
            arena = original(builder)
        elif decision["status"] == "selected":
            arena = torch.zeros(int(decision["chosen_bytes"]), dtype=torch.uint8,
                                device=builder.device)
            builder._workspace_buffer = arena
        else:
            arena = original(builder)
        report["allocation_events"].append({
            "generation": len(report["allocation_events"]) + 1,
            "builder_id": int(id(builder)), "pointer": int(arena.data_ptr()),
            "nbytes": int(arena.numel() * arena.element_size()),
            "during_graph_capture": bool(torch.cuda.is_current_stream_capturing()),
            "query_calls_in_getter": 0,
        })
        _save_trace(report)
        return arena

    cached_getter._fi_startup_auto_hook = True
    FlashInferMetadataBuilder._get_workspace_buffer = cached_getter
    report["worker_health"]["hook_installed"] = True


class AutoWorkspaceWorker(Worker):
    """Native worker with a startup-only, explicitly opted-in sizing policy."""

    def determine_available_memory(self):
        mode = os.environ.get("FI_STARTUP_WORKSPACE_MODE", "default")
        if mode not in ("default", "auto"):
            raise ValueError("FI_STARTUP_WORKSPACE_MODE must be default or auto")
        configured = int(envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE)
        report = {
            "mode": mode, "decisions": [], "allocation_events": [],
            "worker_health": {
                "worker_class": type(self).__module__ + "." + type(self).__name__,
                "model_runner_class": type(self.model_runner).__module__ + "."
                                      + type(self.model_runner).__name__,
                "rank": int(self.rank), "pid": os.getpid(),
                "prepared_before_memory_profile": True, "hook_installed": False,
                "query_calls": 0, "query_calls_in_getter": 0,
                "retained_query_tensors": 0,
                "configured_workspace_bytes": configured,
                "configuration_authorization": "FI_STARTUP_WORKSPACE_MODE experiment environment",
                "vllm_version": importlib.metadata.version("vllm"),
            },
        }
        self._startup_workspace_report = report
        decision = {
            "status": "default" if mode == "default" else "fallback",
            "chosen_bytes": configured, "configured_bytes": configured,
            "observedsource": "native FlashInferMetadataBuilder._get_workspace_buffer",
            "limits": {}, "fallback": None, "query": None,
            "source": _source_record("native_default_no_query"),
        }
        report["decisions"].append(decision)
        if envs.VLLM_ENABLE_STARTUP_PLAN:
            raise RuntimeError("startup-plan KV reuse must be disabled for this experiment")
        if mode == "auto":
            started = time.perf_counter()
            temporary_builder = None
            try:
                if configured != NATIVE_DEFAULT_BYTES:
                    raise ValueError("explicit nondefault workspace configuration is preserved")
                if importlib.metadata.version("vllm") != "0.28.0":
                    raise ValueError("startup lifecycle is pinned to vLLM 0.28.0")
                if self.vllm_config.use_v2_model_runner:
                    raise ValueError("startup lifecycle requires the V1 model runner")
                if envs.VLLM_BATCH_INVARIANT:
                    raise ValueError("batch-invariant split policy is outside this experiment")
                if self.cache_config.kv_cache_memory_bytes is not None:
                    raise ValueError("native automatic KV cache allocation is required")
                with set_current_vllm_config(self.vllm_config):
                    temporary_builder = _make_temporary_builder(self)
                    config, signature = _extract_config(self, temporary_builder)
                    decision["limits"] = config
                    decision["chosen_bytes"] = _native_bytes(temporary_builder, configured)
                    from workspace_policy import choose_workspace
                    policy = choose_workspace(config, verify_sources=True)
                    decision["policy"] = policy
                    decision["source"] = _source_record(
                        "FlashInfer BatchPrefill/BatchDecode.workspace_size; startup-only boundary checks",
                        policy)
                    if not policy.get("supported"):
                        raise ValueError(str(policy.get("reason", "unsupported startup policy")))
                    verification = policy.get("source_verification", {})
                    if not isinstance(verification, dict) or verification.get("passed") is not True:
                        raise ValueError("upstream range source verification did not pass")
                    query = _run_startup_queries(temporary_builder, policy)
                    decision["query"] = query
                    chosen = int(policy["selected_bytes"])
                    native_guard = (int(temporary_builder.max_num_batched_tokens)
                                    * int(temporary_builder.num_qo_heads)
                                    * int(temporary_builder.head_dim) * 16)
                    if chosen < max(int(policy["float_bound_bytes"]), native_guard):
                        raise ValueError("policy arena does not cover range bound and native guard")
                    if chosen >= configured:
                        raise ValueError("policy does not produce a smaller startup workspace")
                    decision.update(status="selected", chosen_bytes=chosen, fallback=None,
                                    observedsource="workspace_policy range bound + official demand queries")
                    report["worker_health"]["query_calls"] = int(query["calls"])
                    _install_cached_getter(report, signature)
            except Exception as error:
                decision["status"] = "fallback"
                decision["fallback"] = type(error).__name__ + ": " + str(error)
                decision["observedsource"] = "native FlashInferMetadataBuilder._get_workspace_buffer"
                decision["chosen_bytes"] = (_native_bytes(temporary_builder, configured)
                                            if temporary_builder is not None else configured)
            finally:
                temporary_builder = None
                gc.collect()
                torch.cuda.empty_cache()
                report["worker_health"]["startup_selection_seconds"] = time.perf_counter() - started
        _save_trace(report)
        result = super().determine_available_memory()
        report["worker_health"]["native_memory_profile_completed"] = True
        _save_trace(report)
        return result


def read_decisions(worker):
    """Collective-RPC helper; expose only JSON data, never allocate an arena."""
    report = getattr(worker, "_startup_workspace_report", None)
    if report is None:
        raise RuntimeError("startup workspace worker report is unavailable")
    result = _json_copy(report)
    existing = []
    for groups in getattr(worker.model_runner, "attn_groups", []):
        for group in groups:
            for builder in group.metadata_builders:
                buffer = getattr(builder, "_workspace_buffer", None)
                if buffer is not None:
                    existing.append({"pointer": int(buffer.data_ptr()),
                                     "nbytes": int(buffer.numel() * buffer.element_size())})
    unique = {row["pointer"]: row for row in existing}
    result["worker_health"]["live_workspaces"] = list(unique.values())
    result["worker_health"]["live_workspace_count"] = len(unique)
    if result["mode"] == "default" and len(unique) == 1:
        result["decisions"][0]["chosen_bytes"] = next(iter(unique.values()))["nbytes"]
    _save_trace(result)
    return result
