"""Native FlashInfer workspace/capacity probe; CUDA is imported only in main/RPC."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback


STAGE = "FI-S03-M06"
S03_FILES = {
    "bench_worker.py": "9b920b1462320781849f94f995195c2f00fbd1ca3bc363adca692609fc1f7bd4",
    "bench_helpers.py": "d05af00aab1c03d957873744ef8fa23f5c7f567692fcfc58e14265baa57e4a94",
}


def validate_phase_cases(phase, cases):
    expected = [{"case_id": "L16384_B4", "input_tokens_per_request": 16384,
                 "batch_size": 4, "output_tokens_per_request": 128}]
    if phase != "capacity" or cases != expected:
        raise RuntimeError("M06 requires the approved 16K four-request performance workload")

def workspace_snapshot(attention_groups, expected_bytes):
    """Read existing builder buffers. This helper never invokes a buffer factory."""
    builders, buffers = [], {}
    for group_index, groups in enumerate(attention_groups):
        for attention_index, group in enumerate(groups):
            for builder_index, builder in enumerate(group.metadata_builders):
                buffer = getattr(builder, "_workspace_buffer", None)
                row = {"group_index": group_index, "attention_index": attention_index,
                       "builder_index": builder_index,
                       "builder_class": type(builder).__module__ + "." + type(builder).__name__,
                       "pointer": None, "nbytes": None}
                if buffer is not None:
                    row.update(pointer=int(buffer.data_ptr()),
                               nbytes=int(buffer.numel() * buffer.element_size()))
                    buffers[row["pointer"]] = {"pointer": row["pointer"], "nbytes": row["nbytes"]}
                builders.append(row)
    observed = list(buffers.values())
    if not observed or any(row["nbytes"] != expected_bytes for row in observed):
        raise RuntimeError("Actual existing FlashInfer workspace differs from configured bytes")
    if len(observed) != 1:
        raise RuntimeError("Expected one shared FlashInfer workspace allocation")
    return {"builders": builders, "unique_buffers": observed, "unique_count": len(observed),
            "total_bytes": sum(row["nbytes"] for row in observed),
            "expected_bytes": expected_bytes, "validated": True}


def inspect_workspace_and_kv(worker, expected_workspace_bytes):
    from startup_worker import read_decisions
    startup = read_decisions(worker)
    decisions = startup.get("decisions", [])
    if not decisions:
        raise RuntimeError("Startup workspace decision is missing")
    expected_workspace_bytes = int(decisions[-1]["chosen_bytes"])
    runner = worker.model_runner
    cache = worker.vllm_config.cache_config
    config = runner.kv_cache_config
    from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
    tokens, concurrency = get_kv_cache_capacity(worker.vllm_config, config)
    scheduler = worker.vllm_config.scheduler_config
    reserve_full = getattr(scheduler, "scheduler_reserve_full_isl", None)
    if reserve_full is not True:
        raise RuntimeError("Actual scheduler_reserve_full_isl must remain True")
    required_tokens = 4 * (16384 + 128)
    mode = startup.get("mode")
    boundary_side_met = int(tokens) < required_tokens if mode == "default" else int(tokens) >= required_tokens
    return {"startup_workspace": startup,
            "scheduler": {
                "source": "worker.vllm_config.scheduler_config",
                "scheduler_reserve_full_isl": reserve_full,
                "max_num_seqs": int(scheduler.max_num_seqs),
                "max_num_batched_tokens": int(scheduler.max_num_batched_tokens),
                "enable_chunked_prefill": bool(scheduler.enable_chunked_prefill),
                "policy": str(scheduler.policy),
                "watermark": float(scheduler.watermark),
                "max_model_len": int(worker.vllm_config.model_config.max_model_len)},
            "capacity_boundary_observation": {
                "required_tokens": required_tokens, "required_prompt_tokens": 4 * 16384,
                "actual_token_capacity": int(tokens), "mode": mode,
                "expected_side": "below_required" if mode == "default" else "at_or_above_required",
                "expected_side_met": boundary_side_met,
                "status": "side_observed" if boundary_side_met else "inconclusive",
                "automatic_cache_adjustment": False,
                "note": "Cross-configuration boundary requires observed default capacity < 66048 <= observed automatic capacity"},
            "actual_workspace": workspace_snapshot(runner.attn_groups, expected_workspace_bytes),
            "kv": {"num_blocks": int(config.num_blocks), "block_size": int(cache.block_size),
                   "total_tensor_bytes": sum(int(tensor.size) for tensor in config.kv_cache_tensors),
                   "token_capacity": int(tokens), "max_concurrency": float(concurrency),
                   "configured_memory_bytes": cache.kv_cache_memory_bytes,
                   "capacity_source": "vllm.v1.core.kv_cache_utils.get_kv_cache_capacity",
                   "allocation_mode": "automatic_gpu_memory_utilization"},
            "automatic_memory": {
                "gpu_memory_utilization": float(cache.gpu_memory_utilization),
                **{name: value if type(value) in (int, float) else None
                   for name in ("requested_memory", "available_kv_cache_memory_bytes", "cudagraph_memory_estimate")
                   for value in [getattr(worker, name, None)]}}}


def native_preemptions(llm):
    """Read the built-in cumulative Prometheus counters without adding a hook."""
    source = "llm.llm_engine.logger_manager.stat_loggers.counter_num_preempted_reqs"
    try:
        manager = llm.llm_engine.logger_manager
        if manager is None:
            return {"available": False, "value": None, "source": source,
                    "reason": "native logger_manager is disabled"}
        found = []
        for logger in manager.stat_loggers:
            counters = getattr(logger, "counter_num_preempted_reqs", None)
            if isinstance(counters, dict):
                found.extend(float(counter._value.get()) for counter in counters.values())
        if not found or any(not math.isfinite(value) or value < 0 for value in found):
            return {"available": False, "value": None, "source": source,
                    "reason": "native cumulative counters are unavailable"}
        value = sum(found)
        if value != int(value):
            raise ValueError("Native preemption counter is not an integer")
        return {"available": True, "value": int(value), "source": source, "reason": None}
    except Exception as error:
        return {"available": False, "value": None, "source": source,
                "reason": type(error).__name__ + ": " + str(error)}


def preemption_delta(before, after):
    if not before["available"] or not after["available"]:
        return None
    difference = after["value"] - before["value"]
    return difference if difference >= 0 else None


def load_s03_helpers(protocol):
    source_root = Path(protocol.get("s03_source_root", "/mnt/f/FlashInferResearch/S03"))
    for name, digest in S03_FILES.items():
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Reused S03 source identity changed: " + name)
    for filename, digest in protocol.get("reused_files", {}).items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Frozen reused source changed: " + filename)
    sys.path.insert(0, str(source_root))
    import bench_helpers
    import bench_worker
    if Path(bench_helpers.__file__).resolve().parent != source_root.resolve():
        raise RuntimeError("Unexpected reused helper import")
    if Path(bench_worker.__file__).resolve().parent != source_root.resolve():
        raise RuntimeError("Unexpected reused worker import")
    return bench_helpers, bench_worker


def native_request_metrics(request):
    """Preserve native vLLM V1 stats and subtract timestamps within one clock."""
    names = ("arrival_time", "queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts",
             "first_token_latency", "num_generation_tokens", "is_corrupted")
    value = getattr(request, "metrics", None)
    raw = {name: None for name in names}
    missing, invalid, uninitialized = [], [], []
    for name in names:
        observed = getattr(value, name, None) if value is not None else None
        if observed is None:
            missing.append(name)
        elif name == "is_corrupted":
            if type(observed) is bool:
                raw[name] = observed
            else:
                invalid.append(name)
        elif name == "num_generation_tokens":
            if type(observed) is int and observed >= 0:
                raw[name] = observed
            else:
                invalid.append(name)
        elif type(observed) in (int, float) and math.isfinite(observed) and observed >= 0:
            raw[name] = float(observed)
            if observed == 0:
                uninitialized.append(name)
        else:
            invalid.append(name)
    derived, availability = {}, {}
    intervals = {
        "queue_seconds": ("scheduled_ts", "queued_ts"),
        "prefill_seconds": ("first_token_ts", "scheduled_ts"),
        "decode_seconds": ("last_token_ts", "first_token_ts"),
        "engine_queued_to_last_token_seconds": ("last_token_ts", "queued_ts"),
        "inference_seconds": ("last_token_ts", "scheduled_ts"),
    }
    for name, (end, start) in intervals.items():
        left, right = raw[end], raw[start]
        ok = left is not None and right is not None and left > 0 and right > 0 and left >= right
        derived[name] = left - right if ok else None
        availability[name] = ok
        if left is not None and right is not None and left < right:
            invalid.append(name + ":negative_interval")
    ttft = raw["first_token_latency"]
    availability["ttft_seconds"] = ttft is not None and ttft > 0
    derived["ttft_seconds"] = ttft if availability["ttft_seconds"] else None
    # RequestStateStats exposes no per-request finished frontend timestamp.
    derived["frontend_end_to_end_seconds"] = None
    availability["frontend_end_to_end_seconds"] = False
    return {"available": value is not None,
            "source": "RequestOutput.metrics / vllm.v1.metrics.stats.RequestStateStats",
            "raw": raw, "clock_domains": {
                "arrival_time": "frontend_wall_clock",
                **{name: "engine_core_monotonic" for name in
                   ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")},
                "first_token_latency": "frontend_wall_clock_interval"},
            "derived": derived, "availability": availability,
            "missing_fields": missing, "invalid_fields": invalid,
            "uninitialized_fields": uninitialized,
            "interval_scope": {
                "native_record": "Native final RequestStateStats snapshot; event history and cumulative rescheduling waits are unavailable",
                "queue_seconds": "Recorded queued_ts to first scheduled_ts; not cumulative waiting across preemptions",
                "prefill_seconds": "First scheduling to first token, including any intervening preemption",
                "decode_seconds": "First to last token, including any intervening preemption",
                "engine_queued_to_last_token_seconds": "Recorded queued event to last generated token",
                "inference_seconds": "First scheduling to last generated token",
                "ttft_seconds": "Native frontend arrival-to-first-output latency interval"},
            "note": "TTFT uses native first_token_latency; engine intervals use monotonic timestamps only. Per-request frontend completion is unavailable; whole-batch elapsed uses perf_counter separately."}


def output_payload(request):
    if len(request.outputs) != 1:
        raise RuntimeError("Expected one greedy candidate per request")
    generated = request.outputs[0]
    return {"text": generated.text, "token_ids": list(generated.token_ids),
            "finish_reason": generated.finish_reason, "stop_reason": generated.stop_reason,
            "request_id": str(request.request_id), "request_metrics": native_request_metrics(request)}


def run_quality(llm, quality, SamplingParams, max_model_len, vocabulary):
    results = []
    for case in quality["cases"]:
        tokens = case.get("prompt_token_ids", case.get("prompts", [None])[0])
        if case.get("scope") == "long" and (not isinstance(tokens, list) or len(tokens) != 8192):
            raise RuntimeError("Long reference-answer input must contain 8192 tokens")
        if (not isinstance(tokens, list) or not tokens or
                any(type(token) is not int or not 0 <= token < vocabulary for token in tokens) or
                len(tokens) + 128 > max_model_len):
            raise RuntimeError("Invalid frozen quality prompt: " + case["case_id"])
        sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128,
                                  ignore_eos=False, logprobs=1, detokenize=True, seed=20261023)
        response = llm.generate([{"prompt_token_ids": tokens}], sampling_params=sampling,
                                use_tqdm=False)
        if len(response) != 1:
            raise RuntimeError("Quality request count differs")
        result = output_payload(response[0])
        selected = []
        logprobs = response[0].outputs[0].logprobs
        if logprobs is None or len(logprobs) != len(result["token_ids"]):
            raise RuntimeError("Selected-token quality logprobs are unavailable")
        for token, candidates in zip(result["token_ids"], logprobs):
            if token not in candidates or not math.isfinite(float(candidates[token].logprob)):
                raise RuntimeError("Quality selected-token logprob is missing or non-finite")
            selected.append(float(candidates[token].logprob))
        if not result["token_ids"]:
            raise RuntimeError("Empty quality response")
        results.append({"case_id": case["case_id"], "scope": case["scope"],
                        "input_token_count": len(tokens), **result,
                        "generated_token_count": len(result["token_ids"]),
                        "selected_logprobs": selected, "selected_token_logprobs": selected,
                        "logprobs_finite": True})
    if len(results) != 6:
        raise RuntimeError("Expected three short and three long reference-answer tasks")
    return results


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def selected_probability_evidence(request):
    generated = request.outputs[0]
    probabilities, errors = [], []
    entries = generated.logprobs
    if entries is None or len(entries) != len(generated.token_ids):
        return {"selected_logprobs": [], "selected_token_logprobs": [],
                "logprobs_finite": False, "logprob_errors": ["missing_or_wrong_count"]}
    for index, (token, candidates) in enumerate(zip(generated.token_ids, entries)):
        entry = candidates.get(token) if isinstance(candidates, dict) else None
        try:
            value = float(entry.logprob)
        except (TypeError, ValueError, AttributeError):
            value = None
        if value is None or not math.isfinite(value):
            errors.append("missing_or_nonfinite_selected_token_" + str(index))
            value = None
        probabilities.append(value)
    return {"selected_logprobs": probabilities, "selected_token_logprobs": probabilities,
            "logprobs_finite": bool(probabilities) and not errors, "logprob_errors": errors}


def quality_score(text, question):
    expected = question["expected"]
    if question.get("score_mode") != "json_fields":
        raise RuntimeError("Concurrent quality requires frozen JSON-field scoring")
    body = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, re.DOTALL | re.IGNORECASE)
    actual, parse_error = None, None
    try:
        actual = json.loads(fence.group(1) if fence else body)
    except (ValueError, TypeError) as error:
        parse_error = type(error).__name__ + ": " + str(error)
    field_results = {}
    for key, value in expected.items():
        observed = actual.get(key) if isinstance(actual, dict) else None
        correct = (isinstance(actual, dict) and key in actual and type(observed) is type(value)
                   and observed == value)
        field_results[key] = {"expected": value, "actual": observed, "correct": correct}
    correct = isinstance(actual, dict) and all(row["correct"] for row in field_results.values())
    return {"answer_correct": correct, "score_details": {
        "score_mode": "json_fields", "expected": expected, "parsed": actual,
        "parse_error": parse_error, "field_results": field_results}}


def concurrent_pressure_observation(outputs, qa_slot):
    qa = outputs[qa_slot - 1]["request_metrics"]["raw"]
    companions = [row["request_metrics"]["raw"] for index, row in enumerate(outputs)
                  if index != qa_slot - 1]
    def valid(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    start, end = qa.get("scheduled_ts"), qa.get("last_token_ts")
    available = (valid(start) and valid(end) and end >= start
                 and all(valid(row.get("scheduled_ts")) and valid(row.get("last_token_ts"))
                         and row["last_token_ts"] >= row["scheduled_ts"] for row in companions))
    result = {"source": "native engine monotonic scheduled_ts/last_token_ts",
              "available": available, "qa_first_scheduled_ts": start,
              "qa_last_token_ts": end, "qa_finished_before_fixed_companion_budget": None,
              "companions_active_at_qa_first_scheduling": None,
              "companions_active_at_qa_completion": None,
              "companions_with_execution_interval_overlapping_qa": None,
              "qa_started_before_any_companion_finished": None,
              "note": "Native first-scheduled to last-token intervals include interleaving; the QA may end before 128 tokens."}
    if available:
        result.update(
            companions_active_at_qa_first_scheduling=sum(row["scheduled_ts"] <= start < row["last_token_ts"] for row in companions),
            companions_active_at_qa_completion=sum(row["scheduled_ts"] <= end < row["last_token_ts"] for row in companions),
            companions_with_execution_interval_overlapping_qa=sum(max(start, row["scheduled_ts"]) < min(end, row["last_token_ts"]) for row in companions),
            qa_started_before_any_companion_finished=all(start < row["last_token_ts"] for row in companions),
            qa_finished_before_fixed_companion_budget=all(end < row["last_token_ts"] for row in companions))
    return result


def run_concurrent_quality(llm, quality, SamplingParams, protocol, common, output,
                           resume_source, helpers, progress, max_model_len, vocabulary,
                           input_digest):
    config = protocol["concurrent_quality"]
    repeats = protocol.get("concurrent_quality_measurement_iterations", 2)
    scenarios = quality["cases"]
    expected_ids = config["cases"]
    if (repeats != 2 or len(scenarios) != 4
            or [case["case_id"] for case in scenarios] != expected_ids):
        raise RuntimeError("Concurrent quality scenario set differs from the approved scope")
    bindings, summaries = [], []
    for scenario_index, scenario in enumerate(scenarios):
        case_id, slot = scenario["case_id"], scenario["qa_slot"]
        if not re.fullmatch(r"[A-Z0-9_]+", case_id) or slot not in (1, 4):
            raise RuntimeError("Invalid concurrent quality scenario identity")
        requests, question = scenario["requests"], scenario["qa"]
        if len(requests) != 4 or requests[slot - 1]["role"] != "qa":
            raise RuntimeError("Concurrent quality requires four requests and one QA slot")
        if sum(row["role"] == "qa" for row in requests) != 1:
            raise RuntimeError("Concurrent quality requires one QA and three companions")
        if requests[slot - 1]["prompt_token_ids"] != question["prompt_token_ids"]:
            raise RuntimeError("Concurrent QA prompt binding differs")
        sampling_list, prompts = [], []
        for request_index, row in enumerate(requests):
            tokens = row["prompt_token_ids"]
            if (not isinstance(tokens, list) or len(tokens) != 16384
                    or len(tokens) + 128 > max_model_len
                    or any(type(token) is not int or not 0 <= token < vocabulary for token in tokens)):
                raise RuntimeError("Concurrent prompt is not a valid frozen 16K input")
            qa_request = request_index == slot - 1
            sampling_args = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 128,
                             "ignore_eos": not qa_request, "detokenize": True,
                             "seed": 20261023 if qa_request else 20261022}
            if qa_request:
                sampling_args["logprobs"] = 1
            declared = row.get("sampling", {})
            for name in ("temperature", "top_p", "max_tokens", "ignore_eos", "detokenize", "seed", "logprobs"):
                if name in declared and declared[name] != sampling_args.get(name):
                    raise RuntimeError("Concurrent request sampling differs from frozen intent")
            sampling_list.append(SamplingParams(**sampling_args))
            prompts.append({"prompt_token_ids": tokens})
        scenario_digest = canonical_sha256(scenario)
        destination = output / "concurrent-quality" / (case_id + ".json")
        identity = {key: common[key] for key in ("stage", "job", "job_id", "backend", "phase",
                    "workspace_mib", "manifest_sha256", "protocol_sha256", "inputs_sha256")}
        identity.update(case_id=case_id, qa_slot=slot, qa_task_id=question["task_id"],
                        concurrent_inputs_sha256=input_digest, scenario_sha256=scenario_digest)
        saved = None
        if resume_source:
            prior = Path(resume_source).resolve() / "concurrent-quality" / (case_id + ".json")
            if prior.is_file():
                candidate = helpers.read_json(prior)
                if (candidate.get("collection_complete") is True
                        and candidate.get("measurement_count") == repeats
                        and len(candidate.get("samples", [])) == repeats
                        and all(candidate.get(key) == value for key, value in identity.items())
                        and candidate.get("samples_sha256") == canonical_sha256(candidate["samples"])):
                    saved = candidate
                    saved["resumed_from"] = str(prior)
                    helpers.write_json(destination, saved)
        if saved is None:
            samples = []
            saved = {**identity, "collection_complete": False, "measurement_count": 0,
                     "expected_measurement_count": repeats, "samples": samples,
                     "resumed_from": None, "sampling_policy": {
                         "qa": {"max_tokens": 128, "ignore_eos": False, "logprobs": 1,
                                "seed": 20261023},
                         "companions": {"max_tokens": 128, "ignore_eos": True,
                                        "seed": 20261022}}, "expected": question["expected"]}
            for repeat in range(repeats):
                progress("concurrent_quality", concurrent_case_id=case_id,
                         concurrent_repeat_done=repeat, concurrent_case_done=scenario_index)
                before = native_preemptions(llm)
                begin = time.perf_counter()
                generated = llm.generate(prompts, sampling_params=sampling_list, use_tqdm=False)
                elapsed = time.perf_counter() - begin
                after = native_preemptions(llm)
                if len(generated) != 4:
                    raise RuntimeError("Concurrent quality generated request count differs")
                outputs = [output_payload(request) for request in generated]
                for index, row in enumerate(outputs):
                    if index != slot - 1 and len(row["token_ids"]) != 128:
                        raise RuntimeError("Concurrent companion did not complete 128 output tokens")
                qa_output = {**outputs[slot - 1], **selected_probability_evidence(generated[slot - 1]),
                             **quality_score(outputs[slot - 1]["text"], question),
                             "generated_token_count": len(outputs[slot - 1]["token_ids"]),
                             "normal_eos": outputs[slot - 1]["finish_reason"] == "stop"}
                samples.append({"iteration": repeat + 1, "elapsed_seconds": elapsed,
                    "elapsed_clock": "frontend_perf_counter_interval", "request_count": 4,
                    "input_tokens_per_request": [16384] * 4, "qa_slot": slot,
                    "actual_generated_tokens": [len(row["token_ids"]) for row in outputs],
                    "total_generated_tokens": sum(len(row["token_ids"]) for row in outputs),
                    "outputs_sha256": canonical_sha256([row["token_ids"] for row in outputs]),
                    "outputs": outputs, "qa_output": qa_output,
                    "preemptions": {"before": before, "after": after,
                                    "delta": preemption_delta(before, after)},
                    "pressure_observation": concurrent_pressure_observation(outputs, slot)})
                saved.update(measurement_count=len(samples), collection_complete=len(samples) == repeats,
                             samples_sha256=canonical_sha256(samples))
                helpers.write_json(destination, saved)
        bindings.append({"case_id": case_id, "path": "concurrent-quality/" + case_id + ".json",
                         "sha256": helpers.file_hash(destination)})
        summaries.append({"case_id": case_id, "qa_slot": slot, "measurement_count": repeats,
                          "answer_correct_count": sum(sample["qa_output"]["answer_correct"] for sample in saved["samples"]),
                          "normal_eos_count": sum(sample["qa_output"]["normal_eos"] for sample in saved["samples"]),
                          "finite_logprob_count": sum(sample["qa_output"]["logprobs_finite"] for sample in saved["samples"])})
        progress("concurrent_quality", concurrent_case_id=case_id,
                 concurrent_repeat_done=repeats, concurrent_case_done=scenario_index + 1)
    return {"inputs_sha256": input_digest, "measurement_repeats": repeats,
            "collection_complete": True, "cases": bindings, "scenario_summaries": summaries,
            "timing_scope": "QA can stop before 128 tokens; these mixed batches are separate from fixed-work performance timing."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--resume-source")
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    helpers, previous = load_s03_helpers(protocol)
    read_json, write_json, file_hash = helpers.read_json, helpers.write_json, helpers.file_hash
    job = next(job for job in protocol["jobs"] if job.get("id", job.get("job_id")) == args.job)
    phase, workspace_mib = job["phase"], int(job["workspace_mib"])
    phase_config = protocol["phases"][phase]
    common = {"stage": STAGE, "job": args.job, "job_id": args.job,
              "backend": "FLASHINFER", "phase": phase, "workspace_mib": workspace_mib,
              "passed": False}
    completed = []

    def progress(activity, case_id=None, **details):
        write_json(output / "progress.json", {"stage": STAGE, "job": args.job,
                   "phase": activity, "experiment_phase": phase,
                   "workspace_mib": workspace_mib, "case_id": case_id,
                   "completed_cases": len(completed), "epoch": time.time(), **details})

    try:
        manifest = read_json(root / "manifest.json")
        for relative, digest in manifest["files"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or file_hash(path) != digest:
                raise RuntimeError("Frozen experiment file changed: " + relative)
        inputs_path = root / phase_config.get("inputs_file", "inputs.json")
        input_digest = file_hash(inputs_path)
        if phase_config.get("inputs_sha256") and input_digest != phase_config["inputs_sha256"]:
            raise RuntimeError("Frozen phase input identity changed")
        common.update(manifest_sha256=file_hash(root / "manifest.json"),
                      protocol_sha256=file_hash(protocol_path), inputs_sha256=input_digest)
        concurrent_quality_file = root / protocol.get("concurrent_quality_inputs_file", "concurrent-quality-inputs.json")
        concurrent_digest = file_hash(concurrent_quality_file)
        if concurrent_digest != protocol["concurrent_quality_inputs_sha256"]:
            raise RuntimeError("Frozen concurrent quality input identity changed")
        concurrent_inputs = read_json(concurrent_quality_file)
        versions = {name: importlib.metadata.version(name) for name in protocol.get("expected_versions", {})}
        if versions != protocol.get("expected_versions", {}):
            raise RuntimeError("Installed dependency versions changed")
        for item in protocol.get("model_files", []):
            path = Path(protocol["model_linux"]) / item["name"]
            observed = path.stat()
            if observed.st_size != item["bytes"] or (
                    "mtime_ns" in item and observed.st_mtime_ns != item["mtime_ns"]):
                raise RuntimeError("Model file metadata changed: " + item["name"])
            if "sha256" in item and file_hash(path) != item["sha256"]:
                raise RuntimeError("Model/tokenizer file changed: " + item["name"])
        inputs = read_json(inputs_path)
        cases = phase_config["cases"]
        validate_phase_cases(phase, cases)
        frozen = {helpers.case_shape(case)["case_id"]: case for case in inputs["cases"]}
        engine = {**protocol.get("engine", {}), **phase_config["engine"]}
        if engine.get("max_model_len") != 16512 or engine.get("max_num_seqs") != 4:
            raise RuntimeError("M06 requires the approved maximum context and four-request service")
        if "kv_cache_memory_bytes_by_workspace_mib" in phase_config:
            engine["kv_cache_memory_bytes"] = phase_config["kv_cache_memory_bytes_by_workspace_mib"][str(workspace_mib)]
        if "engine" in job:
            engine.update(job["engine"])
        engine.update(job.get("engine_overrides", {}))
        for case in cases:
            helpers.validate_prompts(case, frozen[helpers.case_shape(case)["case_id"]]["prompts"],
                                     inputs["vocab_size"], engine["max_model_len"])
        runtime_env = protocol.get("runtime_env", {}).get(args.job, {})
        if not runtime_env:
            raise RuntimeError("Missing per-job frozen runtime environment")
        for name, value in runtime_env.items():
            if str(os.environ.get(name, "")) != str(value):
                raise RuntimeError("Controller did not provide frozen environment: " + name)
        expected_bytes = workspace_mib * 1024 * 1024
        if os.environ.get("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE") != str(expected_bytes):
            raise RuntimeError("Workspace environment differs from the selected job")
        if "flashinfer" in sys.modules:
            raise RuntimeError("FlashInfer was imported before selecting isolated source")
        isolation = protocol["isolated_variants"]["fixed"]
        package = Path(isolation["package_root"]).resolve()
        if file_hash(package / "__init__.py") != isolation["package_entry_sha256"]:
            raise RuntimeError("Fixed FlashInfer package identity differs")
        sys.path.insert(0, str(package.parent))
        progress("loading_model_and_capturing_graphs")
        from vllm import LLM, SamplingParams
        engine.update(model=protocol["model_linux"], attention_config={"backend": "FLASHINFER"})
        os.environ["FI_STARTUP_WORKSPACE_TRACE"] = str(output / "startup-decision.json")
        begin = time.monotonic()
        llm = LLM(**engine)
        initialization_seconds = time.monotonic() - begin
        layers = llm.apply_model(previous.inspect_model_attention)
        runtimes = llm.collective_rpc(previous.inspect_runtime,
                                      args=(str(package), isolation["cache_root"]))
        planning = llm.collective_rpc(inspect_workspace_and_kv, args=(expected_bytes,))
        if len(layers) != 1 or len(layers[0]) != 28 or any(
                row["backend"] != "FLASHINFER" for row in layers[0]):
            raise RuntimeError("Actual 28 attention layers differ from FLASHINFER")
        if len(runtimes) != 1 or len(planning) != 1:
            raise RuntimeError("Expected exactly one native model worker")
        previous.validate_runtime(runtimes[0], protocol.get("expected_model_runner"))
        if (engine.get("kv_cache_memory_bytes") is not None
                or planning[0]["kv"]["configured_memory_bytes"] is not None
                or planning[0]["automatic_memory"]["gpu_memory_utilization"] != 0.65):
            raise RuntimeError("Actual cache allocation is not the native automatic mode")
        realpath = {"attention_layers": layers, "runtime": runtimes,
                    "planning": planning, "requested_engine": engine,
                    "initialization_seconds": initialization_seconds}
        write_json(output / "workerpath.json", realpath)
        progress("quality_probes")
        quality_file = root / protocol.get("quality_inputs_file", "quality-inputs.json")
        quality = run_quality(llm, read_json(quality_file), SamplingParams,
                              engine["max_model_len"], inputs["vocab_size"])
        write_json(output / "quality-results.json", quality)
        warmups = phase_config.get("warmup_iterations", 2)
        iterations = phase_config.get("measurement_iterations", 10 if phase == "memory" else 5)
        ordered = list(reversed(cases)) if job.get("batch", 1) == 2 else cases
        for case in ordered:
            shape = helpers.case_shape(case)
            case_id = shape["case_id"]
            destination = output / "cases" / (case_id + ".json")
            if args.resume_source:
                prior = Path(args.resume_source).resolve() / "cases" / (case_id + ".json")
                if prior.is_file():
                    saved = read_json(prior)
                    if saved.get("passed") is True:
                        helpers.validate_finished_case(saved, case, {key: common[key] for key in
                            ("manifest_sha256", "protocol_sha256", "inputs_sha256")},
                            "FLASHINFER", args.job, iterations)
                        if saved.get("phase") != phase or saved.get("workspace_mib") != workspace_mib:
                            raise RuntimeError("Saved case workspace or phase differs")
                        saved["resumed_from"] = str(prior)
                        write_json(destination, saved)
                        completed.append(case_id)
                        progress("resumed_completed_case", case_id)
                        continue
            prompts = [{"prompt_token_ids": tokens} for tokens in frozen[case_id]["prompts"]]
            sampling_args = dict(protocol.get("sampling", {"temperature": 0.0, "top_p": 1.0,
                                                           "ignore_eos": True, "detokenize": True}))
            sampling_args["max_tokens"] = shape["output_tokens_per_request"]
            sampling = SamplingParams(**sampling_args)
            progress("warming_up", case_id)
            for index in range(warmups):
                llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
                progress("warming_up", case_id, warmup_done=index + 1)
            samples = []
            result = {**common, **shape, "warmup_count": warmups, "measurement_count": 0,
                      "samples": samples, "resumed_from": None}
            for index in range(iterations):
                progress("measuring", case_id, measurement_done=index)
                before = native_preemptions(llm) if phase == "capacity" else {
                    "available": False, "value": None, "source": None, "reason": "memory phase stats disabled"}
                begin = time.perf_counter()
                generated = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
                elapsed = time.perf_counter() - begin
                after = native_preemptions(llm) if phase == "capacity" else dict(before)
                if len(generated) != shape["batch_size"]:
                    raise RuntimeError("Generated request count differs")
                outputs = [output_payload(request) for request in generated]
                tokens = [row["token_ids"] for row in outputs]
                sample = {"iteration": index + 1, "elapsed_seconds": elapsed,
                          "elapsed_clock": "frontend_perf_counter_interval",
                          "actual_generated_tokens": [len(row) for row in tokens],
                          "request_count": len(outputs), "total_generated_tokens": sum(map(len, tokens)),
                          "outputs_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
                          "outputs": outputs,
                          "preemptions": {"before": before, "after": after,
                                          "delta": preemption_delta(before, after)}}
                helpers.validate_sample(sample, shape["batch_size"], shape["output_tokens_per_request"])
                samples.append(sample)
                result["measurement_count"] = len(samples)
                result["passed"] = len(samples) == iterations
                if result["passed"]:
                    result["median_seconds"] = statistics.median(row["elapsed_seconds"] for row in samples)
                write_json(destination, result)
            completed.append(case_id)
            progress("case_complete", case_id)
        concurrent_quality = run_concurrent_quality(
            llm, concurrent_inputs, SamplingParams, protocol, common, output,
            args.resume_source, helpers, progress, engine["max_model_len"],
            inputs["vocab_size"], concurrent_digest)
        memory = llm.collective_rpc(previous.inspect_generation_memory)
        planning_end = llm.collective_rpc(inspect_workspace_and_kv, args=(expected_bytes,))
        if planning[0]["actual_workspace"]["unique_buffers"] != planning_end[0]["actual_workspace"]["unique_buffers"]:
            raise RuntimeError("Formal workspace changed after graph capture")
        bindings = [{"case_id": helpers.case_shape(case)["case_id"],
                     "path": "cases/" + helpers.case_shape(case)["case_id"] + ".json",
                     "sha256": file_hash(output / "cases" / (helpers.case_shape(case)["case_id"] + ".json"))}
                    for case in cases]
        write_json(output / "result.json", {**common, "passed": True, "cases": bindings,
                   "initialization_seconds": initialization_seconds,
                   "workerpath_relative": "workerpath.json", "workerpath_sha256": file_hash(output / "workerpath.json"),
                   "case_order": [helpers.case_shape(case)["case_id"] for case in ordered],
                   "quality_results": quality, "quality_inputs_sha256": file_hash(quality_file),
                   "concurrent_quality": concurrent_quality,
                   "startup_workspace": planning_end[0]["startup_workspace"],
                   "scheduler": planning_end[0]["scheduler"],
                   "capacity_boundary_observation": planning_end[0]["capacity_boundary_observation"],
                   "actual_workspace": planning_end[0]["actual_workspace"], "kv": planning_end[0]["kv"],
                   "planning": {"before": planning[0], "after": planning_end[0]}, "realpath": realpath,
                   "memory": {"worker_process": memory,
                              "initialization_peak_allocated_bytes": runtimes[0]["initialization_peak_allocated_bytes"],
                              "initialization_peak_reserved_bytes": runtimes[0]["initialization_peak_reserved_bytes"],
                              "measurement_note": "Lifetime worker peak includes init, quality, warmups and generation; all RPC outside timing"}})
        progress("complete")
        print("S03_M06_RESULT_SAVED", flush=True)
    except Exception as error:
        write_json(output / "error.json", {"type": type(error).__name__, "message": str(error),
                   "traceback": traceback.format_exc()})
        write_json(output / "result.json", {**common, "passed": False, "reason": str(error)})
        progress("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
