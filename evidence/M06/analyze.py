"""CPU-only correctness evidence for startup workspace adaptation under KV pressure.

The immutable M05 analyzer supplies existing source, receipt, workspace, pointer,
automatic-KV and pure-performance validation. This stage adds independently frozen
concurrent questions, answer scoring and native admission-time coverage.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

STAGE = "FI-S03-M06"
MIB = 1024 * 1024
BASE_ANALYZER_SHA256 = "e88fd297fd9ad6f0e4dd72494dafb9fb673130a94e3a33821c2125d4db52d62f"
PURE_CASE = "L16384_B4"
CASE_SPECS = {PURE_CASE: {"case_id": PURE_CASE, "input_tokens_per_request": 16384,
                         "batch_size": 4, "output_tokens_per_request": 128}}
SCENARIOS = {"CQ_LOOKUP_SLOT4": ("LOOKUP", 4), "CQ_LATEST_SLOT4": ("LATEST", 4),
             "CQ_COMBINE_SLOT4": ("COMBINE", 4), "CQ_LOOKUP_SLOT1": ("LOOKUP", 1)}
JOB_SPECS = {"01_DEFAULT": 394, "02_STARTUP_AUTO": 394,
             "03_STARTUP_AUTO": 394, "04_DEFAULT": 394}
PAIRS = {"first": ("01_DEFAULT", "02_STARTUP_AUTO"),
         "reverse": ("04_DEFAULT", "03_STARTUP_AUTO")}
PARENT_CHARGED = 4091.642331315
APPROVED_COMBINED = 5291.642331315


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _within(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def _finite(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def _base(root):
    """Load helpers from their frozen read-only predecessor on Windows or WSL."""
    candidates = [Path(root).parent / "S03-M05" / "analyze.py",
                  Path("/mnt/f/FlashInferResearch/S03-M05/analyze.py"),
                  Path("F:/FlashInferResearch/S03-M05/analyze.py")]
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if _sha(path) != BASE_ANALYZER_SHA256:
        raise RuntimeError("frozen_m05_analyzer_sha256_mismatch")
    spec = importlib.util.spec_from_file_location("_m06_frozen_m05_analysis", path)
    module = importlib.util.module_from_spec(spec)
    # Compile the frozen source directly to avoid writing __pycache__ in M05.
    exec(compile(path.read_text(encoding="utf-8-sig"), str(path), "exec"), module.__dict__)
    # Function globals belong to this private module object; the predecessor file
    # and its archived result are never written or recalculated.
    module.STAGE, module.CASE_SPECS = STAGE, CASE_SPECS
    return module


def _prior_m05(root, protocol):
    errors, result = [], {"source_stage": "FI-S03-M05", "read_only": True}
    try:
        identity = protocol["prior_m05_evidence"]
        candidates = [Path(identity[name]) for name in ("root_linux", "root_windows") if identity.get(name)]
        previous_root = next((path for path in candidates if path.is_dir()), candidates[0])
        for name in ("ledger", "protocol", "manifest", "summary"):
            if _sha(previous_root / (name + ".json")) != identity[name + "_sha256"]:
                errors.append("prior_m05_" + name + "_sha256_mismatch")
        documents = {name: _read(previous_root / (name + ".json"))
                     for name in ("ledger", "protocol", "summary")}
        if any(value.get("stage") != result["source_stage"] for value in documents.values()):
            errors.append("prior_m05_stage_mismatch")
        for name in ("protocol", "manifest"):
            if documents["ledger"].get(name + "_sha256") != identity[name + "_sha256"]:
                errors.append("prior_m05_ledger_" + name + "_mismatch")
        charged = sum(row.get("charged_seconds", 0) for row in documents["ledger"].get("attempts", []))
        if (abs(charged - 968.069123306) > 1e-6
                or abs(charged - protocol.get("parent_m05_gpu_charged_seconds", 968.069123306)) > 1e-6):
            errors.append("prior_m05_gpu_charge_mismatch")
        if _sha(root / "quality-inputs.json") != _sha(previous_root / "quality-inputs.json"):
            errors.append("six_legacy_quality_inputs_changed_from_m05")
        previous_summary = documents["summary"]
        result.update(root=str(previous_root), gpu_charged_seconds=charged,
                      original_passed=previous_summary.get("passed"),
                      original_performance_net_benefit_observed=previous_summary.get("performance_net_benefit_observed"),
                      original_long_quality_passed=previous_summary.get("long_quality_passed"),
                      original_summary_sha256=identity["summary_sha256"],
                      original_pressure_comparison=previous_summary.get("comparisons", {}).get("capacity", {}).get(PURE_CASE),
                      original_quality_equivalence=previous_summary.get("quality_equivalence"),
                      historical_result_retained=True)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
        errors.append(str(error))
    result.update(valid=not errors, errors=errors)
    return result


def _concurrent_inputs(root, protocol, identities):
    errors, value = [], {}
    try:
        config = protocol["concurrent_quality"]
        filename = config["inputs_file"]
        digest = _sha(_within(root, filename))
        identities["hashes"]["concurrent_quality_inputs_sha256"] = digest
        if config.get("inputs_sha256") != digest or protocol.get("concurrent_quality_inputs_sha256") != digest:
            errors.append("concurrent_inputs_sha256_mismatch")
        if protocol.get("concurrent_quality_inputs_file") != filename:
            errors.append("concurrent_inputs_file_alias_mismatch")
        if (config.get("measurement_repeats") != 2 or protocol.get("concurrent_quality_measurement_iterations") != 2
                or config.get("cases") != list(SCENARIOS)):
            errors.append("concurrent_protocol_scenarios_or_repeats_mismatch")
        manifest = _read(root / "manifest.json")
        if manifest.get("files", {}).get(filename) != digest:
            errors.append("concurrent_inputs_not_bound_in_manifest")
        value = _read(_within(root, filename))
        if value.get("stage") != STAGE or value.get("measurement_repeats") != 2:
            errors.append("concurrent_inputs_stage_or_repeats_mismatch")
        cases = value.get("cases", [])
        if (not isinstance(cases, list) or len(cases) != 4
                or {row.get("case_id") for row in cases} != set(SCENARIOS)):
            raise ValueError("concurrent_input_case_set_mismatch")
        previous_identity = protocol["prior_m05_evidence"]
        previous_candidates = [Path(previous_identity[name]) for name in ("root_linux", "root_windows")
                               if previous_identity.get(name)]
        previous_root = next((path for path in previous_candidates if path.is_dir()), previous_candidates[0])
        original = _read(previous_root / "capacity-inputs.json")
        original_pressure = next(row for row in original["cases"] if row["case_id"] == PURE_CASE)
        current_performance = _read(root / protocol["phases"]["capacity"].get("inputs_file", "capacity-inputs.json"))
        if current_performance.get("cases") != [original_pressure]:
            errors.append("pure_performance_prompt_content_changed_from_m05")
        for scenario in cases:
            case_id, qa = scenario["case_id"], scenario["qa"]
            task, slot = SCENARIOS[case_id]
            if (scenario.get("input_tokens_per_request") != 16384 or scenario.get("batch_size") != 4
                    or scenario.get("qa_slot") != slot or task not in str(scenario.get("qa_task_id", "")).upper()
                    or scenario.get("qa_task_id") != qa.get("task_id")
                    or qa.get("score_mode") != "json_fields" or qa.get("max_tokens") != 128
                    or qa.get("ignore_eos") is not False or not isinstance(qa.get("expected"), dict)
                    or not qa["expected"] or not isinstance(qa.get("expected_json_types"), dict)):
                errors.append("concurrent_input_qa_contract_mismatch:" + case_id)
            expected_types = qa.get("expected_json_types", {})
            if set(expected_types) != set(qa.get("expected", {})):
                errors.append("concurrent_expected_type_keys_mismatch:" + case_id)
            elif not all(_type_matches(qa["expected"][key], label) for key, label in expected_types.items()):
                errors.append("concurrent_expected_value_type_mismatch:" + case_id)
            requests = scenario.get("requests", [])
            if not isinstance(requests, list) or len(requests) != 4:
                errors.append("concurrent_input_request_count_mismatch:" + case_id)
                continue
            for index, request in enumerate(requests, 1):
                tokens, sampling = request.get("prompt_token_ids"), request.get("sampling", {})
                if (not isinstance(tokens, list) or len(tokens) != 16384
                        or any(type(token) is not int or token < 0 for token in tokens)):
                    errors.append("concurrent_input_token_shape_mismatch:" + case_id + ":" + str(index))
                if request.get("prompt_token_sha256") != _canonical_sha(tokens):
                    errors.append("concurrent_input_token_hash_mismatch:" + case_id + ":" + str(index))
                role = "qa" if index == slot else "companion"
                if request.get("role") != role or sampling.get("max_tokens") != 128:
                    errors.append("concurrent_input_request_role_or_length_mismatch:" + case_id + ":" + str(index))
                if role == "qa":
                    if sampling.get("ignore_eos") is not False or tokens != qa.get("prompt_token_ids"):
                        errors.append("concurrent_qa_request_binding_mismatch:" + case_id)
                elif sampling.get("ignore_eos") is not True:
                    errors.append("concurrent_companion_ignore_eos_mismatch:" + case_id + ":" + str(index))
                if role == "companion" and tokens != original_pressure["prompts"][index - 1]:
                    errors.append("concurrent_companion_prompt_changed_from_m05:" + case_id + ":" + str(index))
        by_case = {row["case_id"]: row for row in cases}
        if by_case["CQ_LOOKUP_SLOT1"]["qa"] != by_case["CQ_LOOKUP_SLOT4"]["qa"]:
            errors.append("lookup_first_slot_control_question_changed")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
        errors.append(str(error))
    return {"valid": not errors, "errors": errors, "value": value,
            "by_id": {row["case_id"]: row for row in value.get("cases", []) if isinstance(row, dict) and "case_id" in row}}


def _normalized(value):
    return re.sub(r"\s+", "", value).strip("。.!！,，:：;；\"'` ")


def _field_match(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and _field_match(actual[key], value)
                                               for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _field_match(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, str):
        return isinstance(actual, str) and _normalized(actual) == _normalized(expected)
    return type(actual) is type(expected) and actual == expected


def _type_matches(value, label):
    return {"string": type(value) is str, "str": type(value) is str,
            "integer": type(value) is int, "int": type(value) is int,
            "number": type(value) in (int, float) and math.isfinite(value),
            "boolean": type(value) is bool, "bool": type(value) is bool,
            "array": type(value) is list, "list": type(value) is list,
            "object": type(value) is dict, "dict": type(value) is dict,
            "null": value is None}.get(str(label).lower(), False)


def _score(text, qa):
    parsed, error = None, None
    body = text.strip() if isinstance(text, str) else ""
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, flags=re.DOTALL | re.IGNORECASE)
    try:
        parsed = json.loads(fence.group(1) if fence else body)
    except (ValueError, TypeError) as caught:
        error = str(caught)
    type_results = {key: isinstance(parsed, dict) and key in parsed and _type_matches(parsed[key], label)
                    for key, label in qa.get("expected_json_types", {}).items()}
    field_results = {key: isinstance(parsed, dict) and key in parsed and _field_match(parsed[key], expected)
                     for key, expected in qa["expected"].items()}
    passed = bool(field_results) and all(field_results.values()) and all(type_results.values())
    return {"answer_correct": passed, "parsed_json": parsed, "json_parse_error": error,
            "field_correct": field_results, "field_type_correct": type_results,
            "expected": qa["expected"], "expected_json_types": qa.get("expected_json_types", {})}


def _pressure(outputs, slot):
    """Compare engine-monotonic timestamps only; wall-clock arrival is not mixed in."""
    raw = [row.get("request_metrics", {}).get("raw", {}) for row in outputs]
    qa_raw = raw[slot - 1] if len(raw) == 4 else {}
    scheduled, queued = qa_raw.get("scheduled_ts"), qa_raw.get("queued_ts")
    completion = [row.get("last_token_ts") for index, row in enumerate(raw, 1) if index != slot]
    available = (_finite(scheduled, True) and _finite(queued, True) and len(completion) == 3
                 and all(_finite(value, True) for value in completion)
                 and all(row.get("request_metrics", {}).get("clock_domains", {}).get(key) == "engine_core_monotonic"
                         for row in outputs for key in ("queued_ts", "scheduled_ts", "last_token_ts")))
    earliest = min(completion) if available else None
    difference = scheduled - earliest if available else None
    return {"available": available, "qa_slot": slot, "qa_queued_ts": queued,
            "qa_scheduled_ts": scheduled, "companion_last_token_ts": completion,
            "earliest_companion_last_token_ts": earliest,
            "qa_scheduled_minus_earliest_companion_last_seconds": difference,
            "qa_queue_seconds": scheduled - queued if available else None,
            "qa_scheduled_before_first_companion_finishes": difference < -1e-6 if available else None,
            "qa_scheduled_after_first_companion_finishes": difference >= -1e-6 if available else None,
            "clock_domain": "engine_core_monotonic",
            "role": "pressure_target" if slot == 4 else "first_slot_control"}


def _concurrent_case(path, reference, expected_ids, frozen, inputs_sha):
    errors, records = [], []
    value = _read(path)
    scenario_id, slot = frozen["case_id"], frozen["qa_slot"]
    if _sha(path) != reference.get("sha256"):
        errors.append("concurrent_case_sha256_mismatch")
    expected = {**expected_ids, "case_id": scenario_id, "qa_slot": slot,
                "qa_task_id": frozen["qa_task_id"], "measurement_count": 2,
                "concurrent_inputs_sha256": inputs_sha, "scenario_sha256": _canonical_sha(frozen)}
    for key, item in expected.items():
        if value.get(key) != item:
            errors.append("concurrent_case_" + key + "_mismatch")
    if value.get("collection_complete") is not True:
        errors.append("concurrent_case_collection_not_complete")
    samples = value.get("samples", [])
    if not isinstance(samples, list) or len(samples) != 2:
        errors.append("concurrent_sample_count_mismatch")
        samples = samples if isinstance(samples, list) else []
    if value.get("samples_sha256") != _canonical_sha(samples):
        errors.append("concurrent_samples_sha256_mismatch")
    for index, sample in enumerate(samples, 1):
        sample_errors = []
        if not isinstance(sample, dict):
            errors.append("concurrent_sample_not_object:" + str(index))
            continue
        if sample.get("iteration") != index or sample.get("qa_slot") != slot:
            sample_errors.append("iteration_or_qa_slot_mismatch")
        if not _finite(sample.get("elapsed_seconds"), True) or sample.get("request_count") != 4:
            sample_errors.append("batch_elapsed_or_request_count_invalid")
        outputs, qa_output = sample.get("outputs", []), sample.get("qa_output", {})
        if not isinstance(outputs, list) or len(outputs) != 4 or any(not isinstance(row, dict) for row in outputs):
            errors.append("concurrent_output_set_invalid:" + str(index))
            continue
        counts = []
        for request_index, output in enumerate(outputs, 1):
            tokens = output.get("token_ids")
            if (not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens)
                    or not isinstance(output.get("text"), str)):
                sample_errors.append("request_token_or_text_invalid:" + str(request_index))
                counts.append(None)
                continue
            counts.append(len(tokens))
            if request_index != slot and (len(tokens) != 128 or output.get("finish_reason") != "length"):
                sample_errors.append("companion_not_128_tokens:" + str(request_index))
        if sample.get("actual_generated_tokens") != counts:
            sample_errors.append("actual_generated_tokens_mismatch")
        if any(count is None for count in counts) or sample.get("total_generated_tokens") != sum(counts):
            sample_errors.append("total_generated_tokens_mismatch")
        if sample.get("outputs_sha256") != _canonical_sha([row.get("token_ids") for row in outputs]):
            sample_errors.append("concurrent_output_token_hash_mismatch")
        target = outputs[slot - 1]
        target_fields = ("text", "token_ids", "finish_reason", "stop_reason", "request_id", "request_metrics")
        if not isinstance(qa_output, dict) or any(qa_output.get(key) != target.get(key) for key in target_fields):
            sample_errors.append("qa_output_not_bound_to_target_slot")
        count = len(target.get("token_ids", []))
        logprobs = qa_output.get("selected_token_logprobs", qa_output.get("selected_logprobs", []))
        finite = (isinstance(logprobs, list) and len(logprobs) == count and count > 0
                  and all(type(number) in (int, float) and math.isfinite(number) for number in logprobs))
        normal = target.get("finish_reason") == "stop"
        count_valid = 1 <= count <= 128
        if qa_output.get("generated_token_count") != count:
            sample_errors.append("qa_generated_token_count_mismatch")
        if qa_output.get("logprobs_finite") is not finite or qa_output.get("normal_eos") is not normal:
            sample_errors.append("qa_generation_status_differs_from_saved_evidence")
        score = _score(target.get("text", ""), frozen["qa"])
        if qa_output.get("answer_correct") is not score["answer_correct"]:
            sample_errors.append("qa_worker_score_differs_from_frozen_expected")
        native = sample.get("preemptions", {})
        delta = native.get("delta")
        if delta is not None:
            before, after = native.get("before", {}), native.get("after", {})
            if (type(delta) is not int or delta < 0 or before.get("available") is not True
                    or after.get("available") is not True or type(before.get("value")) is not int
                    or type(after.get("value")) is not int or after["value"] - before["value"] != delta):
                sample_errors.append("native_preemption_counter_invalid")
        errors.extend("sample_" + str(index) + ":" + error for error in sample_errors)
        records.append({"iteration": index, "qa_slot": slot, "qa_task_id": frozen["qa_task_id"],
                        "text": target.get("text"), "token_ids": target.get("token_ids"),
                        "finish_reason": target.get("finish_reason"), "generated_token_count": count,
                        "normal_eos": normal, "token_count_valid": count_valid, "selected_logprobs_finite": finite,
                        "generation_valid": normal and finite and count_valid,
                        **score, "passed": score["answer_correct"] and normal and finite and count_valid,
                        "evidence_valid": not sample_errors,
                        "mixed_batch_elapsed_seconds": sample.get("elapsed_seconds"),
                        "pressure": _pressure(outputs, slot), "preemptions": native,
                        "evidence": sample})
    return {"case_id": scenario_id, "qa_slot": slot, "qa_task_id": frozen["qa_task_id"],
            "complete": len(records) == 2, "valid": not errors and len(records) == 2,
            "errors": errors, "samples": records,
            "correct_count": sum(row["answer_correct"] for row in records),
            "passed_count": sum(row["passed"] for row in records), "expected_count": 2,
            "all_correct_and_valid": len(records) == 2 and all(row["passed"] for row in records)}


def _concurrent_job(root, ledger, protocol, identities, job_id, inputs):
    errors, cases = [], {}
    binding = ledger.get("completed", {}).get(job_id)
    if not binding:
        return {"complete": False, "valid": False, "errors": [], "cases": {},
                "correct_count": 0, "expected_count": 8, "passed_count": 0, "passed": False}
    try:
        result_path = _within(root, binding["result_relative"])
        result = _read(result_path)
        envelope = result.get("concurrent_quality", {})
        digest = identities["hashes"].get("concurrent_quality_inputs_sha256")
        if envelope.get("inputs_sha256") != digest:
            errors.append("result_concurrent_inputs_sha256_mismatch")
        expected_ids = {"stage": STAGE, "job": job_id, "job_id": job_id, "backend": "FLASHINFER",
                        "phase": "capacity", "workspace_mib": 394,
                        "manifest_sha256": identities["hashes"].get("manifest_sha256"),
                        "protocol_sha256": identities["hashes"].get("protocol_sha256"),
                        "inputs_sha256": identities["hashes"].get("capacity_inputs_sha256")}
        references = envelope.get("cases", [])
        if (not isinstance(references, list) or len(references) != 4
                or {row.get("case_id") for row in references} != set(SCENARIOS)):
            raise ValueError("result_concurrent_case_set_mismatch")
        for reference in references:
            case_id = reference["case_id"]
            cases[case_id] = _concurrent_case(_within(result_path.parent, reference["path"]), reference,
                                             expected_ids, inputs["by_id"][case_id], digest)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
        errors.append(str(error))
    errors.extend(case_id + ":" + error for case_id, case in cases.items() for error in case["errors"])
    complete = len(cases) == 4 and all(case["complete"] for case in cases.values())
    return {"complete": complete, "valid": complete and not errors, "errors": errors, "cases": cases,
            "correct_count": sum(case["correct_count"] for case in cases.values()), "expected_count": 8,
            "passed_count": sum(case["passed_count"] for case in cases.values()),
            "passed": complete and not errors and all(case["all_correct_and_valid"] for case in cases.values())}


def _pair_quality(jobs):
    pairs, aggregate = {}, []
    for pair_id, (default_id, auto_id) in PAIRS.items():
        default, auto = jobs[default_id]["concurrent_quality"], jobs[auto_id]["concurrent_quality"]
        rows = []
        if default["complete"] and auto["complete"]:
            for scenario_id in SCENARIOS:
                left = {row["iteration"]: row for row in default["cases"][scenario_id]["samples"]}
                right = {row["iteration"]: row for row in auto["cases"][scenario_id]["samples"]}
                for iteration in (1, 2):
                    d, a = left[iteration], right[iteration]
                    assessed = d["evidence_valid"] and a["evidence_valid"]
                    dcorrect, acorrect = d["answer_correct"], a["answer_correct"]
                    row = {"pair": pair_id, "scenario_id": scenario_id, "repeat": iteration,
                           "default_job": default_id, "auto_job": auto_id,
                           "default_correct": d["answer_correct"], "auto_correct": a["answer_correct"],
                           "default_correct_and_generation_valid": d["passed"],
                           "auto_correct_and_generation_valid": a["passed"],
                           "assessed": assessed,
                           "default_correct_auto_wrong": dcorrect and not acorrect if assessed else None,
                           "auto_correct_default_wrong": acorrect and not dcorrect if assessed else None,
                           "both_wrong": not dcorrect and not acorrect if assessed else None,
                           "both_correct": dcorrect and acorrect if assessed else None,
                           "default_generation_valid_auto_invalid": d["generation_valid"] and not a["generation_valid"] if assessed else None,
                           "default_answer": d["text"], "auto_answer": a["text"],
                           "default_pressure": d["pressure"], "auto_pressure": a["pressure"]}
                    rows.append(row)
        baseline_correct = sum(row["default_correct"] for row in rows if row["assessed"])
        regressions = sum(row["default_correct_auto_wrong"] is True for row in rows)
        complete = len(rows) == 8 and all(row["assessed"] for row in rows)
        regression_free = regressions == 0 if complete and baseline_correct > 0 else None
        pairs[pair_id] = {"jobs": [default_id, auto_id], "complete": complete, "rows": rows,
                          "baseline_correct_count": baseline_correct,
                          "default_correct_auto_wrong_count": regressions,
                          "auto_correct_default_wrong_count": sum(row["auto_correct_default_wrong"] is True for row in rows),
                          "both_wrong_count": sum(row["both_wrong"] is True for row in rows),
                          "regression_free": regression_free,
                          "regression_status": "inconclusive" if regression_free is None else "preserved" if regression_free else "regression"}
        aggregate.extend(rows)
    complete = all(pair["complete"] for pair in pairs.values())
    baseline_correct = sum(row["default_correct"] for row in aggregate if row["assessed"])
    regressions = sum(row["default_correct_auto_wrong"] is True for row in aggregate)
    return {"complete": complete, "pairs": pairs, "paired_count": len(aggregate),
            "expected_pair_count": 16, "baseline_correct_count": baseline_correct,
            "default_correct_auto_wrong_count": regressions,
            "both_wrong_count": sum(row["both_wrong"] is True for row in aggregate),
            "auto_correct_default_wrong_count": sum(row["auto_correct_default_wrong"] is True for row in aggregate),
            "regression_free": regressions == 0 if complete and baseline_correct > 0 else None,
            "absolute_quality_requirement": "四个进程的四场景各重复两次，标准字段答对且正常结束、token概率有限；两边共同答错逐项保留"}


def _pressure_coverage(paired):
    rows = []
    for pair_id, pair in paired["pairs"].items():
        for row in pair["rows"]:
            default, auto = row["default_pressure"], row["auto_pressure"]
            control = SCENARIOS[row["scenario_id"]][1] == 1
            available = default["available"] and auto["available"]
            formed = (default["qa_scheduled_after_first_companion_finishes"] is True
                      and auto["qa_scheduled_before_first_companion_finishes"] is True) if available and not control else None
            rows.append({"pair": pair_id, "scenario_id": row["scenario_id"], "repeat": row["repeat"],
                         "role": "first_slot_control" if control else "pressure_target", "available": available,
                         "default": default, "auto": auto, "pressure_condition_covered": formed})
    pressure_rows = [row for row in rows if row["role"] == "pressure_target"]
    complete = paired["complete"] and len(pressure_rows) == 12
    available = complete and all(row["available"] for row in pressure_rows)
    covered = all(row["pressure_condition_covered"] is True for row in pressure_rows) if available else None
    return {"complete": complete, "native_admission_timestamps_available": available,
            "covered": covered, "covered_pair_count": sum(row["pressure_condition_covered"] is True for row in pressure_rows),
            "expected_pressure_pair_count": 12, "pairs": rows,
            "criterion": "slot4在默认配置中等至少一条伴随请求完成后入场，自动配置中提前入场；slot1仅作为位置参照"}


def _boundary(jobs):
    rows = {}
    for label, (default_id, auto_id) in PAIRS.items():
        default, auto = jobs[default_id], jobs[auto_id]
        dc, ac = default["kv"].get("token_capacity"), auto["kv"].get("token_capacity")
        ready = default["committed"] and auto["committed"] and _finite(dc, True) and _finite(ac, True)
        rows[label] = {"jobs": [default_id, auto_id], "status": "complete" if ready else "pending",
                       "default_capacity_tokens": dc, "candidate_capacity_tokens": ac,
                       "pressure_peak_tokens": 66048, "formed": dc < 66048 <= ac if ready else None}
    complete = all(row["status"] == "complete" for row in rows.values())
    return {"complete": complete, "formed": all(row["formed"] for row in rows.values()) if complete else None,
            "pairs": rows, "scope": "最大长度计算与实际QA入场时间分别保存"}


def _markdown(summary):
    def number(value, scale=1, digits=3):
        return "待采集" if value is None else f"{value / scale:.{digits}f}"
    lines = ["# S03-M06：并发长请求中的正确性验收", "",
             "研究类型：**1到2**。复用上游查询与启动自动适配，补齐取得容量收益时的标准答案检查。", "",
             summary["capacity"]["reason"] + "。", "",
             "## 标准题正确数", "",
             "| 进程 | 新并发题正确数 | 正确且正常结束/概率有效 | 旧短题 | 旧长题 |",
             "|---|---:|---:|---:|---:|"]
    for job_id, job in summary["jobs"].items():
        quality = job["concurrent_quality"]
        lines.append(f"| {job_id} | {quality['correct_count']}/8 | {quality['passed_count']}/8 | "
                     f"{job['short_quality']['score']}/3 | {job['long_quality']['score']}/3 |")
    paired = summary["concurrent_quality_comparison"]
    lines += ["", "## 逐题成对对照", "",
              "| 顺序 | 场景 | 重复 | 默认答对 | 自动答对 | 默认对/自动错 | 两边都错 |",
              "|---|---|---:|---|---|---|---|"]
    for label, pair in paired["pairs"].items():
        for row in pair["rows"]:
            lines.append(f"| {label} | {row['scenario_id']} | {row['repeat']} | {row['default_correct']} | "
                         f"{row['auto_correct']} | {row['default_correct_auto_wrong']} | {row['both_wrong']} |")
    lines += ["", f"默认答对而自动答错：{paired['default_correct_auto_wrong_count']} 次；"
              f"两边共同未通过：{paired['both_wrong_count']} 次；回答保持判断：{paired['regression_free']}。", "",
              "## 实际入场覆盖", "",
              "| 顺序 | 场景 | 重复 | 默认QA入场−最早伴随完成(ms) | 自动QA入场−最早伴随完成(ms) | 覆盖 |",
              "|---|---|---:|---:|---:|---|"]
    for row in summary["concurrent_pressure_coverage"]["pairs"]:
        lines.append(f"| {row['pair']} | {row['scenario_id']} | {row['repeat']} | "
                     f"{number(row['default']['qa_scheduled_minus_earliest_companion_last_seconds'], .001)} | "
                     f"{number(row['auto']['qa_scheduled_minus_earliest_companion_last_seconds'], .001)} | "
                     f"{row['pressure_condition_covered']} |")
    lines += ["", "slot1是位置参照；slot4依据原生同一时钟域的入场和完成时间检查实际负载。", "",
              "## 原128-token性能批次", "",
              "| 顺序 | 默认中位(ms) | 自动中位(ms) | 自动省时 |",
              "|---|---:|---:|---:|"]
    comparison = summary["comparisons"]["capacity"][PURE_CASE]
    for label, row in list(comparison["pairs"].items()) + [("pooled", comparison["pooled"])]:
        lines.append(f"| {label} | {number(row.get('default', {}).get('median_seconds'), .001)} | "
                     f"{number(row.get('candidate', {}).get('median_seconds'), .001)} | "
                     f"{number(row.get('candidate_gain_percent'))}% |")
    lines += ["", "本表只使用四条请求固定生成128token的性能批次；标准题正常结束后的混合批次耗时逐次保存在JSON中。", "",
              "## 启动与缓存", "",
              "| 进程 | 工作区(MiB) | 实际KV(MiB) | token容量 | 指针稳定 |",
              "|---|---:|---:|---:|---|"]
    for job_id, job in summary["jobs"].items():
        lines.append(f"| {job_id} | {number(job['actual_workspace'].get('total_bytes'), MIB)} | "
                     f"{number(job['kv'].get('total_tensor_bytes'), MIB)} | "
                     f"{number(job['kv'].get('token_capacity'), digits=0)} | {job['buffer_stable_after_graph_capture']} |")
    lines += ["", f"采集完整：{summary['collection_complete']}；证据有效：{summary['evidence_valid']}；"
              f"新题正确性：{summary['new_quality_passed']}；成对无退化：{summary['regression_free']}；"
              f"入场条件覆盖：{summary['pressure_covered']}；性能批次省时：{summary['performance_net_benefit_observed']}。", "",
              "R01原失败关口、M02至M05原结果和旧长题错误按冻结文件保留。"]
    if summary["evidence_errors"]:
        lines += ["", "待核实项："] + ["- " + error for error in summary["evidence_errors"]]
    lines += ["", f"GPU本轮 {summary['gpu_charged_seconds']:.1f}/{summary['gpu_budget_seconds']:.1f} 秒；"
              f"剩余 {summary['gpu_remaining_seconds']:.1f} 秒；S03累计 "
              f"{summary['combined_gpu_observed_seconds']:.1f}/{summary['combined_gpu_budget_seconds']:.1f} 秒。"]
    return "\n".join(lines) + "\n"


def summarize(root, ledger, protocol):
    root, helper = Path(root), _base(root)
    identities = helper._identities(root, ledger, protocol)
    inputs = _concurrent_inputs(root, protocol, identities)
    previous = helper._prior(protocol)
    priors = {"m02": helper._prior_m02(root, protocol), "m03": helper._prior_m03(root, protocol),
              "m04": helper._prior_m04(root, protocol), "m05": _prior_m05(root, protocol)}
    jobs = {job_id: helper._job(root, ledger, protocol, identities, job_id) for job_id in JOB_SPECS}
    for job_id, job in jobs.items():
        job["concurrent_quality"] = _concurrent_job(root, ledger, protocol, identities, job_id, inputs)
    comparisons = {PURE_CASE: helper._compare(jobs, PURE_CASE)}
    resources, boundary = helper._resources(jobs, protocol), _boundary(jobs)
    equivalent, paired = helper._quality_equivalence(jobs), _pair_quality(jobs)
    coverage = _pressure_coverage(paired)
    complete = all(job["committed"] for job in jobs.values())
    evidence_errors = identities["errors"] + inputs["errors"] + previous["errors"]
    evidence_errors += [name + ":" + error for name, prior in priors.items() for error in prior["errors"]]
    evidence_errors += [job_id + ":" + error for job_id, job in jobs.items() for error in job["errors"]]
    evidence_errors += [job_id + ":" + error for job_id, job in jobs.items()
                        for case in job["cases"].values() for error in case["errors"]]
    evidence_errors += [job_id + ":" + error for job_id, job in jobs.items() for error in job["concurrent_quality"]["errors"]]
    charged = sum(row.get("charged_seconds", 0) for row in ledger.get("attempts", []))
    reserved = sum(row.get("reserved_seconds", 0) for row in ledger.get("attempts", [])
                   if row.get("state") in ("running", "active"))
    parent_fields = {name: protocol.get(name, value) for name, value in {
        "parent_gpu_charged_seconds": 710.596803747, "parent_m01_gpu_charged_seconds": 688.270119219,
        "parent_r01_gpu_charged_seconds": 379.564977797, "parent_m02_gpu_charged_seconds": 464.609150575,
        "parent_m03_gpu_charged_seconds": 437.887388085, "parent_m04_gpu_charged_seconds": 442.644768586,
        "parent_m05_gpu_charged_seconds": 968.069123306}.items()}
    historical, budget = sum(parent_fields.values()), protocol.get("gpu_budget_seconds", 1200)
    approved = protocol.get("approved_combined_gpu_budget_seconds", APPROVED_COMBINED)
    if (not all(_finite(value) for value in (charged, reserved, historical, budget, approved))
            or abs(historical - PARENT_CHARGED) > 1e-6 or abs(budget - 1200) > 1e-6
            or abs(approved - APPROVED_COMBINED) > 1e-6
            or charged + reserved > budget + 1e-6 or historical + charged > approved + 1e-6):
        evidence_errors.append("gpu_budget_or_approved_parent_chain_limit_exceeded")
    collection_complete = complete and all(job["concurrent_quality"]["complete"] for job in jobs.values())
    evidence_valid = collection_complete and not evidence_errors and all(
        job["evidence_valid"] and job["concurrent_quality"]["valid"] for job in jobs.values())
    short_passed = complete and all(job["short_quality"]["passed"] for job in jobs.values())
    long_passed = complete and all(job["long_quality"]["passed"] for job in jobs.values())
    startup_passed = evidence_valid and resources["passed"] and short_passed and equivalent["passed"]
    new_quality = evidence_valid and all(job["concurrent_quality"]["passed"] for job in jobs.values())
    checked = list(comparisons[PURE_CASE]["pairs"].values()) + [comparisons[PURE_CASE]["pooled"]]
    faster = all(row["status"] == "complete" and row["difference_seconds"] > 0 for row in checked)
    perf = (startup_passed and boundary["formed"] is True and faster) if collection_complete else None
    passed = startup_passed and new_quality and paired["regression_free"] is True and coverage["covered"] is True
    if not collection_complete:
        reason = "采集尚未完成，已完成记录保存，可继续未提交进程"
    elif not evidence_valid:
        reason = "记录已保存，冻结来源、字段绑定或实际分配证据存在待核实项"
    elif not new_quality:
        reason = "采集完整，新并发标准题含未答对或生成有效性未通过项，正确数与成对退化逐题保留"
    elif paired["regression_free"] is not True:
        reason = "新题记录完整，成对无退化判断尚有未通过或证据不足项"
    elif coverage["covered"] is not True:
        reason = "新题正确性检查完成，预定默认延后与自动提前入场条件覆盖尚未全部形成"
    elif not startup_passed:
        reason = "新题检查完成，启动适配、空间回收或旧六题保持关口存在未通过项"
    else:
        reason = "并发长请求标准题正确且正常结束，自动提前入场条件形成，成对检查保持正确"
    pooled = comparisons[PURE_CASE]["pooled"]
    capacity = {"status": "complete" if collection_complete else "pending", "passed": passed, "reason": reason,
                "capacity_increased": resources["capacity_increased"], "time_saved": perf,
                "net_benefit_supported": passed and perf is True, "pairs": resources["pairs"],
                "short_quality_passed": short_passed, "long_quality_passed": long_passed,
                "quality_equivalence_passed": equivalent["passed"],
                "pressure_pooled_time_saved_seconds": pooled["difference_seconds"],
                "pressure_pooled_gain_percent": pooled["candidate_gain_percent"]}
    summary = {"schema_version": 1, "stage": STAGE, "research_kind": "1_to_2",
               "updated_utc": datetime.now(timezone.utc).isoformat(), "passed": passed,
               "collection_complete": collection_complete, "evidence_valid": evidence_valid,
               "startup_adaptation_passed": startup_passed,
               "completed_jobs": sum(job["committed"] for job in jobs.values()), "expected_jobs": 4,
               "valid_jobs": sum(job["evidence_valid"] and job["concurrent_quality"]["valid"] for job in jobs.values()),
               "jobs": jobs, "identities": identities, "concurrent_inputs": {key: value for key, value in inputs.items() if key not in ("value", "by_id")},
               "frozen_helper": {"source_stage": "FI-S03-M05", "analyzer_sha256": BASE_ANALYZER_SHA256},
               "comparisons": {"capacity": comparisons}, "resources": resources, "capacity": capacity,
               "capacity_boundary": boundary, "capacity_boundary_formed": boundary["formed"],
               "concurrent_quality_comparison": paired, "concurrent_pressure_coverage": coverage,
               "new_quality_passed": new_quality, "regression_free": paired["regression_free"],
               "pressure_covered": coverage["covered"], "short_quality_passed": short_passed,
               "long_quality_passed": long_passed, "quality_equivalence": equivalent,
               "previous_evidence": previous, **{"previous_" + name + "_evidence": value for name, value in priors.items()},
               "evidence_errors": evidence_errors, "long_outputs": helper._outputs(jobs),
               "performance_net_benefit_observed": perf,
               "pressure_time_saved_both_orders": faster if collection_complete else None,
               "net_benefit_supported": passed and perf is True,
               "net_benefit_scope": "startup_auto_kv_16k_four_request_correctness_and_separate_pure_performance",
               "timing_scope": "原固定128token性能批次与正常结束标准题批次分别记录；标准题耗时不并入性能收益",
               "gpu_charged_seconds": charged, "gpu_reserved_seconds": reserved,
               "gpu_budget_seconds": budget, "gpu_remaining_seconds": max(0, budget - charged - reserved),
               **parent_fields, "historical_gpu_charged_seconds": historical,
               "combined_gpu_observed_seconds": historical + charged,
               "combined_gpu_budget_seconds": approved, "approved_combined_gpu_budget_seconds": approved,
               "historical_short_request_gate_retained": True,
               "facts": {"new_quality_passed": new_quality, "regression_free": paired["regression_free"],
                         "pressure_covered": coverage["covered"], "performance_batch_time_saved": perf,
                         "legacy_long_quality_passed": long_passed,
                         "old_six_quality_unchanged_from_default": equivalent["passed"],
                         "kv_capacity_increased": resources["capacity_increased"]}}
    _write(root / "summary.json", summary)
    temporary = root / "summary.md.tmp"
    temporary.write_text(_markdown(summary), encoding="utf-8")
    temporary.replace(root / "summary.md")
    return summary


def generate_summary(root):
    root = Path(root)
    return summarize(root, _read(root / "ledger.json"), _read(root / "protocol.json"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    value = generate_summary(parser.parse_args().root)
    print(json.dumps({key: value[key] for key in ("collection_complete", "evidence_valid", "completed_jobs",
                                                "new_quality_passed", "regression_free", "pressure_covered")}))
