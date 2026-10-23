#!/usr/bin/env python3
"""Freeze four-request 16K correctness inputs using the existing CPU tokenizer."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time


ROOT = Path(__file__).resolve().parent
STAGE = "FI-S03-M06"
TARGET = 16384
LIMIT_SECONDS = 300
SOURCE_HELPER_SHA256 = "f4cfa46a27b249fff1014f6b37bd412bbf6efcf44e0442be88cf5fefe1ab82be"
SOURCE_CAPACITY_SHA256 = "d46f2ff718ceee440b2d84c58f172acd30943a64c8136cee2a5e041dfb19edeb"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def hard_timeout(_signum, _frame):
    raise TimeoutError("CPU concurrent-quality preparation exceeded 300 seconds")


def exact_long_prompt(tokenizer, helper, definition):
    # M03's complete-record balancing method; its 512-record ceiling covered 8K.
    lo, hi = 0, 1024
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(helper.encode(tokenizer, helper.make_prompt(definition, mid))) <= TARGET - 96:
            lo = mid
        else:
            hi = mid - 1
    count = lo
    lo, hi = 0, 128
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(helper.encode(tokenizer, helper.make_prompt(definition, count, mid))) <= TARGET - 4:
            lo = mid
        else:
            hi = mid - 1
    for digits in range(1, 33):
        prompt = helper.make_prompt(definition, count, lo, digits)
        tokens = helper.encode(tokenizer, prompt)
        if len(tokens) == TARGET:
            return prompt, tokens, count
    raise RuntimeError("Unable to balance the complete chat prompt to 16384 tokens")


def definitions():
    return [
        {
            "case_id": "QUALITY_CONCURRENT_LOOKUP",
            "title": "指定订单字段查找",
            "question": "请从下方资料查找唯一订单ORDER-A7319的owner和count。只输出一个JSON对象，字段为owner和count。owner必须逐字复制记录中的ASCII代号，count必须是JSON整数。不要解释。",
            "closing": "请回答ORDER-A7319的owner和count，只输出JSON对象。",
            "evidence": [{
                "id": "lookup_target", "fraction": 0.16,
                "text": "目标订单ORDER-A7319：owner=OWNER_R7，count=47，存放在南库。",
                "source_record": {"order": "ORDER-A7319", "owner": "OWNER_R7", "count": 47},
            }],
            "expected": {"owner": "OWNER_R7", "count": 47},
            "expected_json_types": {"owner": "string", "count": "integer"},
        },
        {
            "case_id": "QUALITY_CONCURRENT_LATEST",
            "title": "指定订单最新版本状态",
            "question": "下方资料含有多个订单。只查订单ORDER-B8421，以version数字最大的记录为最新记录。只输出JSON对象，字段为version和status。version必须是JSON整数，status必须逐字复制该记录中的ASCII状态。不要解释。",
            "closing": "请回答ORDER-B8421最新记录的version和status，只输出JSON对象。",
            "evidence": [
                {"id": "latest_version_1", "fraction": 0.09,
                 "text": "目标订单ORDER-B8421：version=1，status=PENDING。",
                 "source_record": {"order": "ORDER-B8421", "version": 1, "status": "PENDING"}},
                {"id": "latest_version_2", "fraction": 0.50,
                 "text": "目标订单ORDER-B8421：version=2，status=CONFIRMED。",
                 "source_record": {"order": "ORDER-B8421", "version": 2, "status": "CONFIRMED"}},
                {"id": "latest_version_3", "fraction": 0.88,
                 "text": "目标订单ORDER-B8421：version=3，status=SHIPPED。",
                 "source_record": {"order": "ORDER-B8421", "version": 3, "status": "SHIPPED"}},
            ],
            "expected": {"version": 3, "status": "SHIPPED"},
            "expected_json_types": {"version": "integer", "status": "string"},
        },
        {
            "case_id": "QUALITY_CONCURRENT_COMBINE",
            "title": "关联同一订单的两处字段",
            "question": "下方资料把订单ORDER-C9532的owner和count分开登记在两条记录里。请查找这两条记录并合并回答。只输出JSON对象，字段为owner和count。owner必须逐字复制ASCII代号，count必须是JSON整数。不要解释。",
            "closing": "请合并ORDER-C9532两条记录中的owner和count，只输出JSON对象。",
            "evidence": [
                {"id": "combine_owner", "fraction": 0.25,
                 "text": "目标订单ORDER-C9532人员登记：owner=OWNER_T4。",
                 "source_record": {"order": "ORDER-C9532", "owner": "OWNER_T4"}},
                {"id": "combine_count", "fraction": 0.75,
                 "text": "目标订单ORDER-C9532数量登记：count=29。",
                 "source_record": {"order": "ORDER-C9532", "count": 29}},
            ],
            "expected": {"owner": "OWNER_T4", "count": 29},
            "expected_json_types": {"owner": "string", "count": "integer"},
        },
    ]


def main():
    started = time.monotonic()
    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError("Run CPU preparation in WSL/Linux for the 300-second hard limit")
    signal.signal(signal.SIGALRM, hard_timeout)
    signal.alarm(LIMIT_SECONDS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "ledger.json").exists():
        raise RuntimeError("An M06 ledger exists; preserve the frozen inputs")
    protocol_path = args.protocol or root / "protocol.json"
    protocol = read_json(protocol_path)
    if protocol.get("stage") != STAGE:
        raise RuntimeError("Only the approved M06 stage may prepare these inputs")
    engine = protocol["engine"]
    if engine.get("max_model_len") != 16512 or engine.get("max_num_seqs") != 4:
        raise RuntimeError("M06 requires the existing 16512-token, four-request service")
    helper_path = root.parent / "S03-M03" / "prepare_quality.py"
    capacity_path = root.parent / "S03-M05" / "capacity-inputs.json"
    if sha256(helper_path) != SOURCE_HELPER_SHA256:
        raise RuntimeError("Frozen M03 exact-length helper changed")
    if sha256(capacity_path) != SOURCE_CAPACITY_SHA256:
        raise RuntimeError("Frozen M05 companion inputs changed")
    spec = importlib.util.spec_from_file_location("_m06_m03_quality_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.TARGET = TARGET
    capacity = read_json(capacity_path)
    pressure = [case for case in capacity["cases"] if case["case_id"] == "L16384_B4"]
    if len(pressure) != 1 or len(pressure[0]["prompts"]) != 4:
        raise RuntimeError("Expected exactly four frozen M05 pressure prompts")
    original_rows = pressure[0]["prompts"]

    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", USE_TF="0", USE_FLAX="0")
    # Use the actual installed tokenizer and its normal dependencies.
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation started with CUDA initialized")
    from transformers import AutoTokenizer
    model = Path(protocol["model_linux"])
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    config = read_json(model / "config.json")
    vocabulary = int(config["vocab_size"])
    qa_tasks = []
    for definition in definitions():
        prompt, tokens, count = exact_long_prompt(tokenizer, helper, definition)
        task = {
            "task_id": definition["case_id"], "case_id": definition["case_id"],
            "title": definition["title"], "scope": "concurrent_long",
            "source_text": prompt, "prompt": prompt,
            "chat_prompt": helper.render(tokenizer, prompt), "prompt_token_ids": tokens,
            "input_token_count": len(tokens), "prompt_token_sha256": token_hash(tokens),
            "score_mode": "json_fields", "expected": definition["expected"],
            "expected_json_types": definition["expected_json_types"],
            "max_tokens": 128, "ignore_eos": False,
            "background_record_count": count,
            "evidence_locations": helper.evidence_locations(tokenizer, prompt, definition["evidence"]),
        }
        qa_tasks.append(task)

    companion_sampling = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 128,
                          "ignore_eos": True, "detokenize": True, "seed": 20261022}
    qa_sampling = {**companion_sampling, "ignore_eos": False, "logprobs": 1, "seed": 20261023}
    scenarios = [("CQ_LOOKUP_SLOT4", 0, 4), ("CQ_LATEST_SLOT4", 1, 4),
                 ("CQ_COMBINE_SLOT4", 2, 4), ("CQ_LOOKUP_SLOT1", 0, 1)]
    cases = []
    for case_id, task_index, qa_slot in scenarios:
        qa = copy.deepcopy(qa_tasks[task_index])
        requests = []
        for index in range(4):
            is_qa = index == qa_slot - 1
            tokens = qa["prompt_token_ids"] if is_qa else original_rows[index]
            requests.append({
                "role": "qa" if is_qa else "companion", "slot": index + 1,
                "prompt_token_ids": copy.deepcopy(tokens), "input_token_count": len(tokens),
                "prompt_token_sha256": token_hash(tokens),
                "sampling": copy.deepcopy(qa_sampling if is_qa else companion_sampling),
                "source_ref": ({"source": "predeclared_quality_task", "task_id": qa["task_id"]}
                               if is_qa else {"source": str(capacity_path),
                                              "source_sha256": SOURCE_CAPACITY_SHA256,
                                              "case_id": "L16384_B4", "slot": index + 1}),
            })
        cases.append({
            "case_id": case_id, "input_tokens_per_request": TARGET, "batch_size": 4,
            "qa_task_id": qa["task_id"], "qa_slot": qa_slot,
            "qa": qa, "requests": requests, "measurement_repeats": 2,
            "role_note": ("Quality request replaces the fourth-arriving request whose admission changed in M05"
                          if qa_slot == 4 else "Same lookup task in the first slot; early EOS may end capacity pressure"),
        })

    for case in cases:
        assert case["requests"][case["qa_slot"] - 1]["role"] == "qa"
        assert sum(row["role"] == "qa" for row in case["requests"]) == 1
        for row in case["requests"]:
            assert row["input_token_count"] == TARGET and len(row["prompt_token_ids"]) == TARGET
            assert all(type(token) is int and 0 <= token < vocabulary for token in row["prompt_token_ids"])
            assert row["sampling"]["ignore_eos"] is (row["role"] == "companion")
            assert row["sampling"]["max_tokens"] == 128
        qa = case["qa"]
        assert set(qa["expected"]) == set(qa["expected_json_types"])
        for key, kind in qa["expected_json_types"].items():
            assert type(qa["expected"][key]) is (int if kind == "integer" else str)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA unexpectedly initialized during CPU input preparation")
    result = {
        "schema": 1, "stage": STAGE, "revision": "concurrent_correctness_v1",
        "classification": "1_to_2", "measurement_repeats": 2, "cases": cases,
        "tasks": qa_tasks, "qa_slot_indexing": "one_based", "vocab_size": vocabulary,
        "system_prompt": helper.SYSTEM, "long_input_tokens": TARGET,
        "source_exact_length_helper": str(helper_path), "source_helper_sha256": SOURCE_HELPER_SHA256,
        "exact_length_adjustment": "M03 complete-record method with search ceiling 1024 records for 16K",
        "source_capacity_path": str(capacity_path), "source_capacity_sha256": SOURCE_CAPACITY_SHA256,
        "companion_policy": "Use the exact M05 L16384_B4 prompt at each unchanged slot",
        "target_policy": "Three targets and expected JSON values frozen before any M06 model execution",
        "tokenizer_model": str(model), "local_files_only": True,
        "cpu_preparation_limit_seconds": LIMIT_SECONDS, "torch_cuda_initialized": False,
    }
    output = root / "concurrent-quality-inputs.json"
    write_json(output, result)
    report = {
        "stage": STAGE, "output": str(output), "inputs_sha256": sha256(output),
        "cpu_preparation_elapsed_seconds": round(time.monotonic() - started, 6),
        "cpu_preparation_limit_seconds": LIMIT_SECONDS, "torch_imported": "torch" in sys.modules,
        "torch_cuda_initialized": False, "gpu_used": False,
        "source_helper_sha256": SOURCE_HELPER_SHA256, "source_capacity_sha256": SOURCE_CAPACITY_SHA256,
        "cases": [{"case_id": case["case_id"], "qa_task_id": case["qa_task_id"],
                   "qa_slot": case["qa_slot"], "input_lengths": [len(row["prompt_token_ids"]) for row in case["requests"]],
                   "expected": case["qa"]["expected"], "evidence_locations": case["qa"]["evidence_locations"]}
                  for case in cases],
    }
    write_json(root / "concurrent-quality-preparation.json", report)
    signal.alarm(0)
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
