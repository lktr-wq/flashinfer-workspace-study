"""Recompute the frozen M06 evidence using Python's standard library only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
JOB_IDS = ("01_DEFAULT", "02_STARTUP_AUTO", "03_STARTUP_AUTO", "04_DEFAULT")
PAIRS = (("first", JOB_IDS[0], JOB_IDS[1]), ("reverse", JOB_IDS[3], JOB_IDS[2]))
CASE_IDS = ("CQ_LOOKUP_SLOT4", "CQ_LATEST_SLOT4", "CQ_COMBINE_SLOT4", "CQ_LOOKUP_SLOT1")
MIB = 1 << 20


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def child(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def score(text, expected):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        answer = json.loads(text)
    except (ValueError, IndexError):
        return False
    return isinstance(answer, dict) and all(
        key in answer and type(answer[key]) is type(value) and answer[key] == value
        for key, value in expected.items())


def recompute(source, lock):
    verified = set()

    def check(relative, expected):
        path = child(source, relative)
        require(digest(path) == expected, "SHA256 mismatch: " + relative)
        verified.add(relative)
        return path

    for name, expected in lock["files"].items():
        check(name, expected)
    manifest, ledger, summary = (read(source / name) for name in
                                ("manifest.json", "ledger.json", "summary.json"))
    for relative, expected in manifest["files"].items():
        check(relative, expected)
    require(set(ledger["completed"]) == set(JOB_IDS), "Expected four committed jobs")
    identities = {name + "_sha256": digest(source / (name + ".json"))
                  for name in ("manifest", "protocol", "inputs")}
    raw, performance, quality, files = {}, {}, {}, {}
    for job_id in JOB_IDS:
        commit = ledger["completed"][job_id]
        result_path = check(commit["result_relative"], commit["sha256"])
        attempt = result_path.parent
        result = read(result_path)
        receipt_path = attempt / "guardian-receipt.json"
        receipt = read(receipt_path)
        verified.add(str(receipt_path.relative_to(source)).replace("\\", "/"))
        require(receipt["passed"] is True and receipt["result_sha256"] == commit["sha256"],
                "Guardian receipt does not bind committed result: " + job_id)
        require(receipt["identities"] == identities, "Receipt identity mismatch: " + job_id)
        require(result["job_id"] == job_id and all(result[k] == v for k, v in identities.items()),
                "Result identity mismatch: " + job_id)
        for field, hash_field in (("workerpath_relative", "workerpath_sha256"),):
            relative = str((attempt / result[field]).relative_to(source)).replace("\\", "/")
            check(relative, result[hash_field])
        case = next(row for row in result["cases"] if row["case_id"] == "L16384_B4")
        relative = str((attempt / case["path"]).relative_to(source)).replace("\\", "/")
        data = read(check(relative, case["sha256"]))
        require(data["warmup_count"] == 2 and len(data["samples"]) == 5,
                "Unexpected measurement count: " + job_id)
        require(all(row["actual_generated_tokens"] == [128] * 4 for row in data["samples"]),
                "Performance output count changed: " + job_id)
        performance[job_id] = data["samples"]
        quality[job_id] = {}
        for entry in result["concurrent_quality"]["cases"]:
            relative = str((attempt / entry["path"]).relative_to(source)).replace("\\", "/")
            qa = read(check(relative, entry["sha256"]))
            require(qa["collection_complete"] and len(qa["samples"]) == 2,
                    "Incomplete QA scenario: " + job_id + "/" + entry["case_id"])
            quality[job_id][entry["case_id"]] = qa
        require(set(quality[job_id]) == set(CASE_IDS), "Unexpected QA scenarios")
        raw[job_id] = result
        files[job_id] = commit["result_relative"]
    comparisons = {}
    for label, left, right in PAIRS + (("pooled", None, None),):
        defaults = [row["elapsed_seconds"] for job in ((left,) if left else (JOB_IDS[0], JOB_IDS[3]))
                    for row in performance[job]]
        autos = [row["elapsed_seconds"] for job in ((right,) if right else (JOB_IDS[1], JOB_IDS[2]))
                 for row in performance[job]]
        d, a = statistics.median(defaults), statistics.median(autos)
        comparisons[label] = {"default_median_seconds": d, "auto_median_seconds": a,
                              "saved_seconds": d - a, "saved_percent": (d - a) / d * 100,
                              "default_samples": defaults, "auto_samples": autos}
    paired, counts = [], {"default": {"correct": 0, "total": 0}, "auto": {"correct": 0, "total": 0}}
    for label, left, right in PAIRS:
        for case_id in CASE_IDS:
            dl, ar = quality[left][case_id], quality[right][case_id]
            require(dl["expected"] == ar["expected"], "QA expected answers differ")
            for d, a in zip(dl["samples"], ar["samples"], strict=True):
                dq, aq = d["qa_output"], a["qa_output"]
                dc, ac = score(dq["text"], dl["expected"]), score(aq["text"], ar["expected"])
                for mode, correct in (("default", dc), ("auto", ac)):
                    counts[mode]["correct"] += int(correct)
                    counts[mode]["total"] += 1
                paired.append({"pair": label, "case_id": case_id, "iteration": d["iteration"],
                    "same_text": dq["text"] == aq["text"], "same_tokens": dq["token_ids"] == aq["token_ids"],
                    "default_correct": dc, "auto_correct": ac,
                    "normal_eos_both": all(q["finish_reason"] == "stop" and
                        q["token_ids"][-1] == 151645 for q in (dq, aq))})
    capacities = [{"job_id": job, "workspace_mib": raw[job]["actual_workspace"]["total_bytes"] / MIB,
                   "kv_mib": raw[job]["kv"]["total_tensor_bytes"] / MIB,
                   "token_capacity": raw[job]["kv"]["token_capacity"],
                   "startup_query_calls": raw[job]["startup_workspace"]["worker_health"]["query_calls"]}
                  for job in JOB_IDS]
    default_cap, auto_cap = capacities[0]["token_capacity"], capacities[1]["token_capacity"]
    latency = {}
    for mode, jobs in (("default", (JOB_IDS[0], JOB_IDS[3])), ("auto", (JOB_IDS[1], JOB_IDS[2]))):
        samples = [row for job in jobs for row in performance[job]]
        latency[mode] = statistics.median(row["outputs"][0]["request_metrics"]["derived"]
                        ["engine_queued_to_last_token_seconds"] for row in samples)
    pooled = comparisons["pooled"]
    saved = summary["comparisons"]["capacity"]["L16384_B4"]["pooled"]
    require(math.isclose(pooled["saved_percent"], saved["candidate_gain_percent"], abs_tol=1e-9),
            "Recomputed performance differs from frozen summary")
    return {"kind": "frozen_evidence_recomputation", "source_stage": "FI-S03-M06",
            "recomputed_utc": datetime.now(timezone.utc).isoformat(), "new_gpu_run": False,
            "integrity_passed": True, "verified_file_count": len(verified),
            "verified_files": sorted(verified), "committed_results": files,
            "performance": comparisons, "capacities": capacities,
            "token_capacity_increase_percent": (auto_cap - default_cap) / default_cap * 100,
            "request1_completion_median_seconds": latency,
            "request1_completion_increase_seconds": latency["auto"] - latency["default"],
            "quality_counts": counts, "qa_pairs": paired,
            "all_paired_text_and_tokens_equal": all(p["same_text"] and p["same_tokens"] for p in paired),
            "default_correct_count": sum(p["default_correct"] for p in paired),
            "default_correct_auto_wrong": sum(p["default_correct"] and not p["auto_correct"] for p in paired),
            "original_flags": {k: summary[k] for k in ("passed", "new_quality_passed", "regression_free",
                "collection_complete", "evidence_valid", "pressure_covered", "net_benefit_supported")}}


def main():
    bundled = HERE / "evidence" / "M06"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=bundled if bundled.is_dir() else Path("F:/FlashInferResearch/S03-M06"))
    parser.add_argument("--out", type=Path, default=Path("F:/FlashInferResearch/S03-M07/evidence-replay") if sys.platform == "win32" else HERE / "evidence-replay")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        report = recompute(args.source.resolve(), read(HERE / "evidence-lock.json"))
        (args.out / "recomputed.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
            allow_nan=False) + "\n", encoding="utf-8")
        p = report["performance"]["pooled"]
        text = ("# 已保存结果复算\n\n"
                "从已保存的原始样本重新计算；本次使用Python标准库，GPU新增运行次数为0。\n\n"
                f"- 来源文件核验：{report['verified_file_count']}份。\n"
                f"- 整批中位耗时：{p['default_median_seconds']:.6f} → {p['auto_median_seconds']:.6f}秒，缩短{p['saved_percent']:.4f}%。\n"
                f"- 历史token容量增加：{report['token_capacity_increase_percent']:.4f}%。\n"
                f"- 首条请求完成时间增加：{report['request1_completion_increase_seconds']:.6f}秒。\n"
                "- 标准题：默认4/16、自动4/16；16组成对文本与token一致。\n"
                "- 原全题答对验收：保留失败，详见recomputed.json中的original_flags。\n")
        (args.out / "复算结果.md").write_text(text, encoding="utf-8")
        print(text)
        print("结果目录：" + str(args.out))
        return 0
    except Exception as error:
        failure = {"integrity_passed": False, "error": type(error).__name__ + ": " + str(error),
                   "new_gpu_run": False}
        (args.out / "error.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        print(failure["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
