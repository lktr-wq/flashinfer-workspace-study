"""Offline S03-M06 workspace-memory controller. All GPU work is started by the Windows launcher."""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
STAGE = "FI-S03-M06"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def child_path(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise RuntimeError("Expected relative evidence path")
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def identities(root):
    return {name + "_sha256": digest(root / (name + ".json"))
            for name in ("manifest", "protocol", "inputs")}


def verify_files(root):
    for name, expected in read_json(root / "manifest.json")["files"].items():
        if digest(child_path(root, name)) != expected:
            raise RuntimeError("Frozen S03-M06 file changed: " + name)
    return identities(root)


def validate_protocol(protocol):
    expected = [("01_DEFAULT", "capacity", 394), ("02_STARTUP_AUTO", "capacity", 394),
                ("03_STARTUP_AUTO", "capacity", 394), ("04_DEFAULT", "capacity", 394)]
    if protocol.get("stage") != STAGE or protocol.get("research_kind") != "1_to_2":
        raise RuntimeError("Unexpected automatic-KV stage")
    if [(j.get("id"), j.get("phase"), j.get("workspace_mib")) for j in protocol.get("jobs", [])] != expected:
        raise RuntimeError("Unexpected automatic-KV comparison order")
    if any(j.get("backend") != "FLASHINFER" for j in protocol["jobs"]):
        raise RuntimeError("Unexpected attention backend")
    budget = protocol.get("gpu_budget_seconds")
    if type(budget) not in (int, float) or not math.isfinite(budget) or not 0 < budget <= 1200:
        raise RuntimeError("GPU-stage budget exceeds the approved 1200-second allowance")
    if protocol.get("per_job_hard_limit_seconds") != 330 or not 0 < int(protocol["storage_limit_bytes"]) <= 1024**3:
        raise RuntimeError("Unexpected per-job or storage limit")
    phase = protocol.get("phases", {}).get("capacity", {})
    if set(protocol.get("phases", {})) != {"capacity"} or phase.get("cases") != [{
        "case_id": "L16384_B4", "input_tokens_per_request": 16384,
        "batch_size": 4, "output_tokens_per_request": 128}]:
        raise RuntimeError("Unexpected performance workload")
    if phase.get("warmup_iterations") != 2 or phase.get("measurement_iterations") != 5:
        raise RuntimeError("Unexpected repetition counts")
    if protocol.get("quality_case_count") != 6 or protocol.get("long_quality_case_count") != 3:
        raise RuntimeError("Unexpected reference-answer quality tasks")
    for engine in (protocol["engine"], phase["engine"]):
        if engine.get("kv_cache_memory_bytes") is not None or engine.get("gpu_memory_utilization") != 0.65:
            raise RuntimeError("Automatic KV allocation and the common memory budget must stay enabled")
    if [j.get("startup_mode") for j in protocol["jobs"]] != ["default", "auto", "auto", "default"]:
        raise RuntimeError("Unexpected startup selection order")
    if [j.get("batch") for j in protocol["jobs"]] != [1, 1, 2, 2]:
        raise RuntimeError("Unexpected case ordering")
    for engine in (protocol["engine"], phase["engine"]):
        if (engine.get("max_model_len") != 16512 or engine.get("max_num_seqs") != 4
                or engine.get("max_num_batched_tokens") != 512):
            raise RuntimeError("Unexpected capacity-boundary engine limits")
    for job in protocol["jobs"]:
        if job.get("engine_overrides") != {"kv_cache_memory_bytes": None}:
            raise RuntimeError("A job overrides native automatic KV allocation")
        if protocol["runtime_env"][job["id"]].get("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS") != "1":
            raise RuntimeError("Native graph memory estimation must be enabled")

def phase_cases(protocol, job):
    return protocol["phases"][job["phase"]]["cases"]


def job_identity(job):
    return {"job": job["id"], "backend": job["backend"],
            "phase": job["phase"], "workspace_mib": job["workspace_mib"]}


def verify_parent(protocol):
    specifications = [
        ("S03", "FI-S03", "parent_s03_ledger", "parent_gpu_charged_seconds", "parent_gpu_budget_seconds"),
        ("M01", "FI-S03-M01", "parent_m01_ledger", "parent_m01_gpu_charged_seconds", "parent_m01_gpu_budget_seconds"),
        ("R01", "FI-S03-M01-R01", "parent_r01_ledger", "parent_r01_gpu_charged_seconds", "parent_r01_gpu_budget_seconds"),
        ("M02", "FI-S03-M02", "parent_m02_ledger", "parent_m02_gpu_charged_seconds", "parent_m02_gpu_budget_seconds"),
        ("M03", "FI-S03-M03", "parent_m03_ledger", "parent_m03_gpu_charged_seconds", "parent_m03_gpu_budget_seconds"),
        ("M04", "FI-S03-M04", "parent_m04_ledger", "parent_m04_gpu_charged_seconds", "parent_m04_gpu_budget_seconds"),
        ("M05", "FI-S03-M05", "parent_m05_ledger", "parent_m05_gpu_charged_seconds", "parent_m05_gpu_budget_seconds")]
    charges = []
    for label, stage, key, used_key, budget_key in specifications:
        path = Path(protocol[key]["path"])
        if not path.is_absolute() or digest(path) != protocol[key]["sha256"]:
            raise RuntimeError("Parent " + label + " ledger snapshot changed")
        ledger = read_json(path)
        charged = spent(ledger)
        if (ledger.get("stage") != stage or not math.isclose(charged, float(protocol[used_key]), rel_tol=0, abs_tol=1e-9)
                or float(ledger["gpu_budget_seconds"]) != float(protocol[budget_key])):
            raise RuntimeError("Parent " + label + " accounting changed")
        if any(item.get("state") == "active" for item in ledger["attempts"]):
            raise RuntimeError("Parent " + label + " still has active work")
        charges.append(Decimal(str(charged)).quantize(Decimal("0.000000001")))
    for key, files in [("prior_r01_evidence", ("ledger", "protocol", "manifest")),
                       ("prior_m02_evidence", ("ledger", "protocol", "manifest", "summary")),
                       ("prior_m03_evidence", ("ledger", "protocol", "manifest", "summary")),
                       ("prior_m04_evidence", ("ledger", "protocol", "manifest", "summary")),
                       ("prior_m05_evidence", ("ledger", "protocol", "manifest", "summary"))]:
        previous = protocol[key]
        source_root = Path(previous["root_linux"])
        for name in files:
            if digest(source_root / (name + ".json")) != previous[name + "_sha256"]:
                raise RuntimeError("Frozen parent evidence changed: " + key + "/" + name)
    prior = protocol["prior_r01_evidence"]
    gate_path = Path(prior["root_linux"]) / "memory-gate.json"
    if digest(gate_path) != prior["original_gate_sha256"] or read_json(gate_path).get("passed") is not False:
        raise RuntimeError("Historical R01 gate must retain its recorded failure")
    charged = sum(charges)
    approved_parent = Decimal(str(protocol["approved_parent_gpu_charged_seconds"]))
    approved_total = Decimal(str(protocol["approved_combined_gpu_budget_seconds"]))
    if charged != approved_parent or (approved_parent != Decimal("4091.642331315") or approved_total != Decimal("5291.642331315")):
        raise RuntimeError("Approved cumulative accounting changed")
    if charged + Decimal(str(protocol["gpu_budget_seconds"])) > approved_total:
        raise RuntimeError("Combined work exceeds the newly approved cumulative allowance")
    return float(charged)

def job_by_id(protocol, name):
    return next(job for job in protocol["jobs"] if job["id"] == name)


def locks(root):
    import fcntl
    handles = []
    try:
        parent_root = Path(read_json(root / "protocol.json")["parent_s03_ledger"]["path"]).parent
        m01_root = Path(read_json(root / "protocol.json")["parent_m01_ledger"]["path"]).parent
        r01_root = Path(read_json(root / "protocol.json")["parent_r01_ledger"]["path"]).parent
        m02_root = Path(read_json(root / "protocol.json")["parent_m02_ledger"]["path"]).parent
        m03_root = Path(read_json(root / "protocol.json")["parent_m03_ledger"]["path"]).parent
        m04_root = Path(read_json(root / "protocol.json")["parent_m04_ledger"]["path"]).parent
        m05_root = Path(read_json(root / "protocol.json")["parent_m05_ledger"]["path"]).parent
        paths = [parent_root / "gpu-stage.lock", m01_root / "gpu-stage.lock", r01_root / "gpu-stage.lock", m02_root / "gpu-stage.lock", m03_root / "gpu-stage.lock", m04_root / "gpu-stage.lock", m05_root / "gpu-stage.lock"]
        old_d01_lock = parent_root.parent / "S03-D01" / "gpu-stage.lock"
        if old_d01_lock.exists():
            paths.append(old_d01_lock)
        paths += [root / "run.lock", root / "gpu-stage.lock"]
        for path in paths:
            handle = path.open("a+")
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handles
    except BaseException:
        for handle in handles:
            handle.close()
        raise


class StopRequested(RuntimeError):
    pass


class Heartbeat:
    def __init__(self, path, launch, timeout):
        self.path, self.launch, self.timeout = Path(path), launch, float(timeout)
        self.seq, self.seen, self.scheme = -1, time.monotonic(), None

    def poll(self, allow_battery=False):
        try:
            value = read_json(self.path)
        except (OSError, ValueError):
            if time.monotonic() - self.seen >= self.timeout:
                raise StopRequested("Windows heartbeat unavailable")
            return
        if value.get("launch_id") != self.launch or type(value.get("seq")) is not int or value["seq"] < self.seq:
            raise StopRequested("Windows heartbeat identity changed")
        if value["seq"] > self.seq:
            self.seq, self.seen = value["seq"], time.monotonic()
        if value.get("stop") or time.monotonic() - self.seen >= self.timeout:
            raise StopRequested("Windows launcher stopped or heartbeat expired")
        if not allow_battery:
            if value.get("ac_online") is not True:
                raise StopRequested("AC power disconnected")
            scheme = value.get("power_scheme")
            if not scheme or (self.scheme is not None and self.scheme != scheme):
                raise StopRequested("Windows power scheme changed")
            self.scheme = scheme


def check_stop(root, launch):
    path = root / "stop.flag"
    if path.exists() and read_json(path).get("launch_id") == launch:
        raise StopRequested("User requested stop")


def directory_size(root):
    total = 0
    for directory, directories, files in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if not (Path(directory) / name).is_symlink()]
        for name in files:
            path = Path(directory) / name
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                pass
    return total


def storage_check(root, protocol):
    used = directory_size(root)
    for entry in protocol.get("storage_baselines", []):
        used += max(0, directory_size(Path(entry["path"])) - entry["bytes"])
    limit = int(protocol["storage_limit_bytes"])
    stop = int(protocol.get("storage_stop_bytes", max(1, limit - 64 * 1024**2)))
    if not 0 < stop <= limit or used >= stop:
        raise StopRequested("S03-M06 storage early-stop line reached: " + str(used))
    return used


def worker_environment(root, protocol, job_id):
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false", MAX_JOBS="1", NVCC_THREADS="1")
    configured = protocol.get("runtime_env", {}).get(job_id, {})
    env.update({str(key): str(value) for key, value in configured.items()})
    temporary = root / "cache" / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "TMP", "TEMP"):
        if key not in configured:
            env[key] = str(temporary)
    return env


def kill_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def verify_result(root, attempt, job, expected_ids):
    result = read_json(attempt / "result.json")
    for key, value in {"stage": STAGE, **job_identity(job), "passed": True, **expected_ids}.items():
        if result.get(key) != value:
            raise RuntimeError("Completed result identity differs: " + key)
    expected_cases = [case.get("id", case.get("case_id")) for case in phase_cases(read_json(root / "protocol.json"), job)]
    cases = result.get("cases")
    if not isinstance(cases, list) or len(cases) != len(expected_cases) or {case.get("case_id") for case in cases} != set(expected_cases):
        raise RuntimeError("Completed result does not contain the expected phase cases")
    for case in cases:
        path = child_path(attempt, case["path"])
        if not path.is_relative_to(attempt / "cases") or digest(path) != case["sha256"]:
            raise RuntimeError("Completed case file changed")
        item = read_json(path)
        for key, value in {"stage": STAGE, **job_identity(job), "passed": True, **expected_ids}.items():
            if item.get(key) != value:
                raise RuntimeError("Completed case identity differs: " + key)
    protocol = read_json(root / "protocol.json")
    expected_cq = set(protocol["concurrent_quality"]["cases"])
    concurrent = result.get("concurrent_quality", {})
    if concurrent.get("inputs_sha256") != protocol["concurrent_quality_inputs_sha256"]:
        raise RuntimeError("Concurrent quality input identity differs")
    bindings = concurrent.get("cases", [])
    if len(bindings) != len(expected_cq) or {b.get("case_id") for b in bindings} != expected_cq:
        raise RuntimeError("Missing concurrent quality scenarios")
    for binding in bindings:
        path = child_path(attempt, binding["path"])
        if not path.is_relative_to(attempt / "concurrent-quality") or digest(path) != binding["sha256"]:
            raise RuntimeError("Concurrent quality binding differs")
        item = read_json(path)
        if item.get("collection_complete") is not True or len(item.get("samples", [])) != 2:
            raise RuntimeError("Concurrent quality scenario collection is incomplete")
    return result


def guardian(request_path):
    request = read_json(request_path)
    root = Path(request["root"])
    attempt = child_path(root, request["attempt_relative"])
    protocol = read_json(root / "protocol.json")
    job = job_by_id(protocol, request["job"])
    heartbeat = Heartbeat(request["heartbeat"], request["launch_id"], protocol.get("heartbeat_timeout_seconds", 20))
    heartbeat.scheme = request["power_scheme"]
    process, timer, reason, result_hash = None, None, "", None
    begin = time.monotonic()
    expired = threading.Event()

    def interrupt(signum, frame):
        raise StopRequested("Per-job or remaining GPU-stage hard deadline reached" if expired.is_set() else "Guardian interrupted")

    def expire():
        expired.set()
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        os.kill(os.getpid(), signal.SIGTERM)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupt)
    try:
        reserved = float(request["reserved_seconds"])
        timer = threading.Timer(max(.001, reserved - min(2, reserved * .1)), expire)
        timer.daemon = True
        timer.start()
        if os.getppid() != request["controller_pid"]:
            raise StopRequested("Controller exited before worker start")
        heartbeat.poll()
        check_stop(root, request["launch_id"])
        if verify_files(root) != request["identities"]:
            raise RuntimeError("Frozen S03-M06 identity changed")
        storage_check(root, protocol)
        command = [protocol["python_linux"], "-B", str(root / protocol.get("worker_script", "workspace_worker.py")),
                   "--root", str(root), "--output-dir", str(attempt), "--protocol", str(root / "protocol.json"), "--job", job["id"]]
        if request.get("resume_source"):
            command += ["--resume-source", str(child_path(root, request["resume_source"]))]
        with (attempt / "worker.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=root, env=worker_environment(root, protocol, job["id"]),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=tuple(request["lock_fds"]))
            write_json(attempt / "worker-process.json", {"pid": process.pid, "job": job["id"], "begin_epoch": time.time()})
            next_storage = time.monotonic() + 30
            while True:
                # A fully saved result is the completion boundary; engine exit may hang.
                if (attempt / "result.json").exists():
                    verify_result(root, attempt, job, request["identities"])
                    result_hash = digest(attempt / "result.json")
                    break
                if process.poll() is not None:
                    raise RuntimeError("Worker exited " + str(process.returncode) + " without a complete result")
                if os.getppid() != request["controller_pid"]:
                    raise StopRequested("Controller exited; saving completed cases")
                heartbeat.poll()
                check_stop(root, request["launch_id"])
                if time.monotonic() >= next_storage:
                    storage_check(root, protocol)
                    next_storage = time.monotonic() + 30
                time.sleep(.2)
    except BaseException as exc:
        reason = type(exc).__name__ + ": " + str(exc)
    finally:
        if timer:
            timer.cancel()
            timer.join(timeout=2)
        kill_group(process)
        receipt = {"stage": STAGE, **job_identity(job), "launch_id": request["launch_id"],
            "attempt_id": attempt.name, "identities": request["identities"], "passed": result_hash is not None and not reason,
            "reason": reason, "result_sha256": result_hash, "result_completed_before_cleanup": result_hash is not None,
            "elapsed_seconds": max(0, time.monotonic() - begin), "end_epoch": time.time()}
        write_json(attempt / "guardian-receipt.json", receipt)
    return 0 if receipt["passed"] else 1


def spent(ledger):
    return sum(float(item.get("charged_seconds", 0)) for item in ledger["attempts"])


def settle(root, ledger, record, receipt):
    expected = {"stage": STAGE, "job": record["job"], "backend": record["backend"], "phase": record["phase"], "workspace_mib": record["workspace_mib"],
                "launch_id": record["launch_id"], "attempt_id": record["attempt_id"], "identities": ledger["identities"]}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Guardian receipt identity changed")
    elapsed = receipt.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError("Invalid guardian elapsed time")
    record.update(state="stopped", charged_seconds=elapsed, reason=receipt.get("reason", ""), end_epoch=receipt["end_epoch"])
    if receipt.get("passed") is True:
        attempt = child_path(root, record["attempt_relative"])
        verify_result(root, attempt, job_by_id(read_json(root / "protocol.json"), record["job"]), ledger["identities"])
        result_hash = digest(attempt / "result.json")
        if result_hash != receipt.get("result_sha256"):
            raise RuntimeError("Result changed after guardian collection")
        ledger["completed"][record["job"]] = {"attempt_id": attempt.name,
            "result_relative": str((attempt / "result.json").relative_to(root)), "sha256": result_hash}
        record.update(state="completed", result_sha256=result_hash)


def verify_saved(root, ledger, protocol, expected):
    if ledger.get("identities") != expected or ledger.get("gpu_budget_seconds") != protocol["gpu_budget_seconds"]:
        raise RuntimeError("Saved S03-M06 configuration changed")
    for job_id, binding in ledger["completed"].items():
        path = child_path(root, binding["result_relative"])
        if digest(path) != binding["sha256"]:
            raise RuntimeError("Saved S03-M06 result changed")
        verify_result(root, path.parent, job_by_id(protocol, job_id), expected)


def analysis_module(root):
    spec = importlib.util.spec_from_file_location("s03_m01_analysis", root / "analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(root, ledger, protocol):
    return analysis_module(root).summarize(root, ledger, protocol)


def case_count(attempt):
    count = 0
    for path in (attempt / "cases").glob("*.json"):
        try:
            if read_json(path).get("passed") is True:
                count += 1
        except (OSError, ValueError):
            pass
    return count


def publish(root, ledger, protocol, state, message="", current=None, attempt=None):
    active = [item for item in ledger["attempts"] if item["state"] == "active"]
    live = sum(max(0, time.monotonic() - item["begin_monotonic"]) for item in active)
    cases = sum(len(phase_cases(protocol, job_by_id(protocol, name))) for name in ledger["completed"])
    for job in protocol["jobs"]:
        if job["id"] not in ledger["completed"]:
            candidates = [item for item in ledger["attempts"] if item["job"] == job["id"]]
            if candidates:
                cases += min(len(phase_cases(protocol, job)), case_count(child_path(root, candidates[-1]["attempt_relative"])))
    progress = {}
    if attempt is not None:
        try:
            progress = read_json(attempt / "progress.json")
        except (OSError, ValueError):
            pass
    write_json(root / "status.json", {"stage": STAGE, "state": state, "message": message, "current_job": current,
        "completed_jobs": len(ledger["completed"]), "total_jobs": len(protocol["jobs"]), "completed_cases": cases, "total_cases": sum(len(phase_cases(protocol, job)) for job in protocol["jobs"]),
        "phase_progress": {phase: {"completed_jobs": sum(job["id"] in ledger["completed"] for job in protocol["jobs"] if job["phase"] == phase),
                                  "total_jobs": sum(job["phase"] == phase for job in protocol["jobs"])}
                           for phase in ("memory", "capacity")},
        "current_phase": job_by_id(protocol, current)["phase"] if current else None,
        "gpu_charged_seconds": spent(ledger), "live_elapsed_seconds": live,
        "gpu_observed_seconds": spent(ledger) + live, "gpu_budget_seconds": protocol["gpu_budget_seconds"],
        "parent_gpu_charged_seconds": protocol["parent_gpu_charged_seconds"],
        "parent_m01_gpu_charged_seconds": protocol["parent_m01_gpu_charged_seconds"],
        "parent_r01_gpu_charged_seconds": protocol["parent_r01_gpu_charged_seconds"],
        "parent_m02_gpu_charged_seconds": protocol["parent_m02_gpu_charged_seconds"],
        "parent_m03_gpu_charged_seconds": protocol["parent_m03_gpu_charged_seconds"],
        "parent_m04_gpu_charged_seconds": protocol["parent_m04_gpu_charged_seconds"],
        "parent_m05_gpu_charged_seconds": protocol["parent_m05_gpu_charged_seconds"],
        "research_kind": "1_to_2",
        "m01_combined_gpu_observed_seconds": float(protocol["parent_m01_gpu_charged_seconds"]) + float(protocol["parent_r01_gpu_charged_seconds"]) + float(protocol["parent_m02_gpu_charged_seconds"]) + float(protocol["parent_m03_gpu_charged_seconds"]) + float(protocol["parent_m04_gpu_charged_seconds"]) + float(protocol["parent_m05_gpu_charged_seconds"]) + spent(ledger) + live,
        "m01_combined_gpu_budget_seconds": protocol["approved_m01_chain_gpu_budget_seconds"],
        "gpu_remaining_seconds": max(0, float(protocol["gpu_budget_seconds"]) - spent(ledger) - live),
        "combined_gpu_observed_seconds": float(protocol["parent_gpu_charged_seconds"]) + float(protocol["parent_m01_gpu_charged_seconds"]) + float(protocol["parent_r01_gpu_charged_seconds"]) + float(protocol["parent_m02_gpu_charged_seconds"]) + float(protocol["parent_m03_gpu_charged_seconds"]) + float(protocol["parent_m04_gpu_charged_seconds"]) + float(protocol["parent_m05_gpu_charged_seconds"]) + spent(ledger) + live,
        "combined_gpu_budget_seconds": protocol["approved_combined_gpu_budget_seconds"],
        "progress": progress, "attempt_dir": str(attempt) if attempt else None, "last_update_epoch": time.time()})


def run(root, launch, run_dir, preflight_only=False):
    root = Path(root).resolve()
    launch = str(uuid.UUID(launch))
    run_dir = Path(run_dir).resolve()
    if run_dir != root / "runs" / launch or not run_dir.is_dir():
        raise RuntimeError("Unexpected launch directory")
    handles = locks(root)
    ledger = protocol = None
    guard = None
    record = attempt = current = None
    try:
        expected = verify_files(root)
        protocol = read_json(root / "protocol.json")
        validate_protocol(protocol)
        verify_parent(protocol)
        heartbeat = Heartbeat(run_dir / "heartbeat.json", launch, protocol.get("heartbeat_timeout_seconds", 20))
        heartbeat.poll(allow_battery=preflight_only)
        if preflight_only:
            write_json(run_dir / "startup-check.json", {"passed": True, "worker_started": False, "gpu_started": False, "ledger_mutated": False})
            return 0
        ledger_path = root / "ledger.json"
        ledger = read_json(ledger_path) if ledger_path.exists() else {"schema": 1, "stage": STAGE,
            **expected, "identities": expected, "gpu_budget_seconds": protocol["gpu_budget_seconds"], "attempts": [], "completed": {}}
        verify_saved(root, ledger, protocol, expected)
        for old in ledger["attempts"]:
            if old["state"] == "active":
                receipt_path = child_path(root, old["attempt_relative"]) / "guardian-receipt.json"
                if receipt_path.exists():
                    settle(root, ledger, old, read_json(receipt_path))
                else:
                    old.update(state="interrupted", charged_seconds=old["reserved_seconds"], reason="Unobserved interruption charged full reservation")
        write_json(ledger_path, ledger)
        for job in protocol["jobs"]:
            current = job["id"]
            if current in ledger["completed"]:
                continue
            heartbeat.poll()
            check_stop(root, launch)
            remaining = float(protocol["gpu_budget_seconds"]) - spent(ledger)
            if remaining <= .5:
                summarize(root, ledger, protocol)
                publish(root, ledger, protocol, "budget_stopped", "Approved remaining GPU-stage allowance exhausted; saved results preserved")
                return 0
            reservation = min(remaining, float(protocol["per_job_hard_limit_seconds"]))
            previous = next((item for item in reversed(ledger["attempts"]) if item["job"] == current
                             and case_count(child_path(root, item["attempt_relative"])) > 0), None)
            attempt = root / "attempts" / (current.lower() + "-" + str(uuid.uuid4()))
            attempt.mkdir(parents=True)
            record = {**job_identity(job), "attempt_id": attempt.name,
                "attempt_relative": str(attempt.relative_to(root)), "launch_id": launch, "state": "active",
                "reserved_seconds": reservation, "charged_seconds": 0, "begin_epoch": time.time(), "begin_monotonic": time.monotonic()}
            ledger["attempts"].append(record)
            write_json(ledger_path, ledger)
            request = {"root": str(root), "attempt_relative": record["attempt_relative"], "job": current,
                "controller_pid": os.getpid(), "launch_id": launch, "heartbeat": str(run_dir / "heartbeat.json"),
                "power_scheme": heartbeat.scheme, "reserved_seconds": reservation, "identities": expected,
                "resume_source": previous["attempt_relative"] if previous else None,
                "lock_fds": [handle.fileno() for handle in handles]}
            write_json(attempt / "guardian-request.json", request)
            publish(root, ledger, protocol, "running", "1-to-2 pressure-state correctness: startup sizing, six quality tasks and two four-request lengths", current, attempt)
            with (attempt / "guardian.log").open("wb") as log:
                guard = subprocess.Popen([protocol["python_linux"], "-B", str(root / "runner.py"), "--guardian-request", str(attempt / "guardian-request.json")],
                    cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=tuple(handle.fileno() for handle in handles))
                while guard.poll() is None:
                    time.sleep(.5)
                    publish(root, ledger, protocol, "running", "1-to-2 pressure-state correctness: startup sizing, six quality tasks and two four-request lengths", current, attempt)
                guard.wait()
            guard = None
            receipt = read_json(attempt / "guardian-receipt.json")
            settle(root, ledger, record, receipt)
            write_json(ledger_path, ledger)
            summarize(root, ledger, protocol)
            if receipt.get("passed") is not True:
                raise StopRequested(receipt.get("reason") or "Worker did not save a full result")
            publish(root, ledger, protocol, "ready", "Job saved", current, attempt)
        summarize(root, ledger, protocol)
        publish(root, ledger, protocol, "complete", "Correctness and performance collection completed; four jobs saved")
        return 0
    except BaseException as exc:
        error = type(exc).__name__ + ": " + str(exc)
        if guard is not None and guard.poll() is None:
            guard.terminate()
            try:
                guard.wait(timeout=8)
            except subprocess.TimeoutExpired:
                pass
        if ledger is not None and protocol is not None:
            if record is not None and record["state"] == "active":
                receipt_path = attempt / "guardian-receipt.json"
                if receipt_path.exists():
                    try:
                        settle(root, ledger, record, read_json(receipt_path))
                    except Exception as settlement:
                        record["settlement_error"] = str(settlement)
            write_json(root / "ledger.json", ledger)
            publish(root, ledger, protocol, "stopped", error, current, attempt)
            try:
                summarize(root, ledger, protocol)
            except Exception:
                traceback.print_exc()
        write_json(run_dir / "error.json", {"error": error, "traceback": traceback.format_exc()})
        print(error, flush=True)
        return 1
    finally:
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-id")
    parser.add_argument("--run-dir")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--guardian-request")
    args = parser.parse_args()
    if args.guardian_request:
        return guardian(args.guardian_request)
    if not args.launch_id or not args.run_dir:
        parser.error("launch-id and run-dir are required")
    try:
        return run(ROOT, args.launch_id, Path(args.run_dir), args.preflight_only)
    except BlockingIOError:
        print("BUSY: an earlier S03 / S03-M06 process is still running or stopping", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
