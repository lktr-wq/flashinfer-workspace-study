"""S03-M04 startup workspace policy; standard library only.

The range bound below is a Python host-side port of the allocator arithmetic
in FlashInfer PR #5236, head fdf3c5142b27dc34b8c2cb1aa8fe8910add79486,
PrefillPlanWorkspaceSizeUpperBound (Apache-2.0, FlashInfer contributors).
The original excerpt, source URLs, attribution and local identity pins are
preserved in policy-evidence.json. This file adds serving admission and a
startup sizing policy. It does not add an official FlashInfer API or a kernel.

Exact workspace_size() calls made by startup_worker are boundary witnesses.
Coverage of intermediate shapes follows from the pinned scheduler bound.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

MIB = 1024 * 1024
POLICY_VERSION = "s03-m04-pr5236-host-bound-v1"
UPSTREAM_PR = "https://github.com/flashinfer-ai/flashinfer/pull/5236"
UPSTREAM_SHA = "fdf3c5142b27dc34b8c2cb1aa8fe8910add79486"
UPPER_BOUND_SOURCE_SHA256 = "2940aac0d4d4aa1eb3d4ed1ac1b0a6240f2b27f845b7b6254b32d758a6293d27"
VENDOR_SOURCE_HASHES = {
    "vendor/pr5236/scheduler.cuh": UPPER_BOUND_SOURCE_SHA256,
    "vendor/pr5236/utils.cuh": "d21b554c57bdd3eec6ffeb130f551611d0b60210c6c84c69a1575d9e7677f7db",
}
CTA_TILE_Q_CANDIDATES = (16, 32, 64, 128)
MIN_SELECTED_BYTES = 32 * MIB
DEFAULT_NATIVE_BYTES = 394 * MIB
SOURCE_HASHES = {
    "data/include/flashinfer/attention/scheduler.cuh": "09c6e89861f48f3b79fee029b6e44c8e16f41d73053fb0154f0e636498d0655f",
    "data/include/flashinfer/utils.cuh": "da42b5f100ccb02a4c05c6f82157ce604dcc535b5d7c1b38a68e42865a7e650e",
    "data/include/flashinfer/allocator.h": "3d80e3d3d29b1ae41218eb1167ca0059f923a9ccd69de79905d7f37df7cb2109",
    "prefill.py": "c315341a5472fcb5cff58de2eb9cda501a45903d769ea9cc79da59852a137632",
    "decode.py": "f67c5704d8ee1c7130b97adde8df8079faa42a11f37a08afa33ab26da71c1fe7",
    "data/csrc/batch_prefill.cu": "3b669453ca1c8740d3043287f5dce726e79f81e05fcc28955e6002d52950a365",
    "data/csrc/batch_prefill_jit_binding.cu": "dad0d81c55b92f71a634cb8240dcb0d150aa6ffd391ef3ef5d87b6207eeca4fc",
}


def default_source_root() -> Path:
    if os.name == "nt":
        return Path("F:/FlashInferResearch/S02-D02/overlay/fixed/flashinfer")
    return Path("/mnt/f/FlashInferResearch/S02-D02/overlay/fixed/flashinfer")


def validate_sources(source_root=None) -> dict:
    """Hash the small, frozen source set without importing Torch or FlashInfer."""
    root = Path(source_root) if source_root is not None else default_source_root()
    files = []
    for relative, expected in SOURCE_HASHES.items():
        path = root / relative
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            error = None
        except OSError as exc:
            actual, error = None, str(exc)
        files.append({"relative_path": relative, "path": str(path),
                      "expected_sha256": expected, "actual_sha256": actual,
                      "matched": actual == expected, "error": error})
    vendor_files = []
    for relative, expected in VENDOR_SOURCE_HASHES.items():
        path = Path(__file__).resolve().parent / relative
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            error = None
        except OSError as exc:
            actual, error = None, str(exc)
        vendor_files.append({"relative_path": relative, "path": str(path),
                             "expected_sha256": expected, "actual_sha256": actual,
                             "matched": actual == expected, "error": error})
    return {"passed": all(row["matched"] for row in files + vendor_files),
            "root": str(root), "files": files, "vendor_files": vendor_files,
            "upper_bound_source_sha256": UPPER_BOUND_SOURCE_SHA256}


def _ceil_div(value: int, denominator: int) -> int:
    return (value + denominator - 1) // denominator


def _allocated_bytes(allocations) -> int:
    """Mirror AlignedAllocator's count-only offset and alignment order."""
    cursor = 0
    for size, alignment in allocations:
        cursor = _ceil_div(cursor, alignment) * alignment + size
    return cursor


def compute_workspace_bound(*, max_num_seqs: int, max_num_batched_tokens: int,
                            max_model_len: int, num_qo_heads: int,
                            num_kv_heads: int, head_dim: int, page_size: int,
                            num_sms: int, index_dtype_bytes: int = 4,
                            enable_cuda_graph: bool = True) -> dict:
    """Port PR #5236's default-split FA2 range bound for all admitted rows.

    The caller admits BF16 FA2 tensor-core decode/prefill with automatic
    splitting, zero colocated CTAs and no custom/cascade/DCP route. Floating
    partial output and softmax statistics are FP32 even for BF16 output.
    All distributions of query rows over requests are bounded, including
    the uniform_q_len planner path. Request lengths may range up to the
    configured model limit; the default split search bounds split CTAs
    independently of individual KV lengths.
    """
    values = (max_num_seqs, max_num_batched_tokens, max_model_len,
              num_qo_heads, num_kv_heads, head_dim, page_size, num_sms)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in values):
        raise ValueError("All scheduler limits and dimensions must be positive integers")
    if num_qo_heads % num_kv_heads:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    if index_dtype_bytes != 4:
        raise ValueError("This source-locked policy admits int32 planner indices")
    if max_num_seqs > max_num_batched_tokens:
        raise ValueError("max_num_seqs must not exceed max_num_batched_tokens")
    group_size = num_qo_heads // num_kv_heads
    packed_rows = max_num_batched_tokens * group_size
    if packed_rows >= 2**31 or max_num_seqs * _ceil_div(max_model_len, page_size) >= 2**31:
        raise ValueError("Configured scheduler indices exceed the admitted int32 range")
    split_ceiling = 2 * num_sms // num_kv_heads
    rows = []
    for tile_q in CTA_TILE_Q_CANDIDATES:
        max_tiles = _ceil_div(packed_rows, tile_q) + max_num_seqs
        padded_float = max(split_ceiling, max_tiles) if enable_cuda_graph else split_ceiling
        padded_int = max(max_tiles, padded_float)
        integer_allocations = [
            (index_dtype_bytes * padded_int, 16),
            (index_dtype_bytes * padded_int, 16),
            (index_dtype_bytes * padded_int, 16),
            (index_dtype_bytes * (max_num_seqs + 1), 16),
            (index_dtype_bytes, 1),
        ]
        if enable_cuda_graph:
            integer_allocations.append((4, 16))
        integer_allocations.extend([
            (index_dtype_bytes * (max_num_batched_tokens + 1), 16),
            (padded_int, 16),
        ])
        float_bytes = _allocated_bytes([
            (num_qo_heads * padded_float * tile_q * head_dim * 4, 16),
            (num_qo_heads * padded_float * tile_q * 4, 16),
        ])
        rows.append({"cta_tile_q": tile_q, "max_num_tiles_q": max_tiles,
                     "max_batch_size_if_split": split_ceiling,
                     "padded_float_batch_size": padded_float,
                     "padded_int_batch_size": padded_int,
                     "float_bound_bytes": float_bytes,
                     "int_bound_bytes": _allocated_bytes(integer_allocations)})
    return {
        "float_bound_bytes": max(row["float_bound_bytes"] for row in rows),
        "int_bound_bytes": max(row["int_bound_bytes"] for row in rows),
        "bound_rows": rows,
        "limits": {"max_num_seqs": max_num_seqs,
                   "max_num_batched_tokens": max_num_batched_tokens,
                   "max_model_len": max_model_len,
                   "max_num_pages_per_request": _ceil_div(max_model_len, page_size),
                   "page_size": page_size, "num_qo_heads": num_qo_heads,
                   "num_kv_heads": num_kv_heads, "head_dim": head_dim,
                   "num_sms": num_sms, "enable_cuda_graph": bool(enable_cuda_graph)},
        "source_identity": {"policy_version": POLICY_VERSION,
                            "upstream_pr": UPSTREAM_PR, "upstream_head_sha": UPSTREAM_SHA,
                            "upper_bound_source_sha256": UPPER_BOUND_SOURCE_SHA256},
        "coverage": "Pinned FA2 scheduler-derived range bound over every admitted query/KV distribution",
        "proof": [
            "The tile selector returns a member of {16,32,64,128}; all members are bounded.",
            "For B requests and R rows, the sum of rounded-up packed query tiles is <= ceil(R*G/t)+B.",
            "Default split search bounds split CTAs by floor(2*SM/KV_heads); CUDA graphs additionally pad to the query tile bound.",
            "FP32 partial values and statistics follow the pinned allocator's 16-byte aligned order.",
            "uniform_q_len restricts row distributions and is covered by the same all-distributions tile bound.",
        ],
    }


def _query_cases(limits: dict) -> list:
    """Concrete witnesses for the existing API, separate from the range proof."""
    max_batch = limits["max_num_seqs"]
    max_rows = limits["max_num_batched_tokens"]
    max_len = limits["max_model_len"]
    cases = []
    for batch in sorted({1, max_batch}):
        for total in sorted({batch, min(max_rows, batch * max_len)}):
            query_lens = [1] * batch
            remaining = total - batch
            for i in range(batch):
                extra = min(remaining, max_len - 1)
                query_lens[i] += extra
                remaining -= extra
            cases.append({"name": f"b{batch}-q{total}-skew-maxkv",
                          "query_lens": query_lens,
                          "kv_lens": [max_len] * batch,
                          "use_cuda_graph": False})
        if limits["enable_cuda_graph"]:
            cases.append({"name": f"b{batch}-decode-graph-maxkv",
                          "query_lens": [1] * batch,
                          "kv_lens": [max_len] * batch,
                          "use_cuda_graph": True})
    return cases


def choose_workspace(config: dict, *, source_root=None, verify_sources: bool = True) -> dict:
    """Choose a startup recommendation; caller handles explicit override/fallback.

    No environment variables are read or changed. The hook must enforce its
    separately approved service limits, match the actual loaded route, and
    retain the native getter when supported=False. Explicit user overrides
    are owned by the hook. This policy never changes wrapper integer buffers.
    """
    result = {"supported": False, "reason": None, "policy_version": POLICY_VERSION,
              "source_verification": None, "bound_rows": []}
    try:
        capability = config.get("device_capability")
        if isinstance(capability, (list, tuple)):
            capability = tuple(capability)
        if capability not in ((8, 9), 89, "sm_89", "SM89"):
            raise ValueError("Admitted device capability is SM89")
        if str(config.get("backend", "FLASHINFER")).upper() not in ("FLASHINFER", "FA2"):
            raise ValueError("Admitted attention backend is FlashInfer FA2")
        if str(config.get("resolved_backend", "fa2")).lower() != "fa2":
            raise ValueError("Resolved FlashInfer implementation must be fa2")
        for field in ("q_dtype", "kv_dtype", "o_dtype"):
            if str(config.get(field, "bfloat16")).lower().replace("torch.", "") not in ("bfloat16", "bf16"):
                raise ValueError(f"{field} must be BF16")
        if (int(config["num_qo_heads"]), int(config["num_kv_heads"]), int(config["head_dim"])) != (12, 2, 128):
            raise ValueError("Current model admission is Qwen 12Q/2KV/head128")
        if int(config["page_size"]) != 16:
            raise ValueError("Current paged KV layout admits page_size=16")
        for field in ("tensor_parallel_size", "decode_context_parallel_size"):
            if int(config.get(field, 1)) != 1:
                raise ValueError(f"{field} must be 1")
        for field in ("has_sinks", "use_cascade", "has_custom_mask", "disable_split_kv"):
            if bool(config.get(field, False)):
                raise ValueError(f"{field} is outside the ordinary FA2 admission")
        if config.get("fixed_split_size", -1) not in (-1, None):
            raise ValueError("Automatic split search is required")
        if int(config.get("num_colocated_ctas", 0)) != 0:
            raise ValueError("Colocated CTA reservations are outside admission")
        if config.get("sliding_window", -1) not in (-1, None):
            raise ValueError("Current admission is full-history attention")
        if str(config.get("pos_encoding_mode", "NONE")).upper() != "NONE":
            raise ValueError("Current FlashInfer route has external RoPE and pos_encoding_mode=NONE")
        if float(config.get("logits_soft_cap", 0) or 0) != 0:
            raise ValueError("Current route has zero logits_soft_cap")
        if not bool(config.get("causal", True)):
            raise ValueError("Current admission is causal attention")
        if verify_sources:
            result["source_verification"] = validate_sources(source_root)
            if not result["source_verification"]["passed"]:
                raise ValueError("Frozen FlashInfer source identity mismatch")
        else:
            result["source_verification"] = {"passed": None, "skipped_for_cpu_arithmetic_only": True,
                                             "upper_bound_source_sha256": UPPER_BOUND_SOURCE_SHA256}
        bound = compute_workspace_bound(
            max_num_seqs=int(config["max_num_seqs"]),
            max_num_batched_tokens=int(config["max_num_batched_tokens"]),
            max_model_len=int(config["max_model_len"]),
            num_qo_heads=int(config["num_qo_heads"]), num_kv_heads=int(config["num_kv_heads"]),
            head_dim=int(config["head_dim"]), page_size=int(config["page_size"]),
            num_sms=int(config["num_sms"]),
            index_dtype_bytes=int(config.get("index_dtype_bytes", 4)),
            enable_cuda_graph=bool(config.get("enable_cuda_graph", True)))
        result.update(bound)
        # Native vLLM's prefill coarse guard is retained as an extra floor.
        guard = int(config["max_num_batched_tokens"]) * int(config["num_qo_heads"]) * int(config["head_dim"]) * 16
        raw = bound["float_bound_bytes"]
        native_bytes = max(int(config.get("native_default_bytes", DEFAULT_NATIVE_BYTES)), guard)
        recommended = max(MIN_SELECTED_BYTES, _ceil_div(raw * 9, 8), guard)
        recommended = _ceil_div(recommended, MIB) * MIB
        if raw > native_bytes:
            raise ValueError("Conservative range bound exceeds the native allocation; retain native path for separate admission")
        chosen = min(recommended, native_bytes)
        result.update({"supported": True, "reason": "source_locked_default_split_range",
                       "selected_bytes": chosen, "selected_mib": chosen / MIB,
                       "raw_float_bound_mib": raw / MIB,
                       "native_default_bytes": native_bytes,
                       "native_prefill_guard_bytes": guard,
                       "margin_numerator": 9, "margin_denominator": 8,
                       "minimum_selected_bytes": MIN_SELECTED_BYTES,
                       "selection": "shrink" if chosen < native_bytes else "keep_native",
                       "query_cases": _query_cases(bound["limits"])})
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reason"] = str(exc)
    return result


def _self_check() -> dict:
    base = dict(backend="FLASHINFER", device_capability=[8, 9], num_sms=24,
                num_qo_heads=12, num_kv_heads=2, head_dim=128,
                max_num_seqs=4, max_num_batched_tokens=512, max_model_len=8320,
                page_size=16, q_dtype="bfloat16", kv_dtype="bfloat16", o_dtype="bfloat16")
    decisions = []
    for tokens in (512, 2048, 8192):
        decision = choose_workspace({**base, "max_num_batched_tokens": tokens})
        assert decision["supported"], decision["reason"]
        assert decision["selected_bytes"] >= decision["float_bound_bytes"]
        assert decision["selected_bytes"] <= decision["native_default_bytes"]
        decisions.append({"max_num_batched_tokens": tokens,
                          "float_bound_bytes": decision["float_bound_bytes"],
                          "int_bound_bytes": decision["int_bound_bytes"],
                          "selected_mib": decision["selected_mib"]})
    assert decisions[0]["float_bound_bytes"] == 22_192_128
    assert decisions[0]["int_bound_bytes"] == 4660
    assert decisions[0]["selected_mib"] == 32
    assert len({row["selected_mib"] for row in decisions}) == 3
    unsupported = choose_workspace({**base, "device_capability": [9, 0]}, verify_sources=False)
    assert not unsupported["supported"]
    return {"passed": True, "gpu_used": False, "torch_imported": "torch" in sys.modules,
            "source_verification": validate_sources(), "decisions": decisions,
            "unsupported_device_reason": unsupported["reason"]}


if __name__ == "__main__":
    print(json.dumps(_self_check(), ensure_ascii=False, indent=2))
