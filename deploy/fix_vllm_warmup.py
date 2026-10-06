"""Guard an SM100-only import before execution on the measured vLLM release.

vLLM 0.31.0 imports unrelated MiniMax Triton kernels before checking its GPU
guard. That import fails in the measured Python 3.10 / RTX 4090 environment.
This moves the existing guard, preserving the Qwen inference implementation.
Reference: https://github.com/vllm-project/vllm/issues/49920
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sysconfig

MARKER = "    # JEV-9B deployment: check the SM100 guard before importing MiniMax kernels.\n"
IMPORT = "from vllm.models.minimax_m3.nvidia.model import MiniMaxM3SparseAttention\n"
GUARD = (
    "    if not (\n"
    "        current_platform.is_cuda() and current_platform.is_device_capability_family(100)\n"
    "    ):\n"
    "        return\n"
)
SIGNATURE = 'def minimax_m3_msa_warmup(worker: "Worker") -> None:\n'


def patch_source(original: str) -> str:
    expected = SIGNATURE + MARKER + GUARD + "    " + IMPORT
    if MARKER in original:
        if expected not in original:
            raise RuntimeError("Existing warmup patch differs from the supported version")
        return original
    if any(original.count(fragment) != 1 for fragment in (IMPORT, GUARD, SIGNATURE)):
        raise RuntimeError("Unexpected vLLM source; refusing an unverified runtime patch")
    updated = original.replace(IMPORT, "").replace(GUARD, "")
    return updated.replace(SIGNATURE, expected)


def main() -> None:
    version = importlib.metadata.version("vllm")
    if version != "0.31.0":
        raise RuntimeError(f"Expected vLLM 0.31.0, found {version}; refusing to patch")
    path = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/warmup/minimax_m3_msa_warmup.py"
    original = path.read_text()
    updated = patch_source(original)
    compile(updated, str(path), "exec")
    backup = path.with_suffix(".py.jev9b-original")
    if updated != original:
        if backup.exists() and backup.read_text() != original:
            raise RuntimeError(f"Existing backup differs; inspect {backup}")
        backup.write_text(original)
        temporary = path.with_suffix(".py.jev9b-pending")
        temporary.write_text(updated)
        temporary.chmod(path.stat().st_mode)
        temporary.replace(path)
    runtime = Path(os.environ.get("JEV_RUNTIME_DIR", Path(__file__).resolve().parent / "runtime"))
    runtime.mkdir(parents=True, exist_ok=True)
    report = {
        "file": str(path), "vllm": version,
        "before_sha256": hashlib.sha256((backup.read_text() if backup.exists() else original).encode()).hexdigest(),
        "after_sha256": hashlib.sha256(updated.encode()).hexdigest(),
        "change": "Move existing SM100 hardware guard before MiniMax-specific import",
        "upstream_issue": "https://github.com/vllm-project/vllm/issues/49920",
    }
    (runtime / "runtime-patch.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Warmup compatibility fix ready; audit record:", runtime / "runtime-patch.json")


if __name__ == "__main__":
    main()
