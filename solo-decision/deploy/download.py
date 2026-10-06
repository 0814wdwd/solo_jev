"""Fetch a pinned text backbone and decision adapter; resume existing downloads."""
from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "deployment.json").read_text())
    destination = Path(os.environ.get("JEV_MODEL_DIR", root / "models/JEV-9B"))
    print(f"Fetching {manifest['model']} at {manifest['revision']}", flush=True)
    snapshot_download(
        repo_id=manifest["model"], revision=manifest["revision"], local_dir=destination,
        allow_patterns=[
            "model-*.safetensors", "model.safetensors.index.json", "config.json",
            "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
            "calibration.json", "adapter_vllm/*", "README.md",
        ],
        max_workers=4,
    )
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    required = set(index["weight_map"].values()) | {
        "config.json", "tokenizer.json", "tokenizer_config.json", "calibration.json",
        "adapter_vllm/adapter_model.safetensors", "adapter_vllm/adapter_config.json",
        "adapter_vllm/decision_head.json",
    }
    missing = sorted(name for name in required if not (destination / name).is_file())
    if missing:
        raise RuntimeError(f"Download incomplete: {missing}")
    (destination / "solo-model-revision.json").write_text(json.dumps({
        "model": manifest["model"], "revision": manifest["revision"],
        "required_files": sorted(required),
    }, indent=2) + "\n")
    print(f"Model ready: {destination}", flush=True)


if __name__ == "__main__":
    main()
