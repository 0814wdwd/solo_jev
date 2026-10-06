"""Linux process lifecycle for this checkout's vLLM server; stdlib only."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
RUNTIME = Path(os.environ.get("JEV_RUNTIME_DIR", ROOT / "runtime")).resolve()
RECORD = RUNTIME / "server.json"
LOG = RUNTIME / "server.log"
VENV = Path(os.environ.get("JEV_VENV", ROOT / ".venv")).resolve()
MODEL = Path(os.environ.get("JEV_MODEL_DIR", ROOT / "models/JEV-9B")).resolve()
HOST = os.environ.get("JEV_HOST", "127.0.0.1")
PORT = int(os.environ.get("JEV_PORT", "8000"))
CHECK_HOST = "127.0.0.1" if HOST in ("0.0.0.0", "::") else HOST
URL = f"http://{CHECK_HOST}:{PORT}"
CHILDREN: dict[int, subprocess.Popen] = {}


def process_identity(pid: int) -> dict | None:
    try:
        # The process name can contain spaces or parentheses.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "pgid": int(fields[2]), "start_ticks": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def read_record() -> dict | None:
    try:
        record = json.loads(RECORD.read_text())
        return record if isinstance(record, dict) else None
    except (OSError, ValueError):
        return None


def owns(record: dict | None) -> bool:
    if not record or record.get("checkout") != str(ROOT):
        return False
    try:
        pid = int(record["pid"])
        identity = process_identity(pid)
        return bool(identity and pid == identity["pgid"] and all(
            record.get(key) == value for key, value in identity.items()
        ))
    except (KeyError, ValueError, TypeError):
        return False


def healthy(url: str = URL) -> bool:
    try:
        # Proxy settings are useful for downloads, never for loopback health.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url + "/v1/models", timeout=2) as response:
            models = json.load(response)
        return any(item.get("id") == "jev-decision" for item in models.get("data", []))
    except (OSError, ValueError, AttributeError):
        return False


@contextlib.contextmanager
def lock():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "control.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def terminate(record: dict, timeout: float = 30) -> None:
    if not owns(record):
        return
    try:
        os.killpg(record["pid"], signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while owns(record) and time.monotonic() < deadline:
        time.sleep(0.25)
    if owns(record):
        try:
            os.killpg(record["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
    child = CHILDREN.pop(record["pid"], None)
    if child is not None:
        child.wait(timeout=10)


def start() -> int:
    with lock():
        record = read_record()
        if owns(record):
            if healthy(record["url"]):
                print(f"Already running: {record['url']} (PID {record['pid']})")
                return 0
            raise RuntimeError("Owned server is running but not ready; inspect deploy/runtime/server.log")
        if healthy():
            raise RuntimeError(f"An unowned Jev server already responds at {URL}; use it directly or choose JEV_PORT")
        if not (VENV / "bin/vllm").is_file() or not (MODEL / "adapter_vllm/adapter_model.safetensors").is_file():
            raise RuntimeError("Missing runtime/model. Run bash deploy/jev9b.sh setup first")
        environment = {**os.environ, "JEV_VENV": str(VENV), "JEV_MODEL_DIR": str(MODEL),
                       "JEV_RUNTIME_DIR": str(RUNTIME)}
        with LOG.open("ab", buffering=0) as output:
            child = subprocess.Popen(["bash", str(ROOT / "serve.sh")], cwd=ROOT,
                                     env=environment,
                                     stdout=output, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        CHILDREN[child.pid] = child
        identity = process_identity(child.pid)
        if identity is None:
            raise RuntimeError(f"Server exited during startup; inspect {LOG}")
        record = {**identity, "checkout": str(ROOT), "url": URL, "model_dir": str(MODEL)}
        temporary = RECORD.with_suffix(".json.pending")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(RECORD)
        print(f"Starting PID {child.pid}; log: {LOG}", flush=True)
        try:
            deadline = time.monotonic() + float(os.environ.get("JEV_START_TIMEOUT", "600"))
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(f"Server exited with code {child.returncode}; inspect {LOG}")
                if healthy():
                    print(f"Ready: {URL}; model=jev-decision", flush=True)
                    return 0
                time.sleep(1)
            raise RuntimeError(f"Server readiness timed out; inspect {LOG}")
        except BaseException:
            terminate(record)
            child.wait(timeout=10)
            CHILDREN.pop(child.pid, None)
            RECORD.unlink(missing_ok=True)
            raise


def stop() -> int:
    with lock():
        record = read_record()
        if not owns(record):
            print("No running server owned by this checkout")
            return 0
        terminate(record)
        RECORD.unlink(missing_ok=True)
        print(f"Stopped owned server PID {record['pid']}")
    return 0


def status() -> int:
    record = read_record()
    owned = owns(record)
    url = record["url"] if owned else URL
    ready = healthy(url)
    print(json.dumps({"owned_process_running": owned, "decision_adapter_ready": ready,
                      "url": url, "pid": record["pid"] if owned else None,
                      "log": str(LOG)}, indent=2))
    return 0 if owned and ready else 1


def doctor() -> int:
    print(json.dumps({"platform": sys.platform, "bootstrap_python": sys.version.split()[0],
                      "venv": str(VENV), "model_dir": str(MODEL),
                      "vllm_command_present": (VENV / "bin/vllm").is_file(),
                      "decision_adapter_present": (MODEL / "adapter_vllm/adapter_model.safetensors").is_file()}, indent=2))
    try:
        subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv"],
                       check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"GPU inspection unavailable: {exc}", file=sys.stderr)
    return status()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["start", "stop", "status", "doctor"])
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("JEV GPU lifecycle management currently requires Linux")
    try:
        return {"start": start, "stop": stop, "status": status, "doctor": doctor}[args.command]()
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
