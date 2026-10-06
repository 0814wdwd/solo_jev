"""Offline lifecycle checks: fake model server, real local processes, no GPU."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeploymentTests(unittest.TestCase):
    def test_guard_patch_preserves_body_and_is_idempotent(self):
        module = load("fix_vllm_warmup")
        source = module.IMPORT + module.SIGNATURE + module.GUARD + "    worker.run()\n"
        changed = module.patch_source(source)
        compile(changed, "fixture.py", "exec")
        self.assertLess(changed.index(module.GUARD), changed.index(module.IMPORT))
        self.assertIn("    worker.run()", changed)
        self.assertEqual(module.patch_source(changed), changed)
        with self.assertRaisesRegex(RuntimeError, "Unexpected"):
            module.patch_source("unrecognized = True\n")

    def test_wrong_version_is_not_patched(self):
        module = load("fix_vllm_warmup")
        with patch.object(module.importlib.metadata, "version", return_value="0.30.0"):
            with self.assertRaisesRegex(RuntimeError, "refusing to patch"):
                module.main()

    @unittest.skipUnless(sys.platform == "linux", "Linux /proc lifecycle")
    def test_start_health_reuse_stop_and_stale_pid_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv = root / "venv"
            model = root / "model"
            (venv / "bin").mkdir(parents=True)
            (model / "adapter_vllm").mkdir(parents=True)
            (model / "adapter_vllm/adapter_model.safetensors").touch()
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            executable = venv / "bin/vllm"
            executable.write_text(f"#!{sys.executable}\n" + '''
import http.server, json, os, sys
from pathlib import Path
Path(os.environ["FAKE_ARGS"]).write_text(json.dumps(sys.argv[1:]))
port = int(sys.argv[sys.argv.index("--port") + 1])
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        payload = json.dumps({"data": [{"id": "jev-decision"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
''')
            executable.chmod(0o755)
            environment = {
                "JEV_VENV": str(venv), "JEV_MODEL_DIR": str(model),
                "JEV_RUNTIME_DIR": str(root / "runtime"), "JEV_PORT": str(port),
                "JEV_HOST": "127.0.0.1", "JEV_START_TIMEOUT": "5", "JEV_EAGER": "1",
                "FAKE_ARGS": str(root / "args.json"),
            }
            with patch.dict(os.environ, environment):
                control = load("control")
                try:
                    self.assertEqual(control.start(), 0)
                    self.assertEqual(control.status(), 0)
                    record = control.read_record()
                    self.assertEqual(control.start(), 0)
                    self.assertEqual(control.read_record(), record)
                    arguments = json.loads((root / "args.json").read_text())
                    for flag in ("--enable-prefix-caching", "--enforce-eager", "--enable-lora"):
                        self.assertIn(flag, arguments)
                    self.assertEqual(arguments[arguments.index("--mamba-cache-mode") + 1], "align")
                finally:
                    control.stop()
                self.assertFalse(control.healthy())
                self.assertFalse(control.RECORD.exists())
                unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"],
                                             start_new_session=True)
                try:
                    identity = control.process_identity(unrelated.pid)
                    stale = {**identity, "start_ticks": identity["start_ticks"] + 1,
                             "checkout": str(control.ROOT), "url": control.URL}
                    control.RECORD.write_text(json.dumps(stale))
                    self.assertEqual(control.stop(), 0)
                    self.assertIsNone(unrelated.poll(), "A reused/stale PID must never be stopped")
                finally:
                    unrelated.terminate()
                    unrelated.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
