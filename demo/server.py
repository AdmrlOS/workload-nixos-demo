"""Admiral NixOS Edge Demo Server: On-Device Qwen2.5 LLM Chat & Checked CUDA Telemetry."""
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Ensure local imports work
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from llm import GLOBAL_MODEL

START = time.monotonic()
LOCK = threading.Lock()
GPU = {"result": "PENDING", "stage": "starting", "timestamp": time.time()}


def run_gpu_probe():
    try:
        run = subprocess.run([sys.executable, str(ROOT / "gpu.py")], capture_output=True, text=True, timeout=25)
        value = json.loads(run.stdout)
        if run.returncode != 0 and value.get("result") == "PASS":
            raise ValueError("Probe returned nonzero with PASS")
    except Exception as error:
        value = {"result": "FAIL", "stage": "probe_process", "error": str(error), "timestamp": time.time()}
    return value


def gpu_loop():
    global GPU
    while True:
        value = run_gpu_probe()
        with LOCK:
            GPU = value
        print(json.dumps({"event": "gpu-check", **value}), flush=True)
        try:
            state = Path(os.environ.get("STATE_DIRECTORY", "/var/lib/admiral-demo"))
            state.mkdir(parents=True, exist_ok=True)
            (state / "gpu-result.tmp").write_text(json.dumps(value, indent=2))
            (state / "gpu-result.tmp").replace(state / "gpu-result.json")
        except OSError:
            pass
        time.sleep(30)


def status():
    with LOCK:
        gpu = dict(GPU)
    try:
        os_release = platform.freedesktop_os_release()["PRETTY_NAME"]
    except OSError:
        os_release = platform.system()
    try:
        version = Path("/etc/admiral-demo/version").read_text().strip()
    except OSError:
        version = "development"
    try:
        load = os.getloadavg()[0]
    except OSError:
        load = 0.0

    return {
        "os": os_release,
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "uid": os.getuid(),
        "groups": os.getgroups(),
        "uptime": round(time.monotonic() - START),
        "load": load,
        "version": version,
        "gpu": gpu,
        "simulation": True,
        "system": os.path.realpath("/run/current-system"),
        "llm": {
            "model": GLOBAL_MODEL.MODEL_ID,
            "cuda_active": GLOBAL_MODEL.cuda.available,
            "result": "PASS" if GLOBAL_MODEL.cuda.available else "FAIL",
            "device": GLOBAL_MODEL.cuda.device_name if GLOBAL_MODEL.cuda.available else "None (GPU Required)",
            "compute_capability": GLOBAL_MODEL.cuda.compute_cap,
            "backend": "CUDA Driver API (libcuda.so.1 / sm_87)" if GLOBAL_MODEL.cuda.available else "None",
            "cpu_fallback": False,
            "system_prompt_configured": True,
            "total_tokens": GLOBAL_MODEL.total_tokens_generated,
            "inferences": GLOBAL_MODEL.total_inferences,
            "last_tok_per_sec": GLOBAL_MODEL.last_tok_per_sec,
            "last_latency_ms": GLOBAL_MODEL.last_latency_ms,
        },
    }


def scene(t):
    """Deterministic simulated scene retained for backwards compatibility."""
    x, y = 6 + 3.6 * math.cos(t * 0.13), 4 + 2.3 * math.sin(t * 0.13)
    heading = math.atan2(2.3 * math.cos(t * 0.13), -3.6 * math.sin(t * 0.13))
    points = []
    for i in range(144):
        angle = i * math.tau / 144
        dx, dy = math.cos(angle), math.sin(angle)
        candidates = []
        if abs(dx) > 1e-9:
            candidates.append(((12 if dx > 0 else 0) - x) / dx)
        if abs(dy) > 1e-9:
            candidates.append(((8 if dy > 0 else 0) - y) / dy)
        distance = min(candidates)
        points.append([round(x + dx * distance, 3), round(y + dy * distance, 3)])
    return {"x": x, "y": y, "heading": heading, "points": points, "simulated": True}


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, data, code=200):
        content = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        route = url.path
        if route == "/api/status":
            self._send_json(status())
        elif route == "/api/chat":
            params = urllib.parse.parse_qs(url.query)
            prompt = params.get("prompt", [""])[0] or params.get("message", [""])[0]
            res = GLOBAL_MODEL.generate_safe(prompt)
            self._send_json(res)
        elif route == "/api/gpu":
            with LOCK:
                gpu = dict(GPU)
            self._send_json(gpu)
        elif route == "/api/scene":
            self._send_json(scene(time.monotonic() - START))
        elif route == "/healthz":
            self._send_json({"dashboard": "ok", "llm": "ready"})
        elif route == "/":
            content = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)
        elif route == "/admiral-logo.svg":
            content = (ROOT / "admiral-logo.svg").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404)

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        route = url.path
        if route == "/api/chat":
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8") if raw else "{}")
                prompt = payload.get("prompt") or payload.get("message") or ""
                result = GLOBAL_MODEL.generate_safe(prompt)
                self._send_json(result)
            except Exception as e:
                self._send_json({"result": "FAIL", "error": str(e)}, code=400)
        elif route == "/api/gpu/probe":
            value = run_gpu_probe()
            with LOCK:
                global GPU
                GPU = value
            self._send_json(value)
        else:
            self.send_error(404)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    threading.Thread(target=gpu_loop, daemon=True).start()
    print("Admiral NixOS Edge AI demo listening on :8080", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
