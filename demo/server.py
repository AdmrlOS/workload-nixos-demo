"""Read-only dashboard; simulated robot telemetry and independent real GPU check."""
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).parent
START = time.monotonic()
LOCK = threading.Lock()
GPU = {"result": "PENDING", "stage": "starting", "timestamp": time.time()}


def gpu_loop():
    global GPU
    while True:
        try:
            run = subprocess.run([sys.executable, str(ROOT / "gpu.py")], capture_output=True, text=True, timeout=25)
            value = json.loads(run.stdout)
            if run.returncode != 0 and value.get("result") == "PASS":
                raise ValueError("Probe returned nonzero with PASS")
        except Exception as error:
            value = {"result": "FAIL", "stage": "probe_process", "error": str(error), "timestamp": time.time()}
        with LOCK:
            GPU = value
        print(json.dumps({"event": "gpu-check", **value}), flush=True)
        try:
            state = Path(os.environ.get("STATE_DIRECTORY", "/var/lib/admiral-demo"))
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
    return {"os": os_release, "kernel": platform.release(), "architecture": platform.machine(),
            "hostname": platform.node(), "uid": os.getuid(), "groups": os.getgroups(),
            "uptime": round(time.monotonic() - START), "load": os.getloadavg()[0],
            "version": version, "gpu": gpu, "simulation": True,
            "system": os.path.realpath("/run/current-system")}


def scene(t):
    # Deterministic warehouse patrol; synthetic lidar returns against room walls.
    x, y = 6 + 3.6 * math.cos(t * .13), 4 + 2.3 * math.sin(t * .13)
    heading = math.atan2(2.3 * math.cos(t * .13), -3.6 * math.sin(t * .13))
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
    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/api/status":
            content, kind = json.dumps(status()).encode(), "application/json"
        elif route == "/api/scene":
            content, kind = json.dumps(scene(time.monotonic() - START)).encode(), "application/json"
        elif route == "/healthz":
            content, kind = b'{"dashboard":"ok"}', "application/json"
        elif route == "/":
            content, kind = (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8"
        elif route == "/admiral-logo.svg":
            content, kind = (ROOT / "admiral-logo.svg").read_bytes(), "image/svg+xml"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    threading.Thread(target=gpu_loop, daemon=True).start()
    print("Admiral robotics demo listening on :8080", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
