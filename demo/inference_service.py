"""Supervise llama-server and publish evidence only after full CUDA offload."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


def offload_evidence(line):
    match = re.search(r"offloaded (\d+)/(\d+) layers to GPU", line)
    if match and int(match[1]) == int(match[2]) > 0:
        return {"device": "CUDA0", "gpu_layers": int(match[1]), "total_layers": int(match[2])}
    return None


def main():
    path = Path(os.environ.get("MINICPM_EVIDENCE", "/run/minicpm/cuda.json"))
    path.unlink(missing_ok=True)
    # llama.cpp routes library model-loading/offload messages at debug level.
    # Without this, a healthy GPU server never publishes readiness evidence.
    env = {**os.environ, "LLAMA_LOG_VERBOSITY": "4"}
    process = subprocess.Popen(sys.argv[1:], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    def stop(*_):
        process.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    evidence = None
    cuda_buffer = False
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            evidence = offload_evidence(line) or evidence
            cuda_buffer |= bool(re.search(r"CUDA0\s+model buffer size", line))
            if evidence and cuda_buffer and not path.exists():
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(evidence))
                temporary.replace(path)
        return process.wait()
    finally:
        path.unlink(missing_ok=True)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
