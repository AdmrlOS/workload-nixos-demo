"""MiniCPM5 chat client for the local, CUDA-only llama.cpp service."""
import json
import os
from pathlib import Path
import threading
import time
import urllib.request

SYSTEM_PROMPT = """You are the Admiral edge demo assistant, powered by MiniCPM5-2B.
Answer clearly and concisely. Use the facts below for Admiral questions. For other
questions use your general knowledge. Say when you do not know; do not invent
product features, benchmark numbers, device specifications or live telemetry.

Admiral (https://admrl.co) is an edge operating system and fleet management
platform. It deploys and manages workloads on Linux edge devices, including
NVIDIA Jetson. Admiral owns the hardware platform: bootloader, Jetson BSP,
Tegra kernel, device tree, device access, host drivers and OTA platform updates.
This demo runs a NixOS 26.05 ARM64 userspace as an OCI system container on Admiral.
Nix provides declarative packages and systemd services; flake.lock pins nixpkgs.
The image exports the NixOS system closure and boots /init with systemd as PID 1.
It does not replace the host kernel. Hardware-specific boot, kernel, udev and
network configuration remain Admiral's responsibility.

Admiral injects the NVIDIA driver read-only at /run/admiral/nvidia/lib.
LD_LIBRARY_PATH exposes it to Nix-built applications. The demo user is UID 1000
with supplementary admiral-video GID 28. No Docker daemon, NVIDIA container
toolkit or JetPack installation is required inside the deployed workload.
The inference runtime is llama.cpp with CUDA runtime/cuBLAS supplied by Nix,
using the actual MiniCPM5-2B Q4_K_M GGUF weights. All transformer layers are
offloaded to CUDA on Jetson Orin (compute capability 8.7). Tokenization, sampling
and HTTP handling use the CPU; do not claim every operation runs on the GPU.
CPU-only model inference is disabled. Missing CUDA, model load failures and
incomplete offload are reported as failures, never simulated responses.

The browser chat sends requests to the Python dashboard on port 8080, which calls
a loopback-only llama-server. There is no browser inference or cloud LLM API.
The system prompt supplies this knowledge; there is no live web search or RAG.
An independent demo-gpu probe checks 65,539 coordinate transforms against a CPU
reference. Its PASS proves that probe, not LLM readiness or model performance.
Chat token counts and decode speed come from llama.cpp; request latency is wall
time. Read current measurements in the dashboard rather than guessing them.

Useful commands: demo-status; demo-chat \"What is Admiral?\"; timeout 30 demo-gpu;
cat /etc/os-release; nix --version; nix registry list;
cat /etc/admiral-demo/flake.lock; nix shell --offline nixpkgs#hello -c hello;
systemctl status robotics-demo minicpm; journalctl -u minicpm -f.
SSH uses keys on port 22. The dashboard uses port 8080. Admiral host management
port 2222 is separate. Deploy the ARM64 OCI image through Admiral, preserving
/init, standard runtime mounts and the injected driver. This demonstrates NixOS
userspace packaging, not universal compatibility with arbitrary NixOS modules.
"""


class MiniCPMEngine:
    MODEL_ID = "openbmb/MiniCPM5-2B"

    def __init__(self, base_url="http://127.0.0.1:8081", evidence_path=None):
        self.base_url = base_url
        self.evidence_path = Path(evidence_path or os.environ.get(
            "MINICPM_EVIDENCE", "/run/minicpm/cuda.json"))
        self._lock = threading.Lock()
        self.total_tokens_generated = 0
        self.total_inferences = 0
        self.last_latency_ms = 0.0
        self.last_tok_per_sec = 0.0

    def _request(self, path, payload=None, timeout=2):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)

    def status(self):
        result = {"model": self.MODEL_ID, "result": "FAIL", "cuda_active": False,
                  "cpu_fallback": False, "backend": "llama.cpp / CUDA",
                  "system_prompt_configured": True,
                  "total_tokens": self.total_tokens_generated,
                  "inferences": self.total_inferences,
                  "last_tok_per_sec": self.last_tok_per_sec,
                  "last_latency_ms": self.last_latency_ms}
        try:
            evidence = json.loads(self.evidence_path.read_text())
            if not (evidence["gpu_layers"] == evidence["total_layers"] > 0
                    and evidence["device"] == "CUDA0"):
                raise RuntimeError("Full CUDA model offload has not been verified")
            if self._request("/health").get("status") != "ok":
                raise RuntimeError("Model is loading")
            result.update(evidence, result="PASS", cuda_active=True)
        except Exception as error:
            result["error"] = f"MiniCPM5-2B unavailable: {error}. CPU fallback is strictly disabled."
        return result

    def messages(self, prompt, history=None):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
            raise ValueError("Prompt must contain 1–4000 characters")
        if history is None:
            history = []
        if not isinstance(history, list) or len(history) > 12:
            raise ValueError("History must contain at most 12 messages")
        cleaned = []
        for i, item in enumerate(history):
            role = "user" if i % 2 == 0 else "assistant"
            if (not isinstance(item, dict) or item.get("role") != role
                    or not isinstance(item.get("content"), str)
                    or len(item["content"]) > 8000):
                raise ValueError("History must contain alternating user/assistant messages")
            cleaned.append({"role": role, "content": item["content"]})
        if len(cleaned) % 2 or sum(len(m["content"]) for m in cleaned) + len(prompt) > 12000:
            raise ValueError("Conversation too long; clear chat and try again")
        return [{"role": "system", "content": SYSTEM_PROMPT}, *cleaned,
                {"role": "user", "content": prompt.strip()}]

    def generate(self, prompt, history=None):
        messages = self.messages(prompt, history)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Model is busy; try again when the current response finishes")
        try:
            state = self.status()
            if state["result"] != "PASS":
                raise RuntimeError(state["error"])
            start = time.monotonic()
            response = self._request("/v1/chat/completions", {
                "model": self.MODEL_ID, "messages": messages, "stream": False,
                "max_tokens": 512, "temperature": 1.0, "top_p": 0.95, "min_p": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            }, timeout=180)
            text = response["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("Model returned no answer; try a shorter question")
            tokens = response["usage"]["completion_tokens"]
            self.last_latency_ms = round((time.monotonic() - start) * 1000, 1)
            self.last_tok_per_sec = round(response.get("timings", {}).get("predicted_per_second", 0), 2)
            self.total_tokens_generated += tokens
            self.total_inferences += 1
            return {"result": "PASS", "text": text, "model": self.MODEL_ID,
                    "tokens": tokens, "tok_per_sec": self.last_tok_per_sec,
                    "latency_ms": self.last_latency_ms, "backend": "llama.cpp / CUDA",
                    "cuda_active": True, "device": state["device"],
                    "total_tokens_all_time": self.total_tokens_generated}
        finally:
            self._lock.release()

    def generate_safe(self, prompt, history=None):
        try:
            return self.generate(prompt, history)
        except Exception as error:
            return {"result": "FAIL", "model": self.MODEL_ID, "error": str(error),
                    "text": "", "cuda_active": False, "cpu_fallback": False}


GLOBAL_MODEL = MiniCPMEngine()
