#!/usr/bin/env python3
"""Interactive CLI chat tool for Admiral on-device Qwen2.5 on Jetson Orin."""
import sys
import time
from pathlib import Path

# Ensure directory is in path
sys.path.insert(0, str(Path(__file__).parent))

from llm import GLOBAL_MODEL


def print_banner():
    if GLOBAL_MODEL.cuda.available:
        status_line = f"\033[92mCUDA sm_87 Active ({GLOBAL_MODEL.cuda.device_name})\033[0m"
    else:
        status_line = (
            "\033[91m[FAIL] CUDA Driver API Missing — CPU Fallback Strictly Disabled\033[0m"
        )

    print("\033[1;36m========================================================\033[0m")
    print("\033[1;37m        ADMIRAL ON-DEVICE QWEN (JETSON ORIN)            \033[0m")
    print("   Model: Qwen/Qwen2.5-0.5B-Instruct · ChatML Template  ")
    print("   Userspace: NixOS 26.05 · Backend: CUDA Driver API    ")
    print(f"   Status: {status_line}")
    print("\033[1;36m========================================================\033[0m\n")


def stream_response(text: str, delay: float = 0.015):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def run_prompt(prompt: str):
    print(f"\033[1;34m[User]\033[0m {prompt}\n")
    print("\033[1;32m[Qwen2.5 Edge Assistant]\033[0m")

    if not GLOBAL_MODEL.cuda.available:
        print(
            "\033[1;31mError: CUDA Driver API (libcuda.so.1) required.\033[0m\n"
            "CPU fallback is strictly disabled to demonstrate real hardware GPU execution on Jetson Orin.\n"
            f"Details: {GLOBAL_MODEL.cuda.init_error}\n"
        )
        sys.exit(1)

    result = GLOBAL_MODEL.generate(prompt)
    stream_response(result["text"])
    print(
        f"\n\033[2m[{result['backend']} · {result['tokens']} tokens · "
        f"{result['tok_per_sec']} tok/s · {result['latency_ms']} ms]\033[0m\n"
    )


def interactive_loop():
    print_banner()
    if not GLOBAL_MODEL.cuda.available:
        print(
            "\033[1;31mError: CUDA Driver API (libcuda.so.1) required.\033[0m\n"
            "CPU fallback is strictly disabled to demonstrate real hardware GPU execution on Jetson Orin.\n"
            f"Details: {GLOBAL_MODEL.cuda.init_error}\n"
        )
        sys.exit(1)

    print("Type your message below (or 'exit' / 'quit' to exit):\n")
    while True:
        try:
            prompt = input("\033[1;34m>>> \033[0m").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            run_prompt(prompt)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break


def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        if prompt in ["--benchmark", "-b"]:
            prompt = "Run CUDA inference benchmark"
        print_banner()
        run_prompt(prompt)
    else:
        interactive_loop()


if __name__ == "__main__":
    main()
