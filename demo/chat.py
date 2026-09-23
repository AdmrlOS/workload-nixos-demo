#!/usr/bin/env python3
"""Interactive CLI chat tool for Admiral On-Device LLM on Jetson Orin."""
import sys
import time
from pathlib import Path

# Ensure directory is in path
sys.path.insert(0, str(Path(__file__).parent))

from llm import GLOBAL_MODEL


def print_banner():
    cuda_status = (
        f"\033[92mCUDA sm_87 Active ({GLOBAL_MODEL.cuda.device_name})\033[0m"
        if GLOBAL_MODEL.cuda.available
        else "\033[93mCPU Reference Mode (No NVIDIA driver injected)\033[0m"
    )
    print("\033[1;36m========================================================\033[0m")
    print("\033[1;37m        ADMIRAL ON-DEVICE LLM (JETSON ORIN)             \033[0m")
    print("   NixOS 26.05 Userspace · CUDA Driver API · sm_87      ")
    print(f"   Status: {cuda_status}")
    print("\033[1;36m========================================================\033[0m\n")


def stream_response(text: str, delay: float = 0.015):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def run_prompt(prompt: str):
    print(f"\033[1;34m[User]\033[0m {prompt}\n")
    print("\033[1;32m[Admiral Edge LLM]\033[0m")
    result = GLOBAL_MODEL.generate(prompt)
    stream_response(result["text"])
    print(
        f"\n\033[2m[{result['backend']} · {result['tokens']} tokens · "
        f"{result['tok_per_sec']} tok/s · {result['latency_ms']} ms]\033[0m\n"
    )


def interactive_loop():
    print_banner()
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
