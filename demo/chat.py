#!/usr/bin/env python3
"""CLI for the same local MiniCPM5 service used by the dashboard."""
import sys
from llm import GLOBAL_MODEL


def run(prompt, history):
    result = GLOBAL_MODEL.generate_safe(prompt, history)
    if result["result"] != "PASS":
        print(result["error"], file=sys.stderr)
        return False
    print(result["text"])
    print(f"\n[{result['backend']} · {result['tokens']} tokens · "
          f"{result['tok_per_sec']} tok/s · {result['latency_ms']} ms]")
    history.extend([{"role": "user", "content": prompt},
                    {"role": "assistant", "content": result["text"]}])
    del history[:-12]
    return True


def main():
    history = []
    if len(sys.argv) > 1:
        return 0 if run(" ".join(sys.argv[1:]), history) else 1
    print("Admiral · MiniCPM5-2B · llama.cpp / CUDA (quit to exit)")
    while True:
        try:
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            return 0
        if prompt.lower() in ("quit", "exit"):
            return 0
        if prompt:
            run(prompt, history)


if __name__ == "__main__":
    sys.exit(main())
