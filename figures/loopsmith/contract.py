#!/usr/bin/env python3
"""Loopsmith Wave C contract. No network. No secrets."""
from __future__ import annotations

REQUIRED_STEPS = ("ingest", "decide_tool", "write_memory", "reply")


def run_loop(user_message: str) -> dict:
    if not user_message or not user_message.strip():
        raise ValueError("ingest failed: empty message")
    ingested = user_message.strip()
    decide_tool = "echo" if ingested.startswith("echo:") else "none"
    memory_write = {"role": "user", "content": ingested, "tool": decide_tool}
    reply = ingested[5:].strip() if decide_tool == "echo" else f"ack:{ingested}"
    return {
        "ingest": ingested,
        "decide_tool": decide_tool,
        "write_memory": memory_write,
        "reply": reply,
    }


def assert_contract(result: dict) -> None:
    missing = [step for step in REQUIRED_STEPS if step not in result]
    if missing:
        raise AssertionError(f"missing steps: {missing}")
    if not result["ingest"]:
        raise AssertionError("ingest empty")
    if result["write_memory"]["content"] != result["ingest"]:
        raise AssertionError("memory did not store ingest")
    if not result["reply"]:
        raise AssertionError("reply empty")


def main() -> None:
    cases = ["echo: prestige", "status check"]
    for raw in cases:
        assert_contract(run_loop(raw))
    echo = run_loop("echo: prestige")
    if echo["reply"] != "prestige" or echo["decide_tool"] != "echo":
        raise AssertionError("echo path broken")
    print(f"ok loopsmith {len(REQUIRED_STEPS)}/{len(REQUIRED_STEPS)}")


if __name__ == "__main__":
    main()
