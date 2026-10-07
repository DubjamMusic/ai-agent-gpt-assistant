"""token-alchemist — Wave Q keel.

Role: prompt bound for ai-agent-gpt-assistant.
Knowledge: tool allow-list, secret redaction, no live model calls.
Does not read .env values. Does not import agent_v2.
"""

from __future__ import annotations

WAVE_ID = "Q-keel-20261007"
FIGURE_ID = "token-alchemist"
ALLOWED_TOOLS = frozenset({"search_notes", "draft_summary", "score_density"})
SECRET_MARKERS = ("sk-", "api_key=", "BEGIN PRIVATE", "ghp_")


def bind_prompt(task: str, tools: list[str]) -> dict:
    if not task or not task.strip():
        raise ValueError("token-alchemist refuses an empty task")
    lowered = task.lower()
    if any(marker.lower() in lowered for marker in SECRET_MARKERS):
        raise ValueError("token-alchemist refuses secret-shaped input")
    unknown = [name for name in tools if name not in ALLOWED_TOOLS]
    if unknown:
        raise ValueError(f"tools outside allow-list: {unknown}")
    keel = f"keel:{FIGURE_ID}:{len(tools)}:{len(task.strip())}"
    return {
        "waveId": WAVE_ID,
        "figureId": FIGURE_ID,
        "keel": keel,
        "tools": list(tools),
        "refusedSecrets": True,
        "liveModelCall": False,
    }


def _self_check() -> None:
    mark = bind_prompt("score the weekly density note", ["score_density"])
    expected = "keel:token-alchemist:1:29"
    if mark["keel"] != expected:
        raise SystemExit(f"FAIL {mark['keel']} != {expected}")
    try:
        bind_prompt("paste sk-live-secret", [])
    except ValueError:
        pass
    else:
        raise SystemExit("FAIL secret marker was accepted")
    print("PASS", mark["keel"])


if __name__ == "__main__":
    _self_check()
