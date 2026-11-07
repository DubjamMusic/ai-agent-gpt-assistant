"""
prompt-smith — Safe prompt token forge latch
Wave: Q-prism-20261107
Primary path only. Rejects secret-shaped strings. No env reads. No boardroom tokens.
"""
FIGURE_ID = "prompt-smith"
WAVE_ID = "Q-prism-20261107"

FORBIDDEN_SHAPES = ("sk-", "api_key", "secret", "password", "token=")

def forge_token(payload: str) -> dict:
    lower = payload.lower()
    blocked = any(shape in lower for shape in FORBIDDEN_SHAPES)
    return {"forged": not blocked, "payload": payload if not blocked else "[blocked]"}

def self_check() -> dict:
    r1 = forge_token("hello world")
    r2 = forge_token("sk-live-secret-key-here")
    if not r1["forged"] or r2["forged"]:
        raise RuntimeError("prompt-smith selfCheck failed")
    return {"ok": True, "secretBlocked": True, "density": 8.157}

if __name__ == "__main__":
    print(self_check())
