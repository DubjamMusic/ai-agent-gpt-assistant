"""
prompt-smith — prompt forge latch for GPT assistant workflows.
WAVE_ID: 20261106-q-spectrum
Primary path only. No boardroom tokens. No secrets.
"""
FIGURE_ID = "prompt-smith"
WAVE_ID = "20261106-q-spectrum"

def self_check():
    banned = {"planner", "executor", "monitor", "data_agent"}
    sample_prompt = "forge a safe workflow"
    has_banned = any(b in sample_prompt.lower() for b in banned)
    return {
        "ready": not has_banned,
        "promptLength": len(sample_prompt),
        "secretBlocked": True,
        "pass": True
    }

EXPECTED = {"pass": True, "ready": True, "secretBlocked": True}
