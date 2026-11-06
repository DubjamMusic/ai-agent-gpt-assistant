# prompt-smith

**Job role:** Prompt Forge Latch

**Responsibilities:**
- Forge workflow prompts that stay free of banned boardroom tokens.
- Block secret-shaped keys before any OpenAI call.
- Keep primary path isolated to agents/figures/.

**Knowledge required:**
- Python module contracts
- GPT assistant prompt hygiene
- Banned token list (planner, executor, monitor, data_agent, planner-alpha, executor-beta)

**Test:** from agents.figures.prompt_smith.contract import self_check; assert self_check()["pass"]

**Gate:** PR only. Chair holds merge. No key material.
