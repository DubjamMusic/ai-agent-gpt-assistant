# VOLT-BROKER-26 — Assistant Watt

WAVE_ID: `2026-09-26-k-axle`
Repo: `DubjamMusic/ai-agent-gpt-assistant`
Merge policy: **pr-only**

## Job
Bind a scored watt card for the Python assistant so token budget, memory.py, and tools.py stay the live surface without rewriting agent_v2.py. This is not loopsmith (Wave C circuit). New figure, new path, new wave id.

## Responsibilities
- Own `watt.json` only.
- Print a reproducible density to three decimals.
- Keep merge policy pr-only.
- Never commit secrets from `.env`.

## Knowledge required
- Density = (N^wN * V^wV * S^wS * D^wD)^(1 / totalWeight).
- Watt scores the assistant loop; it does not ship a new OpenAI client.
- Chair holds merge.

## Primary path this wave
`figures/volt-broker/**` only.

## Out of scope
Do not rewrite `agent.py`, `agent_v2.py`, or `memory.py` this wave. Model-swap and secret rotation stay on a separate issue.
