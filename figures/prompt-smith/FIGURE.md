# prompt-smith

**WAVE_ID**: `2026-11-03-q-ember`

## Job
Forge safe, reusable prompt templates for the GPT assistant that stay free of secret material and banned role tokens.

## Responsibilities
- Emit one prompt card per wave under `figures/prompt-smith/`.
- Assert the prompt contains no API key patterns or .env references.
- Score a simple correctness proxy (length between 40-200 chars, no banned tokens).
- PR-only writes. Do not touch agent.py or tools.py in this wave.

## Required knowledge
- Structure of agent_v2.py and memory.py.
- OpenAI function-calling patterns used in the repo.
- HustleCodex safety rules around secrets.

## Pass test
```js
// expected: { pass: true, length: 112, banned: false, secret: false }
```

## Merge policy
PR only. Chair holds the gate.
