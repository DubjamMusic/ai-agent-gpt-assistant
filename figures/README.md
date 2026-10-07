# token-alchemist

Wave `Q-keel-20261007`. Prompt bound for the Python assistant. Not a boardroom enum and not a clone of forge-caller.

- Role: prompt bound / tool allow-list smith
- Primary path: `figures/token_alchemist.py`
- Knowledge: allowed tools `search_notes`, `draft_summary`, `score_density`; secret markers must fail closed; no live OpenAI call
- Check: `python figures/token_alchemist.py` expects `PASS keel:token-alchemist:1:28`
- Merge policy: PR only
