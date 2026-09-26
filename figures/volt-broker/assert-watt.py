#!/usr/bin/env python3
import json
from pathlib import Path

card = json.loads((Path(__file__).with_name("watt.json")).read_text())

if card["figureId"] != "volt-broker":
    raise SystemExit("bad figureId")
if card["predecessorNotThisWave"] == card["figureId"]:
    raise SystemExit("reused predecessor")
if card["waveId"] != "2026-09-26-k-axle":
    raise SystemExit("bad waveId")

banned = ["planner", "executor", "monitor", "data_agent"]
identity = " ".join(
    [card["figureId"], card["codename"], card["role"], *card.get("knowledge", [])]
).lower()
for token in banned:
    if token in identity:
        raise SystemExit("banned token " + token)

w, s = card["weights"], card["sample"]
total = w["N"] + w["V"] + w["S"] + w["D"]
density = (s["N"] ** w["N"] * s["V"] ** w["V"] * s["S"] ** w["S"] * s["D"] ** w["D"]) ** (1 / total)
rounded = round(density, 3)
if rounded != card["expectedDensity"]:
    raise SystemExit(f"density mismatch got={rounded} expected={card['expectedDensity']}")
if card["mergePolicy"] != "pr-only":
    raise SystemExit("merge policy must be pr-only")
print(f"ok volt-broker density={rounded}")
