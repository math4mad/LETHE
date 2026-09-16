# FOUNDRY · 铸造厂 — reserved bench (dashed ring)

**Status:** unseated. This directory is reserved for Bench II of the park.

**The work it will hold:** automatic semic annotation — the pipeline that
pours new words into the lattice: LLM-assisted decomposition into the 16 bases,
HowNet/义原 alignment studies, NSM-lexicon crosswalks, and the audit rig that
refuses any annotation which cannot reproduce the ledger.

**The law it inherits:** Foundling §2 (rules cross, data does not); Park Law I
(every word decomposes; what resists awaits its basis).

## Intake fixture — staged cold, 2026-09-16

The bench is still unseated (no 义原模具 has crossed the gate), but the
fixture no longer waits on the treasury:

| part | path | what it does |
|---|---|---|
| I · 格模 | `schema/derive_lattice.py` → `schema/lattice16.json` | casts the 16-base lattice **from the engine's bytes**, never hand-copied; `--check` = drift audit (exit 1 on divergence, with the exact drifted cells enumerated) |
| II · 门禁 | `bin/validate-annotation.py` | four stations: 形制(schema) → 律 I(非零可分解) → 复现(ledger words must reproduce the in-book vector ≤1e-6) → 试浇(new words poured against the frozen ledger, posterior tilt reported as evidence) |
| III · 收据 | `../../../bin/pin.sh` | FOUNDLING §3 receipt writer — `pin` hashes + sidecar `.lethe-pin.json`; `--verify` re-checks and screams on tamper |
| 负例 | `drafts/*.json` | three standing samples: one ledger reproduction, one new-word pour (保温箱 → 55.64% 超市 / 43.27% 路边摊), one deliberate REFUSAL (霓虹灯: unknown basis `shiny`, out-of-range 1.2) |

When the semic dies arrives it is pinned first (§3), poured into `drafts/`, and
only the vectors that pass all four stations may touch the engine. REFUSED is a
lane, not a shame.


*When this bench is seated: replace the dashed ring on the gate with a live link.
One writer; announce by letter.*
