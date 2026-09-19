# DIAGNOSTICA · 诊断台 — bench IV, seated 2026-09-16 (prep for tomorrow)

**Status:** SEATED, half-lit. The bench where the park learns to *notice*
that its concept spaces are moving.

**The work it holds:** the GP drift-diagnostic line — the instrument that
watches a concept space slide from 夜市 toward 便利店, decides whether the
drift is gradual or abrupt, and prunes the basis functions that carry no
signal.

## Files on this bench

| file | role |
|---|---|
| `gp-diagnoistoc_v2.py` | CLI edition. Bayesian model comparison (log marginal likelihood, P(abrupt\|D)) replaces v1's ad-hoc mean+2σ threshold; three ground-truth regimes (gradual / step / ramp) make the detector falsifiable; spectral truncation is scored by measured reconstruction RMSE. |
| `gp_diagnoistoc_v2_marimo.py` | Reactive edition. Same pipeline with live controls: regime dropdown, noise σ, H_smooth / H_abrupt length scales, expansion order, truncation tolerance. Verified via `marimo export html` (all cells execute, zero errors). |

The ancestral v1 pair crossed from the root on 2026-09-18 (owner's order:
"gp 诊断器那批文件要放到合适的地方"): `gp_diagnostica_v1.py` (was
`gp-diagnoistoc.py` at root) and `gp_diagnoistoc_marimo.py` — the
constant-signal truncation tautology is preserved here as a cautionary
fossil, renamed to bench convention. Figures retired to `figures/`
(`full_pipeline_with_truncation.png`, `gp_diag_v2.png`).

## Run

```bash
python3 gp-diagnoistoc_v2.py                 # CLI (from this directory)
../run_gp_marimo.sh --run v2                 # dashboard (from repo root)
# or directly:
python3 -m marimo run gp_diagnoistoc_v2_marimo.py
```

## Known honest failures (features, not bugs)

- **ramp → ABRUPT [MISS]:** a ramp's derivatives are discontinuous at its
  two kinks; the verdict is arguably defensible.
- **step → 16/16 terms kept:** Gibbs-type slow Legendre decay; truncation
  refuses to lie about a discontinuity.

## Tomorrow's ledger · 继续添加

- [ ] BOCD (Bayesian Online Change Point Detection) as a third detector
- [ ] Feed real vocabularies: run drift diagnosis on 雷荷波 ball-array
      occupancy over time, not synthetic sigmoid
- [ ] Sensitivity sweep: length-scale grid × noise, report the
      distinguishability surface of the two hypotheses
- [ ] Marginal-likelihood–weighted model averaging for the mu used in
      spectral expansion
- [ ] LETHE pin the bench outputs (`bin/pin.sh`) and archive to JSON
- [ ] A third edition: marimo cells emitting into `website/` console
