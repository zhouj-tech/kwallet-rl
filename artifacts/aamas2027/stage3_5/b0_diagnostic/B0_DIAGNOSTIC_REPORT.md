# B0 Hybrid Diagnostic Report (Stage 3.5)

**POST-HOC EXPLORATORY DIAGNOSTIC — not confirmatory.** No model was retrained; no frozen Stage 2/Stage 3 artifact was modified. The two hybrids only re-combine the frozen BF-T0.5 rule components with the frozen SC-FAC seed=123 policy on the frozen NEW12-v1 streams.

## 1. Pilot table

| Hybrid | C | Hybrid Money | SC123 Money | BF Money | Δ vs SC | Δ vs BF | Gap recovery |
|---|---|---|---|---|---|---|
| H1 | 800 | 4278.415 | 3897.975 | 4661.325 | +380.440 | -382.909 | 0.498 |
| H1 | 1200 | 14322.539 | 14327.010 | 15298.890 | -4.471 | -976.352 | -0.005 |
| H2 | 800 | 4029.944 | 3897.975 | 4661.325 | +131.969 | -631.380 | 0.173 |
| H2 | 1200 | 12656.502 | 14327.010 | 15298.890 | -1670.507 | -2642.388 | -1.719 |

Gap recovery = (M_hybrid − M_SC123) / (M_BF − M_SC123). 0 = none recovered; 1 = reached BF; >1 = exceeded BF.

## 2. Settled-value / flush-cost decomposition

| Hybrid | C | hybrid settled | hybrid flushes | flush cost | Δsettled vs SC | Δflushcost vs SC | Δsettled vs BF | Δflushcost vs BF |
|---|---|---|---|---|---|---|---|---|
| H1 | 800 | 6320.153 | 204.174 | 2041.738 | +249.607 | -130.833 | -919.180 | -536.271 |
| H1 | 1200 | 20437.701 | 611.516 | 6115.163 | -109.963 | -105.492 | -277.293 | +699.058 |
| H2 | 800 | 6222.540 | 219.260 | 2192.596 | +151.994 | +20.025 | -1016.793 | -385.413 |
| H2 | 1200 | 17180.644 | 452.414 | 4524.142 | -3367.020 | -1696.513 | -3534.351 | -891.962 |

Per-pilot means (across 12 regimes of 200 episodes):

| Hybrid | C | accepted/ep | insufficient drops | oversize drops | SC settled | SC flushes | BF settled | BF flushes | regimes > SC | regimes ≥ BF |
|---|---|---|---|---|---|---|---|---|---|---|
| H1 | 800 | 239.63 | 34.236 | 726.134 | 6070.546 | 217.257 | 7239.333 | 257.801 | 12 | 4 |
| H1 | 1200 | 591.66 | 7.812 | 400.531 | 20547.664 | 622.065 | 20714.995 | 541.610 | 4 | 0 |
| H2 | 800 | 235.37 | 38.492 | 726.134 | 6070.546 | 217.257 | 7239.333 | 257.801 | 6 | 0 |
| H2 | 1200 | 509.66 | 89.805 | 400.531 | 20547.664 | 622.065 | 20714.995 | 541.610 | 4 | 4 |

## 3. Per-regime directional summary

- H1 C=800: 12/12 regimes above SC123; best Δ vs SC +672.63 (LNB), worst +279.17 (US).
- H1 C=1200: 4/12 regimes above SC123; best Δ vs SC +646.35 (LNS), worst -505.74 (PLS).
- H2 C=800: 6/12 regimes above SC123; best Δ vs SC +1175.78 (PLS), worst -644.25 (TLNS).
- H2 C=1200: 4/12 regimes above SC123; best Δ vs SC +191.47 (PLS), worst -3599.45 (US).

Full per-regime numbers: `B0_PER_REGIME.csv`.

## 4. Screening decision (heuristics, not significance)

### H1

- Rule A (improve over SC123 at BOTH capacities): **False**
- Rule B (> +200 Money or > 30% gap recovery at either capacity): **True**
- Δ vs SC: C800 +380.440, C1200 -4.471; recovery: C800 0.498, C1200 -0.005
- **Flagged promising for expansion (PI approval required for any extra seeds): True**

### H2

- Rule A (improve over SC123 at BOTH capacities): **False**
- Rule B (> +200 Money or > 30% gap recovery at either capacity): **False**
- Δ vs SC: C800 +131.969, C1200 -1670.507; recovery: C800 0.173, C1200 -1.719
- **Flagged promising for expansion (PI approval required for any extra seeds): False**

## 5. Provenance and caveats

- Frozen references loaded from Stage 3 tables and cross-checked directly against the frozen Stage 2 raw episodes (tarball SHA 576c9baa00c44385…).
- SC123 macro references (raw, equal to Stage 3): C800 3897.975417, C1200 14327.009583; BF: C800 4661.324583, C1200 15298.890417.
- Before pilots, the B0 evaluator reproduced every frozen SC123 and BF-T0.5 episode metric EXACTLY (see outputs/parity/parity_ep200.json).
- H1 may select a flush wallet equal to the BF settlement wallet; when that happens E0 processes the flush first and the settlement becomes infeasible for that step. This is a genuine interaction effect, preserved by submitting the joint action to the unmodified E0.
- One deterministic pilot per cell; no uncertainty estimates, no multiple seeds, no new significance claims.
