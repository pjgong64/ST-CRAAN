# ST-CRAAN — Spatio-Temporal Credit Risk Assessment Network

Code accompanying:

> Jungsatidkul, P.; Innate, S. Dynamic Forecasting of Agricultural Default from Space via
> Spatio-Temporal AI. *Forecasting* (under review).

ST-CRAAN is a dual-branch forecaster for agricultural loan default. A recurrent branch consumes a
nine-channel monthly Earth observation tensor (May–December); a dense branch consumes a static
financial and administrative vector with a trainable entity embedding for the ADM3 (sub-district)
identifier. The two representations are fused at a concatenation bottleneck and optimised end-to-end
under an asymmetric, cost-calibrated objective.

---

## Data availability

This repository contains **code only**. Two data tiers behave differently:

| Tier | Status | Where it comes from |
|---|---|---|
| Earth observation inputs | **Public** | MODIS MOD09GA V061 / MOD11A1 V061, CHIRPS v2.0, SMAP SPL4SMGP, VIIRS Black Marble VNP46A1, HDX ADM3 boundaries — all reachable through the Google Earth Engine catalogue under the collection IDs listed in Table 1 of the paper |
| Static borrower records | **Restricted** | Proprietary to the Bank for Agriculture and Agricultural Cooperatives (BAAC); supplied de-identified. Onward disclosure is barred by the bank's data-governance rules and by Thailand's Personal Data Protection Act B.E. 2562 (2019) |

No borrower-level record, and no derived file from which one could be reconstructed, is in this
repository or in its git history. Requests for access to the restricted tier go to BAAC.

`data/synthetic/` contains **two fully artificial borrower records** that reproduce the exact schema,
variable names, data types and admissible value ranges of both tensors. They are generated, not
sampled, and are not derived from any real customer record. Their purpose is structural verification:
the pipeline runs end-to-end against them so a reader can confirm the input contract and that the
graph compiles. **They are far too small to reproduce any estimate reported in the paper.**

---

## Repository layout

```
GEE/       Earth Engine scripts: collection filtering, QA masking, monthly compositing,
           ADM3 zonal reduction. Run first; they export the environmental panel.
R/         Everything downstream — hierarchical fallback imputation, phenological feature
           engineering, encoding and conditioning, ST-CRAAN definition and training,
           benchmarks, evaluation, SHAP and Risk Calendar interpretability.
data/
  synthetic/   Two artificial records (schema demonstration + smoke test).
models/    Serialised model artefacts written by the training scripts. Not tracked.
outputs/   Figures, tables and metric dumps. Not tracked.
```

## Environment

- R VERSION (see `sessionInfo()` output in `docs/session-info.txt`)
- keras / tensorflow — TF VERSION, GPU build
- data.table, xgboost, h2o, ranger
- Google Earth Engine account with access to the public collections above

Training was run on a single NVIDIA GeForce RTX 4060 (8 GB). Cumulative 10-fold training cost was
about 54 minutes; full-portfolio inference over 1,496,721 borrowers takes about 32 seconds.

## Reproducing the pipeline

1. **Environmental panel.** Run the scripts in `gee/` in numerical order. They export one row per
   (ADM3, month) with the nine channels. Public data only — this step is fully reproducible.
2. **Static panel.** Requires the restricted BAAC extract. Substitute `data/synthetic/` to exercise
   the code path without it.
3. **Pipeline.** Run the scripts in `R/` in numerical order. Every conditioning parameter —
   skewness diagnostics, winsorisation bounds, encoding statistics, per-feature minima and maxima —
   is estimated on the 2020–2021 Development Set only and applied unchanged to the 2022 Test Set.

## Citation

See `CITATION.cff`, or cite the archived release:

> Jungsatidkul, P.; Innate, S. ST-CRAAN: Spatio-Temporal Credit Risk Assessment Network (vX.Y.Z).
> Zenodo, YEAR. https://doi.org/CONCEPT-DOI

## License

MIT — see `LICENSE`. The license covers the code in this repository only. It confers no rights over
BAAC data.
