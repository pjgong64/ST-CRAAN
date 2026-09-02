# Synthetic smoke-test sample

`smoke_sample.csv` — two artificial borrowers, three crop years each (6 rows × 137 columns).

## What this is

Two **fully artificial** borrower records. They were generated, not sampled, subsetted, perturbed or
anonymised from real data; no real customer record was used as a seed at any point, and the ADM3
codes (`999901`, `999902`) are deliberately outside the Thai administrative code space, so they map
to the entity-embedding layer's out-of-vocabulary row rather than to any real sub-district.

The file exists to do one thing: **show exactly which features enter the model and in what shape**, so
that a reader without access to the restricted BAAC extract can see the input contract and run the
pipeline end to end for structural verification.

The header is byte-identical to the schema of the real input table, in the same column order.

## Layout of the 137 columns

| Block | Columns | Contents |
|---|---|---|
| Static | 1–48 | Identifiers, savings, loan and repayment, agricultural production profile, and the derived financial ratios of Supplementary Table S2. Column 33, `Y`, is the default label |
| Join key | 49 | `ADM3` again — the key linking the borrower row to its sub-district environmental sequence |
| Sequential | 50–137 | 11 environmental channels × 8 months (May–December), wide format: `NDVI5 … NDVI12`, `NDWI…`, `LST…`, `Precipitation…`, `SMAP…`, `NTL…`, `SAI_NDVI…`, `VHI…`, `SAI_NDWI…`, `Rainfall_Deficit…`, `Heat_Stress_Days…` |

Note that 11 environmental channels are present here but only **nine** reach the model. `SAI_NDVI`
and `SAI_NDWI` were computed during feature engineering and then dropped at the screening step
described below; they are retained in the file because the raw extract contains them.

## The two records

| ID | Profile | Label |
|---|---|---|
| 900001 | Deleveraging steadily — outstanding balance falls across the three years, deposits accumulate, margins stay positive; ordinary growing seasons | `Y = 0, 0, 0` |
| 900002 | Leverage rising against a shrinking margin; in year 3 the environmental sequence carries the stress signature the forecaster is meant to detect — canopy water (NDWI) turns down before greenness (NDVI), LST and Heat Stress Days climb through the October–November pre-harvest window, and the rainfall deficit deepens | `Y = 0, 0, 1` |

Both classes are therefore exercised, and both branches of the asymmetric loss receive a gradient.

## Value ranges in this file

| Field | Range |
|---|---|
| `NDVI` | 0.147 – 0.346 |
| `NDWI` | 0.209 – 0.619 |
| `LST` (°C) | 34.70 – 42.63 |
| `Precipitation` | 0.07 – 14.20 |
| `SMAP` | 2.40 – 14.51 |
| `NTL` | 66.1 – 152.9 |
| `VHI` | 0.189 – 0.638 |
| `Rainfall_Deficit` | −63.2 – 28.4 |
| `Heat_Stress_Days` | 0 – 22 |
| `Out` (THB) | 104,950 – 322,400 |
| `CL` (THB) | 240,000 – 345,000 |
| `eInc` (THB) | 51,114 – 88,444 |

Derived ratios are arithmetically consistent with the base variables they are built from — for example
`Credit_Utilization_Rate = Out / CL`, `Profitability_Buffer = (eInc − eCost) / eInc`,
`Debt_per_Rai = Out / Area`, and `eInc = (eYie / 1000) × ePrcton`. Year 1 has no preceding period, so
the four longitudinal features (`Debt_Reduction`, `Debt_Momentum`, `Saving_Growth`,
`Saving_Momentum`) and `Previous_Default` carry the sentinel `−1`.

---

## These columns are NOT what the network consumes

This is the most important thing the file demonstrates. **The values here are raw, on their natural
measurement scales, and must pass through the conditioning pipeline before any tensor is built.**
Feeding them to the model as they stand does not merely degrade accuracy — it prevents the recurrent
branch from training at all.

**Why.** In this file `NDVI` lives around 0.15–0.35 while `CL` reaches 345,000: roughly six orders of
magnitude between two features that share the same input layer. The LSTM gates are sigmoid and
hyperbolic-tangent functions. A pre-activation formed from an input of order 10⁵ saturates them
immediately, and a saturated gate has a derivative of essentially zero — σ′(z) = σ(z)(1 − σ(z)) → 0.
Because the recurrence multiplies these derivatives together at every one of the eight timesteps, the
gradient reaching the early months of the season vanishes. The network then learns nothing from
exactly the early-vegetative window the forecaster depends on, while the unscaled financial features
dominate the loss surface purely by magnitude.

**The transformation applied before modelling**, in order:

1. **`log(1 + x)`** on the features with pronounced positive skew — the monetary variables
   (`Out`, `CL`, `eCost`, `eInc`, `Depb`, `Dep`, `Debt_per_Rai` …), whose distributions are long-tailed
   across the portfolio.
2. **Winsorisation at the 1st and 99th percentiles**, bounding the influence of extreme borrowers on
   the scaling statistics.
3. **Min–max scaling to the unit interval**, so every channel and every financial covariate enters the
   network on a comparable scale and no gate is saturated by construction.

Every parameter this pipeline needs — the skewness diagnostics, the winsorisation bounds, the encoding
statistics, and the per-feature minima and maxima — is estimated **on the 2020–2021 Development Set
only** and applied unchanged to the 2022 Test Set, so no information from the forecast year leaks
backwards into the fitted pipeline.

## Screening comes after the transformation, and it changes the column count

Transformation is not the last step before the tensors are built.

First, four columns leave the feature matrix by name — `Year`, `ID`, `Y` and `ADM3`. `Year` and `ID`
are record keys, `Y` is the label, and `ADM3` is removed from the *numeric* matrix precisely because
it is not discarded: it is routed to the trainable entity-embedding layer instead.

`ADM3RL`, the **residential** sub-district, is a different variable and stays. It need not be the same
sub-district as the cultivated plot, so it carries information the cultivation identifier does not.
It is not fed as a raw administrative code: it is replaced by its target encoding with additive
Bayesian smoothing, the scalar `TE_ADM3RL` that appears in the interpretability analysis of the paper.
The same treatment applies to the other categorical codes in the file — planting month is encoded
cyclically as a sine/cosine pair, and rice variety is dummy-encoded. Together with the remaining
numeric columns these form the static feature set.

Because the target encoding is fitted from the label, its statistics are computed on the Development
Set only, exactly like the scaling parameters, and then applied unchanged to the forecast year.

The static feature count follows from those steps:

```
static columns in this file                                        48
  minus drop_cols: Year, ID, Y, ADM3                               -4   ->  44
  plus  TE_ADM3      (target encoding of the dropped identifier)   +1   ->  45
  plus  MP   -> MP_sin, MP_cos                                     +1   ->  46
  plus  MH   -> MH_sin, MH_cos                                     +1   ->  47
  plus  IDRV -> 2 dummies (3 levels, first as reference)           +1   ->  48
                                                                       ------
static branch input                                                        48
```

Note that `TE_ADM3` and the ADM3 entity embedding coexist: the embedding is concatenated **on top of**
the encoded scalar, not in place of it, so the neural models see every spatial feature the tree
benchmarks see plus a learned 9-dimensional representation. The dense-branch input is therefore
48 + 9 = 57.

Then, with every feature on the unit interval, each candidate is screened for association with the
binary default target using a **point-biserial effect size** under a deliberately **lenient**
threshold: a feature is dropped only if **|r_pb| < 0.01**. The intent is to remove channels that carry
essentially nothing, not to prune aggressively toward a minimal feature set — at 0.01 a feature has to
be almost perfectly uninformative to fail.

Screening is run on the **Development Set (2020–2021) only**, for the same reason the scaling
statistics are: a screen fitted on the forecast year would leak information backwards into the choice
of inputs, which is a subtler form of the same leakage the Out-of-Time protocol exists to prevent.

Two channels failed the screen: **`SAI_NDVI`** and **`SAI_NDWI`**, the two Standardized Anomaly
Indices. Each spans eight months, so 16 columns leave the sequential block.

| Stage | Static branch | Sequential branch |
|---|---|---|
| Present in this file | columns 1–48 | 11 channels × 8 months = **88** |
| After screening → into the model | **48** features | 9 channels × 8 months = **72** |

The sequential tensor the recurrent branch consumes is therefore shaped `(N, 8, 9)` — eight monthly
timesteps, nine channels. The dense branch receives the 48 static features concatenated with the
9-dimensional ADM3 embedding vector, giving an input width of 48 + 9 = 57.

If you run the pipeline against this file and the sequential tensor comes out with 11 channels rather
than 9, the screening step has not executed — check that it ran before the reshape, not after.

Two further points that this six-row extract makes visible:

- `ADM3` is **not** one of the 48 static covariates. It bypasses the numeric pipeline entirely and
  enters through the trainable entity-embedding layer.
- The sentinel `−1` in the year-1 longitudinal features is a *missingness marker*, not a measurement.
  It must be handled as such before scaling; a genuine `Debt_Momentum` of −1 (debt cleared in full)
  would otherwise be indistinguishable from "undefined in the first observed year".

---

## What this file cannot do

It cannot reproduce any number reported in the manuscript. Two borrowers cannot train a recurrent
encoder, cannot populate a 6,236-row embedding table, and cannot support any evaluation metric. Every
result in the paper comes from the restricted dataset described in the Data Availability Statement.
The sample is a contract specification and a smoke test — nothing more.
