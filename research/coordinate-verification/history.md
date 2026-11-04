# Coordinate-only spot verification — research history (2026)

This is the **maintained scientific record**, not a claim of deployment readiness.
The complete research implementation, all committed ledgers, per-pair scores,
protocols and intermediate negative results are preserved in the separate
[historical commit `d98da95`](https://github.com/lemuel-manske/meros/commit/d98da954d20f2d76d1e4921dd477e593bd72bffd).
The earlier Cycle 1 source is [`6c89988`](https://github.com/lemuel-manske/meros/commit/6c89988907e5e1aaa495835875d45c7be4ceeb82).
Do not delete the `experiment/coordinate-verification-audit` archive branch
without separately preserving the committed history. Locally ignored plots
under `results/` are **not guaranteed to exist in Git**.

For methodology, the authoritative historical documents are
[Cycle 1 checkpoint](https://github.com/lemuel-manske/meros/blob/6c89988907e5e1aaa495835875d45c7be4ceeb82/research/coordinate-verification/checkpoint.md)
and [Cycles 2–6 final report](https://github.com/lemuel-manske/meros/blob/d98da954d20f2d76d1e4921dd477e593bd72bffd/research/coordinate-verification/final-report.md).
The complete list of experimental methods and measurements is in the
[archived campaign ledger](https://github.com/lemuel-manske/meros/blob/d98da954d20f2d76d1e4921dd477e593bd72bffd/research/coordinate-verification/campaign-ledger.json).

## Question and data

Can two **unordered sets of head-spot coordinates alone** provide evidence that
two photographs show the same `Epinephelus itajara`, while rejecting an
unrelated individual without relying on a flexible forced geometric fit?

The researcher confirmed that photographs come from **different encounters**
between **2004 and 2013**. Individual dates and temporal order are unknown.
Of 22 annotated records, 18 contain eligible spot point sets (43–215 points);
these belong to **six individuals**: Rogerio (9 photos), Rafael (5), and
Athila/Jose/Kaue/Maira (one each). The 153 unordered image pairs comprise
46 genuine and 107 impostor pairs, but share the same animals and observations:
they are **not 153 independent biological trials**. Only **two individuals
have repeated encounters**. The dataset is final; more photos cannot be obtained.

The evaluation repeatedly used the same small historical dataset. Even after
holding out a photo or its identity when computing a particular threshold,
**model selection was not performed on independent unseen individuals**.
Do not claim validated population FAR, calibrated identity probabilities or
automatic biological identification.

## What was tried

The IDs below identify **scientific questions**, often with multiple variants.
A failure of one variant does not prove its broader method family impossible.
Original data, seeds, code, plots (where committed) and all variants are
specified in the archived report and ledgers.

| ID | Experiment / mechanism | Evidence and outcome |
| --- | --- | --- |
| E00 | Inventory, independent-encounter and leakage audit | 18 eligible photos, 6 identities, only 2 repeated; no defensible independent population test. |
| E01 | Reproduce count-only, Shape Context/TPS and triangle/TPS controls | Forced non-rigid fits and direction/seed instability; geometry is not identity confidence. |
| E02 | Uniform, shuffled-axis, jittered, inhibition and grid null patterns | Easy simulated impostors underestimate real impostor evidence; rerun the *entire* search under each null. |
| E03 | Vary hypothesis budgets and gallery sizes | Max-selected scores depend on search effort and gallery size; a single threshold is not portable. |
| E04 | No-warp nearest-neighbor distance / angular-gap distribution | Exploratory AUC 0.896; improvement over the geometric controls, but poor crop/missing-spot tolerance. |
| E05 | Joint local 3-/5-/8-neighbor, triangle and quadrilateral distributions | Three-neighbor joint energy descriptor selected as strong no-warp baseline (AUC 0.932); triangle/quad-only did not dominate. |
| E06 | Population-based motif rarity and disjoint motifs | Rarity weighting and disjoint motifs did not improve rejection on this corpus. |
| E08 | Descriptor-seeded similarity/affine verification with constraints | Did not improve genuine/impostor discrimination; TPS escalation not justified. |
| E09 | Resubstitution versus source-only versus bilateral point holdout | Similarity AUC 0.759 → 0.587 → 0.478; apparent registration did not generalize. |
| E10 | Local graph / conserved neighbor-edge consistency | Partial-crop resilience, but weaker overall separation and some incorrect accepts. |
| E11 | Density/repulsion and block-preserving nulls | Structured nulls expose false confidence missed by uniform random controls. |
| E12 | Combine graph, neighborhood and statistical evidence | Some AUC improvements but worse strongest-impostor rejection; not selected. |
| E15–E16 | Missing/spurious spots, crop, rotation, scale, anisotropy and noise | Crop and omissions remain critical; synthetic cases are not new independent observations. |
| E17 | Aggregate evidence across encounters | Improved graph ranking, but strong signatures already saturate a two-identity ranking task; no proven rejection gain. |
| E18 | Open-set gallery and max-impostor policies, abstention | E24: 12/14 known queries correctly accepted, 0/4 unknowns accepted, 6/18 abstained for the conservative policy; **not** deployment calibration. |
| E22 | Self-radius normalization and count-equalized thinning | Some stress improvements, but reduced real-individual discrimination. |
| E23 | Partial assignment / transport for crops | Trimming or allowing partial matches also improves matches to impostors; rejected. |
| E24 | Add restrained global distance context to joint local descriptors | **Selected research candidate** `global_joint0.25`, AUC 0.941 and AP 0.911; see details below. |
| E25 | PCA/whitening, affine/pose normalization | No dominant improvement in impostor rejection and crop robustness. |
| E26 | Ablate radii, angular, global and nuisance features | Radii/global AUC 0.948, but weaker strongest-impostor rejection; angular evidence helps at hard negative tail. |
| E27 | Affine-invariant quadrilateral area motifs | AUC 0.616–0.633, insufficient distinctive evidence. |
| E28 | Angular phase and radius–angle association | Cosine 0.854, signed 0.773, bispectrum 0.910 AUC; more features did not improve rejection. |

The five resumed cycles tested **17 coherent experimental questions** beyond
Cycle 1, with 65 score definitions including Cycle 1 recorded in the complete
archive. These results include unsuccessful variants; do not selectively
represent the campaign as solely E24.

## Retained method and interpretation

The retained implementation is
`src/meros/sandbox/geometry/spot_signature.py`.

- **E05.local3.energy:** per spot, log-scaled distances to three nearest spots
  divided by median nearest-neighbor spacing, plus magnitudes of the first
  three angular Fourier moments; compare *joint feature distributions* with
  energy distance. No correspondence estimation or warp.
- **E24.global_joint0.25:** append, for each spot, distances at the 25th,
  50th and 75th percentiles to all other spots, normalized by the point-cloud
  RMS radius and weighted by **0.25**; compare joint distributions using
  the same energy distance. Larger scores (closer to zero) mean more similar
  distributions. These are **not calibrated identity probabilities**.

All numbers below are **exploratory, retrospectively compared on the same
historical corpus**:

| Method | ROC-AUC | Average precision |
| --- | ---: | ---: |
| Spot count | 0.683 | 0.436 |
| E04 original no-warp | 0.896 | 0.775 |
| Shape Context + TPS | 0.709 | 0.652 |
| Triangle matching + TPS | 0.799 | 0.721 |
| E05.local3.energy | 0.932 | 0.895 |
| **E24.global_joint0.25** | **0.941** | **0.911** |
| E26 radii/global without angular features | 0.948 | 0.903 |

E24 was selected for its **hard-negative rejection behavior**, not the maximum
global AUC. At the retrospective cutoff defined by the worst of the 107
observed impostor pairs, 33/46 genuine pairs scored higher. This cutoff was
chosen **after seeing all the scores** and is not a validated acceptance rule.

When restricting the pair comparison to only the two repeat individuals
(Rafael/Rogerio), E24 AUC = 0.904, versus E04 = 0.898 and E05 = 0.882.
The difference is much smaller than the aggregate AUC contrast. In the
two-enrolled-identity open-set demonstration, a conservative held-identity
policy accepted **12/14 known** correctly, **0/4 unknown** and abstained
on **6/18**; the less conservative fixed 95th-percentile policy accepted
13/14 known, 0/4 unknown and abstained on 5/18. These are not validated
population thresholds. "Abstain" can mean unknown identity **or insufficient
visible evidence**, not proof of a novel animal.

## Critical counterexamples and stopping decision

1. **Missing spots / visibility:** E24 scored below the actual genuine-pair
   median on 51/54 synthetic half-missing and 39/54 half-crop cases.
   This is sensitivity, not a biological false-nonmatch rate.
2. **Global arrangement is not uniquely encoded:** 46/54 rearranged-block
   synthetic controls scored above that median. Preserving local statistics
   while rearranging which regions contain them can fool the signature.
3. **Real impostor overlap:** Rafael/3 versus Rogerio/9 has E24 score
   -0.0263, while genuine Rogerio/11 versus Rogerio/9 has -0.0474.
   The unrelated pair looks *more similar* to E24 than the genuine one.
4. **Selection bias:** 17 experiments repeatedly reused the same two
   individuals with independent repeat images; the final AUC is not an
   independent validation of the selected method.

Further tuning of weights, neighborhood sizes, bins or decision cutoffs on
these same photos was stopped to avoid overfitting. The strongest supported
conclusion is **promising coordinate-only research evidence**, not reliable
automatic population identification.

## Minimal reproduction and archive access

```sh
make check
python -m meros.sandbox.geometry.spot_signature
```

The retained script displays real same-identity and different-identity examples
without consulting point-to-point correspondences or anatomy. The historical
commit contains the original full benchmarking scripts, variant definitions,
protocols, CSVs and JSON summaries for reproduction of the **archived**
AUC/open-set experiments. The slim branch does not attempt to regenerate all
old exploratory outputs; those remain tied to their historical source revision.

Do not commit generated figures, large score tables or disposable run artifacts
to the maintained code branch. If archival files are needed, check out the
historical commit in a separate worktree rather than copying old experimental
dependencies into the current codebase.
