# The Limits of Query Monitoring for Distributed Model Extraction

Code for the paper *The Limits of Query Monitoring for Distributed Model Extraction* (under double-blind review at TMLR).

A model provider faces a distillation campaign spread over many accounts, and it cannot link those accounts through network or payment metadata. What can it learn from the **queries alone**? This repository contains the full query-only monitor, every experiment behind the paper's numbers, the figure script, and a standalone tool that measures the quantity the paper finds decisive: how coordinated a provider's *benign* traffic is.

> **Every number in the paper is traceable to a line of code.** Each experiment script opens with a cell titled **"Paper numbers reproduced by this script"** (a table: paper location → value → where it is computed). Inline comments
> ```python
> # [PAPER] Table 3 (row "Half padding", col "Hollowed"): weekly paired detection 0.57
> ```
> sit directly above the code that computes each number. [`REPRODUCING.md`](REPRODUCING.md) collects all of them: an index by table and figure, the main-text numbers by section, and the full per-script map.

## Main results and where they come from

| Claim in the paper | Where in the paper | Script |
|---|---|---|
| Dataset-like campaigns over 50 volume-matched accounts are caught in 0.86–1.00 of untouched weeks at ≈2% false alarms; per-account monitors catch none | Table 1, Table 3, Fig. 1b | [`step12`](experiments/02_development_weeks/step12_realistic_deployment.py), [`step35`](experiments/04_frozen_holdout/step35_frozen_holdout.py) |
| Padding colluders with natural prompts defeats the first detector; hollowed selection still catches padded campaigns (0.50–0.57; 0.89 with E5; 0.96 in whole weeks) | §5, Table 3 | [`step23`](experiments/02_development_weeks/step23_hollowed_calibration_scale.py), [`step35`](experiments/04_frozen_holdout/step35_frozen_holdout.py) |
| Campaigns that evaded every detector kept at most 42% of a detectable campaign's capability gain | §6, Fig. 1a, Fig. 6 | [`step19`](experiments/03_capability_frontier/step19_nonsaturating_distillation.py), [`step29d`](experiments/03_capability_frontier/step29d_tilted_distillation.py), [`step32`](experiments/03_capability_frontier/step32_written_attacker.py) |
| Benign coordination (real/shuffled top eigenvalue of R_c) went from 1.14 to 2.22 between two periods of the same log | §5, §7 | [`step23`](experiments/02_development_weeks/step23_hollowed_calibration_scale.py), [`step35`](experiments/04_frozen_holdout/step35_frozen_holdout.py), [`census`](census/benign_coordination_census.py) |
| A slow ramp through the calibration weeks blinds every detector | §5, §7 | [`step21`](experiments/02_development_weeks/step21_sparse_boundary_adversary.py), [`step35`](experiments/04_frozen_holdout/step35_frozen_holdout.py) |
| No detector is both calibrated and powerful at colluder shares of 2% or below | §5, Fig. 5 | [`step20`](experiments/02_development_weeks/step20_baselines_adaptive_scaling.py), [`step27_31`](experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py), [`step32`](experiments/03_capability_frontier/step32_written_attacker.py) |
| The monitor detects topic shift, not extraction: benign GSM8K evaluation sweeps fire it 79–80% of weeks | §5 | [`step15`](experiments/02_development_weeks/step15_query_cluster_fair_baselines.py), [`step18`](experiments/02_development_weeks/step18_qc_repair_scale_evals.py) |
| Randomised throttling gives an exactly calibrated failover test, which benign teams with failover also fire (82–98%) | §8, App. K | [`step34`](experiments/05_randomized_interventions/step34_randomized_interventions.py) |

## Repository layout

```
.
├── README.md
├── REPRODUCING.md                  every paper number -> script -> line (generated from the annotations)
├── census/
│   └── benign_coordination_census.py   measure benign coordination in your own logs (§9, item 1)
├── experiments/                    one script per experiment "Step" (numbering follows docs/prediction_log.md)
│   ├── 01_controlled_studies/      Steps 1–8: synthetic worlds and real CIFAR-10 extraction (App. E, Fig. 3, Table 4)
│   ├── 02_development_weeks/       Steps 9–13, 15, 17, 18, 20–23, 27–31: WildChat-1M development weeks (§5, Tables 1, 5–11, Figs. 4–5)
│   ├── 03_capability_frontier/     Steps 14, 16, 19, 29d, 32: distillation into small students (§6, Fig. 6, Tables 12–14)
│   ├── 04_frozen_holdout/          Step 35: the single frozen evaluation on 17 untouched weeks (§7, Table 3, Fig. 1b)
│   ├── 05_randomized_interventions/ Step 34: randomised throttles, answer marks, dual reference (§8, App. K)
│   └── future_work_not_run/        Steps 36–37: implementations of open tests (§11). NOT RUN; no paper number uses them
├── figures/
│   └── make_figures.py             draws Figs. 1–6 from the recorded numbers; comments name each data source
├── docs/
│   ├── prediction_log.md           all 184 locked predictions with outcomes, each linked to its script
│   └── corrections.md              corrections made during the study
├── paper/                          LaTeX source of the submission and the compiled PDF
└── tools/
    └── sync_notebooks.py           regenerates the Kaggle .ipynb files from the annotated .py sources
```

Each experiment exists twice: `stepNN_name.py` is the canonical, reviewable source (a [jupytext](https://jupytext.readthedocs.io) "percent" script; `# %%` marks a notebook cell), and `stepNN_name.ipynb` is the same code as a notebook for Kaggle. Edit the `.py` and run `python tools/sync_notebooks.py`.

## Experiments

Steps are numbered in the order they were run. Predictions for every step were written into the notebook header before it ran; [`docs/prediction_log.md`](docs/prediction_log.md) lists them with their outcomes (74 of 184 failed), and [`docs/corrections.md`](docs/corrections.md) lists the errors we found and fixed along the way. Both are referenced from the paper's Reproducibility appendix rather than printed in it; in the scripts' annotations, "Prediction log (Step N …)" refers to `docs/prediction_log.md`. Step numbers 24–26 and 33 have no locked predictions and no number in the paper depends on them, so they are not part of this release.

| Step | Script | What it tests | Main paper items | Kaggle time |
|---|---|---|---|---|
| 1 | [`step01_missing_mode`](experiments/01_controlled_studies/step01_missing_mode.py) (+ CPU [`_toy`](experiments/01_controlled_studies/step01_missing_mode_toy.py)) | Toy-world validation of the spectral signal | Prediction log | short |
| 2 | [`step02_theory_scaling_zoo`](experiments/01_controlled_studies/step02_theory_scaling_zoo.py) | Analytic calibration, scaling law, attacker designs | App. E, Prediction log | 40–70 min |
| 3 | [`step03_highdim_rff`](experiments/01_controlled_studies/step03_highdim_rff.py) | Random Fourier modes in high dimension | Prediction log | 25–45 min |
| 4 | [`step04_linkage_price_law`](experiments/01_controlled_studies/step04_linkage_price_law.py) | A unified detectability law; spectral linkage | App. E, Prediction log | 30–60 min |
| 5 | [`step05_real_extraction_cifar10`](experiments/01_controlled_studies/step05_real_extraction_cifar10.py) | Real extraction attacks on a CIFAR-10 victim | App. E, Table 16 | 45–70 min |
| 6 | [`step06_baselines`](experiments/01_controlled_studies/step06_baselines.py) | Per-account, Higher-Criticism and global baselines | App. E | 45–70 min |
| 7 | [`step07_robustness`](experiments/01_controlled_studies/step07_robustness.py) | Correlated benign traffic, adaptive attackers | App. E | 60–80 min |
| 8 | [`step08_dose_response_taskaware`](experiments/01_controlled_studies/step08_dose_response_taskaware.py) | How much benign structure detection survives | Fig. 3, Table 4 | 45–60 min |
| 9 | [`step09_wildchat_llm`](experiments/02_development_weeks/step09_wildchat_llm.py) | First WildChat-1M test; calibration transfer (12% false alarms) | §5, App. D.1 | 30–50 min |
| 10 | [`step10_wildchat_drift_granularity`](experiments/02_development_weeks/step10_wildchat_drift_granularity.py) | Drift-aware calibration schemes | Fig. 4a, Table 5 | 40–60 min |
| 11 | [`step11_wildchat_dimension_robustness`](experiments/02_development_weeks/step11_wildchat_dimension_robustness.py) | Robustness to embedding dimension | App. F | 60–80 min |
| 12 | [`step12_realistic_deployment`](experiments/02_development_weeks/step12_realistic_deployment.py) | Volume-matched colluders, whole weekly populations, all baselines | **Table 1**, Fig. 4b | 60–90 min |
| 13 | [`step13_twoscale_mechanism_scale`](experiments/02_development_weeks/step13_twoscale_mechanism_scale.py) | Why the two-scale test works; precision; sharding | §5, Table 7 | 50–70 min |
| 14 | [`step14_mechanism_distillation`](experiments/03_capability_frontier/step14_mechanism_distillation.py) | Follow-up mimicry; what detected vs evading campaigns extract | §5, Table 12 | 90–120 min |
| 15 | [`step15_query_cluster_fair_baselines`](experiments/02_development_weeks/step15_query_cluster_fair_baselines.py) | Oracle-matched comparison; human-written GSM8K; mechanism checks | §5, Table 6 | 70–100 min |
| 16 | [`step16_price_of_stealth`](experiments/03_capability_frontier/step16_price_of_stealth.py) | Elicitation regime (Qwen2.5-0.5B) | §6, Fig. 6b, Table 13 | 1.5–2 h |
| 17 | [`step17_calibrated_query_cluster`](experiments/02_development_weeks/step17_calibrated_query_cluster.py) | Calibrated query-cluster detector under mimicry | App. F | 60–80 min |
| 18 | [`step18_qc_repair_scale_evals`](experiments/02_development_weeks/step18_qc_repair_scale_evals.py) | Calibration repairs; scale; benign evaluation sweeps | §5, App. D.5 | 70–90 min |
| 19 | [`step19_nonsaturating_distillation`](experiments/03_capability_frontier/step19_nonsaturating_distillation.py) | Extraction regime (SmolLM2-360M); price of stealth | **§6**, Fig. 6a, Table 14, Table 2 gains | 3–4 h |
| 20 | [`step20_baselines_adaptive_scaling`](experiments/02_development_weeks/step20_baselines_adaptive_scaling.py) | Source-shift baselines, adaptive attacker, scale to 10⁴ | §5, Fig. 5, Table 10 | 60–80 min |
| 21 | [`step21_sparse_boundary_adversary`](experiments/02_development_weeks/step21_sparse_boundary_adversary.py) | Sacrificial scheduling, calibration ramp, low-FPR ROC | §5, App. D.6 | 80–100 min |
| 22 | [`step22_hollowed_profile_sequential`](experiments/02_development_weeks/step22_hollowed_profile_sequential.py) | Hollowed selection; sequential detection | Table 8, Table 2 | 80–110 min |
| 23 | [`step23_hollowed_calibration_scale`](experiments/02_development_weeks/step23_hollowed_calibration_scale.py) | Centred hollowed matrix R_c; conformal calibration | **§5**, Table 9 | 60–90 min |
| 27–31 | [`step27_31_bulk_edge_multioutlier_tilted`](experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | Multi-outlier selection; least-favourable (tilted) campaign | Table 11, Fig. 5, §4 (p = 1.48%) | 60–100 min |
| 29d | [`step29d_tilted_distillation`](experiments/03_capability_frontier/step29d_tilted_distillation.py) | Does the tilted campaign distil? | §6, Fig. 6a, Table 14 | 2–2.5 h |
| 32 | [`step32_written_attacker`](experiments/03_capability_frontier/step32_written_attacker.py) | Prompts written in users' voice; account types; linkage expansion | §6, Cor. 1, Fig. 1a | 4.5–5 h |
| 34 | [`step34_randomized_interventions`](experiments/05_randomized_interventions/step34_randomized_interventions.py) | Randomised throttles (Prop. 5), answer marks, dual reference | §8, App. K | 60–90 min |
| 35 | [`step35_frozen_holdout`](experiments/04_frozen_holdout/step35_frozen_holdout.py) | **Frozen pipeline on 17 untouched weeks**, second encoder, whole weeks | **§7, Table 3**, Fig. 1b, Table 2 | 5–6 h |
| 36, 37 | [`future_work_not_run/`](experiments/future_work_not_run) | Open tests from §11 (sequential hollowed test, relays, live teacher) | none (not run) | — |

## Running the experiments

The experiments were run as Kaggle notebooks on one NVIDIA T4 or P100 GPU.

1. **Kaggle (recommended).** Upload `stepNN_*.ipynb`, set *Accelerator → GPU* and *Internet → On*. For the WildChat steps (9 onwards), add a Hugging Face token with access to the gated WildChat datasets as a Kaggle secret named `HF_TOKEN` (*Add-ons → Secrets*). Run once with `QUICK = True` in the configuration cell (minutes, tiny sizes) and then with `QUICK = False` for the paper's numbers. Each script prints a scorecard of its locked predictions at the end.
2. **Elsewhere.** Install the requirements (`pip install -r requirements.txt`) and run a script with `python experiments/.../stepNN_*.py` or open it as a notebook through jupytext. The scripts read `HF_TOKEN` from the environment when Kaggle secrets are unavailable, and write caches and outputs to `/kaggle/working` if it exists, otherwise to the current directory.

**Caches.** Later steps reuse embedding caches written by earlier ones (for example `step11_wildchat_E384_full.npz`, `step5_cache_full.npz`). When a cache is missing, each script rebuilds it, which adds roughly 30 minutes the first time. Steps 6–7 reuse Step 5's victim (Step 8 builds its own); Steps 12 onwards reuse Step 11's WildChat embeddings; Step 35 builds its own caches from WildChat-4.8M.

**Seeds and run-to-run variation.** Seeds are fixed in each configuration cell. GPU nondeterminism and dataset streaming order still move weekly detection rates between runs; the paper reports the spread (up to 0.14 across four development-week runs of one cell, at most 0.07 across campaign seeds on untouched weeks). Compare re-runs against the paper within that spread.

### Data and models

All data are public; WildChat requires accepting its license on Hugging Face.

| Used for | Resource |
|---|---|
| Benign traffic, development weeks | `allenai/WildChat-1M` (ODC-BY license; accounts are hashed IPs) |
| Benign traffic, untouched weeks | `allenai/WildChat-4.8M` (ODC-BY license; 6 May – 1 Sep 2024; 31 earlier weeks as calibration history) |
| Campaign prompts | MetaMathQA, Magicoder-OSS-Instruct, Alpaca, GSM8K (train), MMLU |
| Capability evaluation | GSM8K (test), MATH-500 |
| Controlled studies | CIFAR-10 |
| Query embeddings | `sentence-transformers/all-MiniLM-L6-v2`, `intfloat/e5-small-v2` |
| Students / rewriter | SmolLM2-360M, Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-1.5B-Instruct |

Local caches contain WildChat user text: keep them out of version control (they are in `.gitignore`) and do not redistribute them. No result depends on identifying a user.

## Measuring benign coordination in your own traffic

```bash
python census/benign_coordination_census.py traffic.npz --min-queries 10 --out census.json
```

`traffic.npz` holds `emb` (query embeddings), `account` (account id per query), `window` (e.g. ISO week) and, optionally, `task` (score that a query is in a valuable task region). For each window the script reports the real-to-shuffled top-eigenvalue ratio of the centred hollowed matrix, the number of outlier eigenvectors, per-account topicality and the share of on-task accounts. These are the quantities that decide what any query-only monitor can see (§9, item 1). The script is a compact re-implementation; the paper's own values (1.14, 2.22, 3.06) come from Steps 23 and 35.

## Figures

```bash
python figures/make_figures.py      # writes figures/out/*.pdf (Figs. 1–6)
```

The figure script plots the numbers recorded from the experiments rather than re-running them; a comment above each data array names the script that produces it.

## Numbers in the paper that no script computes

- Counts from public reports of real campaigns in §1 and §8 (e.g. 24,000 accounts, more than 3,500 accounts, 5,380 accounts) are quoted from the cited reports.
- Analytic values derived in the text from measured inputs: the ≈51 extra on-task queries of the square-root law (from p = 1.48%, Step 27–31) and K* ≈ 14 posing accounts (from q = 0.25%, Step 32), and the 16–42 legitimate accounts throttled per detection (App. K, from Step 32's precision).
- Composite values that join two scripts, such as Table 2's gains at the week of first detection (Steps 19 + 22/23/27–31/35) and the covertness cost of 58% or more (Steps 19, 29d, 32). The comments in both scripts say so.

## Responsible use

The experiments include constructions that evade the monitor: padding colluder accounts with natural prompts, ramping a campaign through the calibration weeks, sacrificial scheduling, and prompts rewritten in users' voice. They are included because a monitor cannot be evaluated without them, and each follows directly from the monitor's design or from long-known attack ideas. Please use them to evaluate and improve defences.

## Citation

```bibtex
@article{anonymous2026limits,
  title   = {The Limits of Query Monitoring for Distributed Model Extraction},
  author  = {Anonymous},
  journal = {Under review at Transactions on Machine Learning Research},
  year    = {2026}
}
```

## License

Code: MIT (see [`LICENSE`](LICENSE)). The datasets and models keep their own licenses.
