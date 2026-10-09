# Reproducing the paper: which code produces which number

This file maps every table, figure and quoted number in *The Limits of Query Monitoring against Distributed Model Extraction* to the script that computes it. The same information sits inside each script: a header cell titled **"Paper numbers reproduced by this script"** and inline comments of the form

```python
# [PAPER] Table 3 (row "Half padding", col "Hollowed"): weekly paired detection 0.57
```

directly above the line that computes or prints the number. Search a script for `[PAPER]` to jump to them.

**How to read the values.** Numbers come from single Kaggle runs with the fixed seeds in each configuration cell. Re-runs differ by run-to-run variation the paper reports explicitly (weekly detection varied by up to 0.14 across four runs of one development-week cell, and by at most 0.07 across campaign seeds on untouched weeks), so expect agreement to within that spread, not to the last digit. Where a paper number combines two scripts (for example, Table 2 joins detection times with distillation gains), both scripts are listed and the comment says so. Entries marked *plausibly* or *not isolated* are ones where the code that produces the value could not be pinned to a single line.

"Prediction log (Step N …)" in a paper-location column refers to [`docs/prediction_log.md`](docs/prediction_log.md), which accompanies the paper (its Reproducibility appendix, App. P, gives the totals); corrections are in [`docs/corrections.md`](docs/corrections.md).

Figures are drawn by [`figures/make_figures.py`](figures/make_figures.py) from the recorded numbers; its comments name the experiment behind each data array. Fig. 2 is a diagram.

## 1. Index: tables and figures

| Paper item | What it shows | Produced by |
|---|---|---|
| Fig. 1a | Capability vs hollowed detection (§1) | Step 19: [`step19_nonsaturating_distillation.py`](experiments/03_capability_frontier/step19_nonsaturating_distillation.py)<br>Step 29d: [`step29d_tilted_distillation.py`](experiments/03_capability_frontier/step29d_tilted_distillation.py)<br>Step 32: [`step32_written_attacker.py`](experiments/03_capability_frontier/step32_written_attacker.py) |
| Fig. 1b | Frozen detectors on untouched weeks (§1) | Step 35: [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) |
| Fig. 3 | Barrier in a controlled setting (App. B) | Step 8: [`step08_dose_response_taskaware.py`](experiments/01_controlled_studies/step08_dose_response_taskaware.py) |
| Fig. 4a | Calibration schemes (App. D.1) | Step 10: [`step10_wildchat_drift_granularity.py`](experiments/02_development_weeks/step10_wildchat_drift_granularity.py) |
| Fig. 4b | Per-detector false alarms (App. D.1) | Step 12: [`step12_realistic_deployment.py`](experiments/02_development_weeks/step12_realistic_deployment.py) |
| Fig. 5 | Scale / selection gap (App. D.5) | Step 20: [`step20_baselines_adaptive_scaling.py`](experiments/02_development_weeks/step20_baselines_adaptive_scaling.py)<br>Step 27–31: [`step27_31_bulk_edge_multioutlier_tilted.py`](experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) |
| Fig. 6a | Extraction regime (App. G) | Step 19: [`step19_nonsaturating_distillation.py`](experiments/03_capability_frontier/step19_nonsaturating_distillation.py)<br>Step 29d: [`step29d_tilted_distillation.py`](experiments/03_capability_frontier/step29d_tilted_distillation.py) |
| Fig. 6b | Elicitation regime (App. G) | Step 16: [`step16_price_of_stealth.py`](experiments/03_capability_frontier/step16_price_of_stealth.py) |
| Table 1 | Weekly detection at deployed thresholds, development weeks (§5) | Step 12: [`step12_realistic_deployment.py`](experiments/02_development_weeks/step12_realistic_deployment.py) |
| Table 2 | Capability held at first detection (§6). Detection weeks: Steps 22, 23, 27–31 (development) and 35 (untouched); gains in parentheses: Step 19 curves | Step 14: [`step14_mechanism_distillation.py`](experiments/03_capability_frontier/step14_mechanism_distillation.py)<br>Step 19: [`step19_nonsaturating_distillation.py`](experiments/03_capability_frontier/step19_nonsaturating_distillation.py)<br>Step 22: [`step22_hollowed_profile_sequential.py`](experiments/02_development_weeks/step22_hollowed_profile_sequential.py)<br>Step 23: [`step23_hollowed_calibration_scale.py`](experiments/02_development_weeks/step23_hollowed_calibration_scale.py)<br>Step 27–31: [`step27_31_bulk_edge_multioutlier_tilted.py`](experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py)<br>Step 35: [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) |
| Table 3 | Frozen pipeline on 17 untouched weeks (§7). Every cell: Step 35; the other scripts supply only the development-week values quoted in the caption | Step 14: [`step14_mechanism_distillation.py`](experiments/03_capability_frontier/step14_mechanism_distillation.py)<br>Step 23: [`step23_hollowed_calibration_scale.py`](experiments/02_development_weeks/step23_hollowed_calibration_scale.py)<br>Step 27–31: [`step27_31_bulk_edge_multioutlier_tilted.py`](experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py)<br>Step 35: [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) |
| Table 4 | Controlled dose-response (App. E) | Step 8: [`step08_dose_response_taskaware.py`](experiments/01_controlled_studies/step08_dose_response_taskaware.py) |
| Table 5 | Small monitored populations (App. F) | Step 10: [`step10_wildchat_drift_granularity.py`](experiments/02_development_weeks/step10_wildchat_drift_granularity.py) |
| Table 6 | Oracle-matched comparison (App. F) | Step 15: [`step15_query_cluster_fair_baselines.py`](experiments/02_development_weeks/step15_query_cluster_fair_baselines.py)<br>Step 18: [`step18_qc_repair_scale_evals.py`](experiments/02_development_weeks/step18_qc_repair_scale_evals.py) |
| Table 7 | Mechanism probes (App. F) | Step 13: [`step13_twoscale_mechanism_scale.py`](experiments/02_development_weeks/step13_twoscale_mechanism_scale.py) |
| Table 8 | Selection on hollowed R vs C (App. F) | Step 22: [`step22_hollowed_profile_sequential.py`](experiments/02_development_weeks/step22_hollowed_profile_sequential.py) |
| Table 9 | Selection on centred hollowed R_c vs C (App. F) | Step 23: [`step23_hollowed_calibration_scale.py`](experiments/02_development_weeks/step23_hollowed_calibration_scale.py) |
| Table 10 | Scale in pooled populations (App. F) | Step 20: [`step20_baselines_adaptive_scaling.py`](experiments/02_development_weeks/step20_baselines_adaptive_scaling.py) |
| Table 11 | Multi-outlier selection at scale (App. F) | Step 27–31: [`step27_31_bulk_edge_multioutlier_tilted.py`](experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py)<br>Step 32: [`step32_written_attacker.py`](experiments/03_capability_frontier/step32_written_attacker.py) |
| Table 12 | Detection vs what the campaign extracts (App. H) | Step 14: [`step14_mechanism_distillation.py`](experiments/03_capability_frontier/step14_mechanism_distillation.py) |
| Table 13 | Elicitation regime gains (App. H) | Step 16: [`step16_price_of_stealth.py`](experiments/03_capability_frontier/step16_price_of_stealth.py) |
| Table 14 | Extraction regime gains (App. H) | Step 19: [`step19_nonsaturating_distillation.py`](experiments/03_capability_frontier/step19_nonsaturating_distillation.py)<br>Step 29d: [`step29d_tilted_distillation.py`](experiments/03_capability_frontier/step29d_tilted_distillation.py) |
| Table 15 | What works where (App. L); collects rows from Tables 2, 3, 9 and 11 | Step 14: [`step14_mechanism_distillation.py`](experiments/03_capability_frontier/step14_mechanism_distillation.py)<br>Step 23: [`step23_hollowed_calibration_scale.py`](experiments/02_development_weeks/step23_hollowed_calibration_scale.py)<br>Step 27–31: [`step27_31_bulk_edge_multioutlier_tilted.py`](experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py)<br>Step 35: [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) |
| Table 16 | Main settings (App. O) | Step 5: [`step05_real_extraction_cifar10.py`](experiments/01_controlled_studies/step05_real_extraction_cifar10.py)<br>Step 6: [`step06_baselines.py`](experiments/01_controlled_studies/step06_baselines.py)<br>Step 9: [`step09_wildchat_llm.py`](experiments/02_development_weeks/step09_wildchat_llm.py)<br>Step 10: [`step10_wildchat_drift_granularity.py`](experiments/02_development_weeks/step10_wildchat_drift_granularity.py)<br>Step 11: [`step11_wildchat_dimension_robustness.py`](experiments/02_development_weeks/step11_wildchat_dimension_robustness.py)<br>Step 12: [`step12_realistic_deployment.py`](experiments/02_development_weeks/step12_realistic_deployment.py)<br>Step 19: [`step19_nonsaturating_distillation.py`](experiments/03_capability_frontier/step19_nonsaturating_distillation.py) |

## 2. Numbers quoted in the main text, by section

Rows from the full map whose location includes a main-text section. A row can appear under several sections.

### §1 Introduction

| Paper location | Quantity | Value | Script — where |
|---|---|---|---|
| Abstract ('0.86–1.00 of weeks'); §1 item 1 ('86–100% of weeks'); §7 and App. I ('0.86–1.00') | dataset-campaign detection range over both encoders | 0.86–1.00 | [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) — Cell 8 `RES_M` and Cell 13 `RES_E`, source metamath_unique |
| Abstract ('0.50–0.96 of weeks'); §1 item 2 ('50–96% of untouched weeks') | hollowed detection of padded campaigns, over seeds, E5 and whole weeks | 0.50–0.96 | [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) — Cell 8 `RES_M`/`RUNS`, Cell 13 `RES_E`, Cell 12 `RES_L` (metamath_mixed, tsRc) |

### §2 The setting

| Paper location | Quantity | Value | Script — where |
|---|---|---|---|
| §2 "Data" | development data: hashed IPs and weeks of WildChat-1M | 98,331 hashed IPs over 56 weeks | [`step09_wildchat_llm.py`](experiments/02_development_weeks/step09_wildchat_llm.py) — plausibly the account/week counts printed at the end of Cell 2 (same loading code in Steps 9–12); not isolated |
| §2 "Data"; App. Table 16 (Embedding, Fourier modes, Whitening floor) | attacker pool share; MiniLM → PCA rescaled to total variance 4; m = 400 complex modes, h = 4, floor 10−4 | 20%; "rescaled to total variance 4"; "m = 400"; "h = 4"; "10−4" | [`step09_wildchat_llm.py`](experiments/02_development_weeks/step09_wildchat_llm.py) — Cell 1 configuration (`POOL_FRAC`, `D_DEF`, `H_BW`, `M_MODES`, `FLOOR`) and the PCA step in Cell 2 |
| §2 "Data" | monitored accounts have at least 10 queries in the week; windows hold 64–364 accounts | as quoted | [`step12_realistic_deployment.py`](experiments/02_development_weeks/step12_realistic_deployment.py) — Cell 1 `MIN_PER_ACCT`; window sizes plausibly the "accounts per test-week window" print in Cell 10 (not isolated) |
| §2 'Data'; §7; App. I 'Design' | untouched data: test weeks; history weeks; English user turns; hashed IPs | 17 weeks; 31 weeks; 757,000; 76,000 | [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) — Cell 2 print (turns, hashed IPs, history/holdout weeks); Cell 8 print "[matched] holdout test weeks" |

### §4 Theory

| Paper location | Quantity | Value | Script — where |
|---|---|---|---|
| §4 (discussion after Prop. 4); App. B 'The least-favourable campaign'; Prediction log (Step 29, "Task region rare: p ≤ 5%") | benign share of the task region | p = 1.48%; Pass (1.48%) | [`step27_31_bulk_edge_multioutlier_tilted.py`](experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) — Cell 9, `P_TASK` ("[29a]" print; "P29a"); the "about 51" extra queries over twelve weeks of 1,640 is derived from p analytically in the paper, not computed here |
| Cor. 1 (§4); App. B 'The least-favourable campaign'; Prediction log (Step 32, "Majority-on-task accounts rare") | benign share of majority-on-task accounts q | "q = 0.25%"; "Pass (0.25%)" (the paper derives "about 14 posing accounts" at A = 10^5 from it; K* is not computed here) | [`step32_written_attacker.py`](experiments/03_capability_frontier/step32_written_attacker.py) — Cell W, `Q_TYPE`; print "[types]"; Cell Z "P32m" |

### §5 Development weeks

| Paper location | Quantity | Value | Script — where |
|---|---|---|---|
| §5 "Calibration breaks first"; App. D.1; Prediction log (Step 9, "Fixed calibration transfers to later weeks") | false alarms on later (held-out) null weeks of thresholds set once on early weeks, nominal 1% | 12% (Prediction log: "Fail (12%)") | [`step09_wildchat_llm.py`](experiments/02_development_weeks/step09_wildchat_llm.py) — Part A cell, `row["suite"]` in print "held-out-week FPR ... suite="; Part C print "P9.1" |
| Table 1, last row "Realised false alarms"; Fig. 4b; §5 "Calibration breaks first"; App. D.1; Prediction log (Step 12, "Every detector's false alarms ≤ 2%") | held-out false alarms at nominal 1%, rolling calibration, week-clustered intervals | Two-scale 0.7%, Global 4.1%, MMD 11.6%, Spectral 7.8%, Per-account / HC / Fraudar 5.3 / 7.5 / 2.9%; volume test 14.2% (§5: "2.9–14.2%"; D.1: "4.1–7.8%"); Prediction log: Fail (only two-scale) | [`step12_realistic_deployment.py`](experiments/02_development_weeks/step12_realistic_deployment.py) — Cell 7 print "held-out FPR" (`FPR[d]` with `boot_ci`); verdict "[FPR]" lines; Cell 10 print "P12.7" |
| Table 1 caption; §5; App. D | K = 50 volume-matched colluders, 24 development weeks; "about 1,640 queries a week" | as quoted | [`step12_realistic_deployment.py`](experiments/02_development_weeks/step12_realistic_deployment.py) — Cell 1 (`KS`, `N_TEST_W`); 1,640 plausibly the `N~` printed for K = 50 in Cell 8 (protocol value reused by later Steps); not isolated |
| §5 'Coordinated, dataset-like campaigns are catchable'; App. D.2 | two-scale flagged-group precision (share of the group that are colluders) | 0.94–1.00 (§5); 1.00 at K = 50, 0.94–0.96 at K = 10 (App. D.2) | [`step13_twoscale_mechanism_scale.py`](experiments/02_development_weeks/step13_twoscale_mechanism_scale.py) — Cell 6 `run_block`, `prec` list -> `ts_precision`; printed "two-scale group precision" (weekly, `metamath_unique`/`metamath_rephrase`, K = 10/50) and verdict `ts_prec=` |
| §5 'Coordinated, dataset-like campaigns are catchable'; App. D.2 | two-scale flagged-group recall (share of colluders in the group) | 12–31% | [`step13_twoscale_mechanism_scale.py`](experiments/02_development_weeks/step13_twoscale_mechanism_scale.py) — Cell 6 `run_block`, `rec` list -> `ts_recall`; printed "recall" and verdict `ts_rec=` (range plausibly over weekly sources and K) |
| §5 "Padding, and a fix that looks at directions"; App. D.4 | two-scale detection cut by padding / follow-up mimicry | from 0.85 to 0.21–0.24 | [`step14_mechanism_distillation.py`](experiments/03_capability_frontier/step14_mechanism_distillation.py) — Cell 9 (`metamath_unique` vs `metamath_pad50`, `metamath_followup`) |
| §5 'Coordinated, dataset-like campaigns are catchable'; App. D.2 'Its edge is calibration, not ranking'; Table 6 row "MetaMathQA (generated)"; Prediction log (Step 15, P15.5) | oracle-matched 1% TPR, two-scale vs global stream vs MMD (weekly, K = 50) | 0.80 vs 0.70 vs 0.58; Prediction log "Partial (0.80 vs 0.58; group MMD fails)" | [`step15_query_cluster_fair_baselines.py`](experiments/02_development_weeks/step15_query_cluster_fair_baselines.py) — Cell 8 `roc(null_v[d], att_v[d])` -> `row[d + "_roc"]["tpr1"]`; Cell 12 print "P15.5" and verdict "ROC:" lines (`twoscale`, `glob`, `mmd`) |
| §5 'Coordinated, dataset-like campaigns are catchable'; App. D.2 | AUC, global stream vs two-scale | 0.97 against 0.88 | [`step15_query_cluster_fair_baselines.py`](experiments/02_development_weeks/step15_query_cluster_fair_baselines.py) — Cell 8 `roc(...)["auc"]`; Cell 12 verdict "ROC:" line (`glob`, `twoscale`) |
| §5 'It detects topic shift, not extraction'; App. D.3; Prediction log (Step 15, P15.4) | weekly paired two-scale detection, human GSM8K vs MetaMathQA (K = 50) | 0.94 vs 0.79; Prediction log "Pass (0.94 vs 0.79)" | [`step15_query_cluster_fair_baselines.py`](experiments/02_development_weeks/step15_query_cluster_fair_baselines.py) — Cell 8 rows `gsm8k_human` / `metamath_unique` (`twoscale`); Cell 12 print "P15.4" |
| Prediction log (Step 17, P17.1); App. F 'Query-level pooling' ("robust per-cell calibration fired on 28% of null weeks"); App. D.4 and §5 'Padding, and a fix that looks at directions' (first of the "four repairs that pool topic cells across accounts") | corrected query-cluster (QC) held-out false alarms, weekly | Fail (28%) | [`step17_calibrated_query_cluster.py`](experiments/02_development_weeks/step17_calibrated_query_cluster.py) — Cell 8 `FPR["qc"]` ("held-out FPR" print); Cell 12 print "P17.1" |
| §5 'Scale: selection fails before the barrier does' | colluder shares at which no detector was both calibrated and powerful | 1–2.5% | [`step17_calibrated_query_cluster.py`](experiments/02_development_weeks/step17_calibrated_query_cluster.py) — Cell 10 rolling populations, K = 10 and 25 of A = 1,000 (plausibly combined with Step 15 Part P; summary statement, not isolated) |
| §5 'It detects topic shift, not extraction'; App. D.3 | benign groups sweeping the GSM8K test set from 3 or 10 accounts: two-scale alarm rate | 79–80% of weeks | [`step18_qc_repair_scale_evals.py`](experiments/02_development_weeks/step18_qc_repair_scale_evals.py) — Part Q/E weekly loop (Cell 8), rows `eval_gsm8k(3x300)` / `eval_gsm8k(10x100)`; print "P18.6 benign pipeline" |
| §5 'Attacks on the monitor'; App. D.6 'Optimising against the statistic' | weekly two-scale detection, prompts ordered greedily against the statistic, no padding | from 0.80 to 0.31 | [`step20_baselines_adaptive_scaling.py`](experiments/02_development_weeks/step20_baselines_adaptive_scaling.py) — Part B/D weekly loop (Cell 8): 0.31 = deployed two-scale on `metamath_adaptive` (print "P20.4"); 0.80 plausibly the `metamath_unique` row of the same run |
| §5 'It detects topic shift, not extraction'; App. D.3 | dataset-versus-chat classifier and topic-concentration statistic on the selected group | ranked campaigns moderately (AUC 0.69–0.84) but caught none at a 1% false-alarm rate | [`step20_baselines_adaptive_scaling.py`](experiments/02_development_weeks/step20_baselines_adaptive_scaling.py) — Part B/D weekly loop (Cell 8), `style_grp` / `topic_conc` `_roc` (auc, tpr1); verdict "oracle" lines; print "P20.1" |
| Fig. 5 series "top eigenvector only"; §5 'Scale: selection fails before the barrier does'; App. D.5 'Up to 10^4 accounts' | two-scale power as colluder share goes from 20% to 0.5% | fell from 1.00 to 0.00 | [`step20_baselines_adaptive_scaling.py`](experiments/02_development_weeks/step20_baselines_adaptive_scaling.py) — Part L (Cell 10), `LAW` field `tpr` / `EVT` field `tpr_selected` |
| §5 'Attacks on the monitor'; App. D.6 'Choosing who gets flagged' | sacrificial scheduling (30% most distinctive queries on five accounts): detection on C; other colluders in the flagged group | 0.92; none of the other 45 colluders | [`step21_sparse_boundary_adversary.py`](experiments/02_development_weeks/step21_sparse_boundary_adversary.py) — Part X, `SCHED["sacrificial"]` (`paired`, `recall_nonsac`); print "[X] sacrificial" |
| §5 'Attacks on the monitor'; App. D.6 'Teaching the monitor that the campaign is normal' | campaign ramped through the eight calibration weeks vs abrupt: deployed two-scale detection | from 0.95 to 0.12 | [`step21_sparse_boundary_adversary.py`](experiments/02_development_weeks/step21_sparse_boundary_adversary.py) — Part T, `TEMP["abrupt"]` / `TEMP["ramped"]`; print "[T] deployed two-scale paired detection" |
| §5 'Padding, and a fix that looks at directions'; App. D.4; Prediction log (Step 23, "Common-mode removal: real/shuffled λmax (Rc) ≥ 1") | top eigenvalue of Rc, real / shuffled weeks | 1.14 (Prediction log: Pass (1.14)) | [`step23_hollowed_calibration_scale.py`](experiments/02_development_weeks/step23_hollowed_calibration_scale.py) — Cell 9, `RATIO` ("[C1]" print); print "P23.1" |
| §5 'Padding, and a fix that looks at directions'; App. D.4; App. Table 9 caption; Prediction log (Step 23, "Conformal hollowed selection: false alarms ≤ 2%") | held-out false alarms, conformal thresholds (Rc and C) | no false alarm in 960 windows; Pass (0/960 windows; week-level bound 12%) | [`step23_hollowed_calibration_scale.py`](experiments/02_development_weeks/step23_hollowed_calibration_scale.py) — Cell 9, `FPR[("conf","tsRc")]`, `FPR[("conf","tsC")]`; print "P23.2" |
| §5 'Padding, and a fix that looks at directions'; App. D.4; Table 3 caption (development-week hollowed values); Prediction log (Step 23, "Calibrated hollowed selection ≥ 0.6 under padding": Fail (0.48)) | hollowed vs C selection on padded campaigns; hollowed on dataset | 0.46–0.48 against 0.00–0.02; dataset 1.00 | [`step23_hollowed_calibration_scale.py`](experiments/02_development_weeks/step23_hollowed_calibration_scale.py) — Cell 9, `"conf/tsRc"` / `"conf/tsC"` for metamath_mixed, metamath_followup, metamath_unique; print "P23.3" |
| §5 'Scale: selection fails before the barrier does'; App. D.5 | multi-outlier selection (Tmax), oracle thresholds | 93–100% of unpadded campaigns up to 10^4 accounts; padded 77–100% up to 5,000 accounts and 17% at 10^4 | [`step27_31_bulk_edge_multioutlier_tilted.py`](experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) — Cell 12, `row["Tmax"]` |
| §5 'Scale: selection fails before the barrier does'; App. D.5 'Why this is not yet a production result'; Prediction log (Step 32, "Multi-outlier deployable at 10^4") | best deployed calibration of multi-outlier selection | "5% false alarms, not 1%"; "at 5,000–7,552 account-weeks"; "Fail (5% false alarms at 7,552)" | [`step32_written_attacker.py`](experiments/03_capability_frontier/step32_written_attacker.py) — Cell P, `DEP["fpr"]["Tmax"]` (A = 10,000 capped to 0.8 × test account-weeks); print "[P-dep]"; Cell Z "P32i/n" |
| §5 'Scale: selection fails before the barrier does'; App. D.5 'For providers'; App. L item 6; Prediction log (Step 32, "Linked 30%: expansion ≥ 0.8 …") | seeded hollowed expansion from 30% linked colluders (15 of 50, +50 accounts) at 2,500–10,000 | "recovered 96–100% of the rest"; "Partial (1.00, 0.96; metadata 0.67)" | [`step32_written_attacker.py`](experiments/03_capability_frontier/step32_written_attacker.py) — Cell P, `seeded(...)` → `link_seeded`, `link_recall`, `link_meta_only`; print "[P] … linked colluders"; Cell Z "P32o" |

### §6 Capability–detectability frontier

| Paper location | Quantity | Value | Script — where |
|---|---|---|---|
| §6 "Two regimes"; App. G.1; Fig. 6b (elicitation, Qwen2.5-0.5B, three seeds) | gain from 100 math prompts, and no more from 900; padding costs nothing measurable | 9.8 points | [`step16_price_of_stealth.py`](experiments/03_capability_frontier/step16_price_of_stealth.py) — Cell 6 (`G`); figure in Cell 8 |
| App. Table 14 caption; §6 "Two regimes"; App. G.1; Fig. 6a caption; Prediction log (Step 19), "SmolLM2-360M lacks the skill" | base SmolLM2-360M GSM8K accuracy (first 400 problems); P19.1 | 3.5%; Pass (3.5%) | [`step19_nonsaturating_distillation.py`](experiments/03_capability_frontier/step19_nonsaturating_distillation.py) — Cell 7, `BASE_A`; Cell 9 "P19.1" print |
| App. Table 14, row "SmolLM2-360M, GSM8K", col "All math"; §6 "Two regimes"; App. G.1; Fig. 6a | gain at 1,000 / 3,000 / 10,000 queries | +7.5 ± 0.0 / +10.4 ± 0.1 / +15.9 ± 0.9 (text: 7.5, 10.4 and 15.9 points) | [`step19_nonsaturating_distillation.py`](experiments/03_capability_frontier/step19_nonsaturating_distillation.py) — Cell 7, summary print "[A] n=...: unique" |
| App. Table 14, col "Paired cost"; §6 "Two regimes"; App. G.1 | cost of padding half of every campaign (and share of the gain) | −3.6 / −2.8 / −3.8 (text: 3.6, 2.8 and 3.8 points; 48%, 27% and 24% of the gain) | [`step19_nonsaturating_distillation.py`](experiments/03_capability_frontier/step19_nonsaturating_distillation.py) — Cell 7, "paired mixed−unique"; the percentages are cost / all-math gain (not printed) |
| §6 "The covert end"; App. G.2 | dataset-campaign gain at 3,000 queries used as the reference (tilted 2.2 against 10.4) | 10.4 | [`step19_nonsaturating_distillation.py`](experiments/03_capability_frontier/step19_nonsaturating_distillation.py) — Cell 7, unique n = 3000 (the tilted 2.2 comes from Step 29d) |
| §6 "Two regimes"; App. G.1 | Qwen2.5-1.5B on MATH-500 | flat from 1,000 to 4,000 queries | [`step19_nonsaturating_distillation.py`](experiments/03_capability_frontier/step19_nonsaturating_distillation.py) — Cell 8 (Part B) |
| Table 2 (row "Any / Least detectable", development weeks); App. G.2 ("at most 0.01 on development weeks"); App. Table 15 (development weeks, "Least detectable": Two-scale / Hollowed / Multi-outlier); §6 'What the attacker gets' ('The covert end': evaded every detector) | weekly detection of the least-detectable campaign | ≈0.01; 0.00 / 0.01 / 0.01 | [`step27_31_bulk_edge_multioutlier_tilted.py`](experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) — Cell 10, `RES` row tilted_c05 (`tsC`, `tsRc`, `Tmax`) |
| §6 "The covert end"; App. G.2; Fig. 1a and Fig. 6a ("least detectable (tilted)") | gain at 3,000 queries, against 10.4; covertness cost | 2.2 points; 79% of the gain | [`step29d_tilted_distillation.py`](experiments/03_capability_frontier/step29d_tilted_distillation.py) — Cell 7, tilted_c05 n = 3000; 79% = 1 − 2.2/10.4 with Step 19's reference (not computed here) |
| §6 and App. G.2 "The frontier, in one sentence" | share of gain given up by the tilted campaign, one of the covert campaigns behind "gave up 58% or more" | 79% | [`step29d_tilted_distillation.py`](experiments/03_capability_frontier/step29d_tilted_distillation.py) — 79% = 1 − 2.2/10.4, as above; the 58% bound comes from Step 32 |
| §6 'The covert end'; App. G.2; Prediction log (Step 32, "Iterative rewriting reaches AUC ≤ 0.65"); App. N 'Representation' | held-out discriminator AUC, real on-task prompts vs written rewrites (first pass, then two regeneration rounds) | "AUC 0.97–0.98" (§6); "AUC 0.98 … after two more rounds … 0.97" (G.2); "Fail (0.98, 0.97, 0.97)" (Prediction log); "AUC 0.98" (App. N) | [`step32_written_attacker.py`](experiments/03_capability_frontier/step32_written_attacker.py) — Cell V, `crossfit(...)` → `VOI_AUCS`; prints "[W] round …" and "[W] discriminator AUC by round"; Cell Z "P32j" |
| §6 'The covert end'; App. G.2 | rewrites that passed as user-like (posterior < 0.5 and in S_v) | "7.9%"; "395 of 5,000 rewrites (7.9%)" | [`step32_written_attacker.py`](experiments/03_capability_frontier/step32_written_attacker.py) — Cell V, `keep` → `VOI_KEEP`, `len(VOICED)`; print "[W] discriminator AUC by round … kept"; Cell Z "P32b" |
| §6 'The covert end'; App. G.2; Prediction log (Step 32, "Distribution-matched attacker evades and gains ≥ 6") | importance-resampled (distribution-matched) campaign `matched_c05`: detection and gain at 3,000 queries | "evaded every detector and gained 4.4 points—42% of the dataset gain"; "Fail (evades; +4.4, ESS 151)" | [`step32_written_attacker.py`](experiments/03_capability_frontier/step32_written_attacker.py) — Cell W `[W] matched_c05` row (`RES`); Cell D `DRES[("matched_c05", sd)]`; Cell Z "P32k". The 42% is relative to Step 19's all-math +10.4 |
| §6 'The covert end'; App. G.2; Prediction log (P32k row) | effective sample size of the resampled rewrites | "effective sample of only 151 distinct rewrites"; "ESS 151" | [`step32_written_attacker.py`](experiments/03_capability_frontier/step32_written_attacker.py) — Cell V, `ESS = 1/sum(wts**2)`; print "[W] … effective sample size"; Cell Z "P32k" |
| §6 'The frontier, in one sentence'; App. G.2 | share of the detectable campaign's gain given up by covert campaigns | "58% or more" | [`step32_written_attacker.py`](experiments/03_capability_frontier/step32_written_attacker.py) — not isolated (summary): 58% = 1 − 4.4/10.4 uses this script's `matched_c05` gain (the best covert campaign); the others gave up more (e.g. 79% for Step 29d's tilted campaign) |
| §6; App. G.2; App. N 'Capability' | covert gains and their size in problems | "covert gains of 1.1–4.4 points are differences of 4–18 of 400 problems" (§6); "differences are 4–18 problems out of 400" (App. N) | [`step32_written_attacker.py`](experiments/03_capability_frontier/step32_written_attacker.py) — Cell D (`tilted_v_c05` +1.1 to `matched_c05` +4.4; 400 = `S32["N_EVAL"]`) |
| §6 and App. G.3 ('null runs alarmed within 16 weeks a quarter of the time on untouched weeks') | share of null runs with a first weekly alarm within 16 weeks | a quarter of the time | [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) — Cell 11, `SEQ[('all', d, 'null')]['frac_f']`; Cell 14 P35j ("first-alarm Tmax") |

### §7 Frozen evaluation on untouched weeks

| Paper location | Quantity | Value | Script — where |
|---|---|---|---|
| §7 'The test that matters: one frozen evaluation on untouched weeks'; App. I | multi-outlier design constants | bulk-edge factor of 1.05, minimum group spread of 5 accounts, cap of 20 eigenvectors | [`step27_31_bulk_edge_multioutlier_tilted.py`](experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) — Cell 8, `stats27` (`1.05 * e_hat`, participation number `>= 5`, `min(20, ...)`) |
| Table 3 (MiniLM, 64–252 accounts, row "False alarms"); Fig. 1b ("false alarms (1.8–1.9%)"); §7 'What held' and App. I ('1.8–1.9%'); Prediction log (Step 35, frozen calibration: 'Pass (1.8%, 1.9%, 1.9%') | held-out false-alarm rate, Two-scale / Hollowed / Multi-outlier / Volume scan (volume scan = rolling 8-week `vol_roll`) | 1.8% / 1.9% / 1.9% / 1.5% | [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) — Cell 8, `FPR_M = fpr_table(...)`, print "[P35a] held-out FPR" (tsC / tsRc / Tmax / vol_roll); Cell 14 P35a print and "[FPR matched]" |
| Abstract ('0.86–1.00 of weeks'); §1 item 1 ('86–100% of weeks'); §7 and App. I ('0.86–1.00') | dataset-campaign detection range over both encoders | 0.86–1.00 | [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) — Cell 8 `RES_M` and Cell 13 `RES_E`, source metamath_unique |
| §7 'What held'; App. I; Table 2 (Hollowed, Half padding, untouched); App. Table 15 (untouched, Hollowed) | hollowed detection of padded campaigns across three campaign seeds | 0.50–0.57 | [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) — Cell 8, `RES_M` (seed 0) and `RUNS[sd]` (seeds 1, 2); Cell 14 verdict rows "[matched]", "[seed1]", "[seed2]" metamath_mixed tsRc |
| Table 3 (MiniLM, 64–252 accounts, row "Least detectable (tilted)"); Fig. 1b (0.04); Table 2 (row "Least detectable", '≤0.04'); App. Table 15 (untouched, "Least detectable"); §7 and App. I ('at or below 0.04'); App. G.2 ('at most 0.04 on untouched weeks'); Prediction log (Step 35, tilted: 'Pass (≤ 0.04)') | weekly paired detection of the tilted campaign, Two-scale / Hollowed / Multi-outlier / Volume scan | 0.00 / 0.04 / 0.00 / 0.10 | [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) — Cell 8, `RES_M`, print "[matched] tilted_c05"; Cell 14 P35d |
| Table 3 (E5-small-v2, row "False alarms"); §7 and App. I ('0–1.3%') | held-out false-alarm rate, Two-scale / Hollowed / Multi-outlier / Volume scan | 1.3% / 0.3% / 0.0% / 6.2% | [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) — Cell 13, `FPR_E = fpr_table(...)`, print "[E5] held-out FPR" |
| Table 3 (E5-small-v2, row "Half padding"); §7 and App. I ('0.89 with E5') | weekly paired detection, Two-scale / Hollowed / Multi-outlier / Volume scan | 0.00 / 0.89 / 0.00 / 0.09 | [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) — Cell 13, `RES_E`, print "[matched] metamath_mixed" (second occurrence); Cell 14 P35f |
| Table 3 (MiniLM, whole weeks, row "Half padding"); §7 and App. I ('0.96 in whole weekly populations') | weekly paired detection, Two-scale / Hollowed / Multi-outlier / Volume scan | 0.00 / 0.96 / 0.00 / 0.51 | [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) — Cell 12, `RES_L`, print "[large] metamath_mixed" |
| §7 'What did not'; App. I; App. D.4 ('On the untouched weeks the ratio was 2.22'); Prediction log (Step 35, E5: '3.06 vs 2.22'; 'Census ratio 2.22, not 1.14') | real-to-shuffled top-eigenvalue ratio of R_c, MiniLM / E5 | 2.22 / 3.06 | [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) — Cell 9 `CEN["minilm"]` and Cell 13 `CEN["e5"]` (`census`), prints "[census]"; Cell 14 P35f |
| §7 'What held' and App. I ('49–59% of the non-sacrificial colluders'; 'both still caught every campaign'; selection on C 'again 0.00 here'); Prediction log (Step 35, sacrificial: 'Fail (1.00; recall 0.49)') | sacrificial scheduling: detection and non-sacrificial recall | 49–59%; 1.00; 0.00 | [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) — Cell 10, `SCHED`, print "[X] sacrificial" (and "[X] random" for comparison); Cell 14 P35h and "[SCHED]" |
| §7 'What did not' and App. I (ramp); Prediction log (Step 35, ramp: 'Fail (0.89 to 0.00)'); App. D.6 ('On the untouched weeks the attack defeated every detector') | abrupt -> ramped detection: multi-outlier; conformal two-scale; hollowed; rolling two-scale | 0.89 to 0.00; 0.80 to 0.02; 0.98 to 0.32; 0.79 to 0.32 | [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) — Cell 10, `TEMP`, print "[T] week" (cumulative; last line = final); Cell 14 P35i and "[TEMP]" |
| §7 'Covert is not invisible'; App. I; Prediction log (Step 35, tilted: 'Ranked by hollowed (AUC 0.97)') | AUC of hollowed selection: tilted (MiniLM), tilted (E5), random real prompts (MiniLM) | 0.97, 0.92, 0.88 | [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) — `roc(...)` in Cell 4 via `run_sources`; Cell 14 verdict rows "[matched] tilted_c05", "[e5] tilted_c05", "[matched] random", tsRc `auc=` |
| §2 'Data'; §7; App. I 'Design' | untouched data: test weeks; history weeks; English user turns; hashed IPs | 17 weeks; 31 weeks; 757,000; 76,000 | [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py) — Cell 2 print (turns, hashed IPs, history/holdout weeks); Cell 8 print "[matched] holdout test weeks" |

### §8 Real campaigns, relays, randomisation

| Paper location | Quantity | Value | Script — where |
|---|---|---|---|
| §8 'The provider's own randomness'; App. J (Prop. 5 paragraph); App. K; Prediction log (Step 34, P34a) | randomisation null on real, drifting traffic: rejection at 1% on null runs | one rejection at 1% in 120 null runs; Partial (1/40 at 250; 0/40 at 2,500 and 10,000) | [`step34_randomized_interventions.py`](experiments/05_randomized_interventions/step34_randomized_interventions.py) — Cell R, scenario `null` print "[R] A=... null"; Cell Z "P34a" print |
| §8 'The provider's own randomness'; App. J; App. K; Prediction log (Step 34, P34b/P34c) | failover campaign (reassigned within the block / one block later) detected | caught/detected in every run; 1.00 at 250 and 2,500 | [`step34_randomized_interventions.py`](experiments/05_randomized_interventions/step34_randomized_interventions.py) — Cell R, scenarios `fail0`, `fail1`; Cell Z "P34b/c" print |
| §8 'The provider's own randomness'; App. J; App. K; Prediction log (Step 34, P34d) | benign shared-queue team with failover fires the test | 82–98% of runs; Pass (0.82–0.98) | [`step34_randomized_interventions.py`](experiments/05_randomized_interventions/step34_randomized_interventions.py) — Cell R, scenario `team`; Cell Z "P34d" print |

## 3. Full map, script by script

Every row written by the annotation of each script, including appendix tables and the prediction log.

### `experiments/01_controlled_studies/`

#### Step 1 — [`step01_missing_mode.py`](experiments/01_controlled_studies/step01_missing_mode.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| Prediction log (Step 1), row "Union S(k) matches jittered-grid theory" (P1) | max relative error of the colluders' union structure factor vs jittered-grid theory | Pass (≤ 2% error) | Part A cell (shell table); VERDICT REPORT cell, `relA`, print "[P1] max rel err union vs theory" |
| Prediction log (Step 1), row "Colluder pair correlation −(1 − S̄)/K" (P2) | mean colluder-pair C_ab, measured vs predicted | Pass (−0.0451 vs −0.0451) | Part B cell (`pred_pair`); VERDICT REPORT cell, print "[P2] pair mean=... pred=..." |
| Prediction log (Step 1), row "BBP outlier below the bulk" (P3); App. E "Spectral detection condition" ("on a 2-D torus the measured lower outlier matched the BBP prediction to four decimals") | attacked λ_min, measured vs BBP prediction | Pass (0.0702 vs 0.0701) | Part B cell (`att["lam_min"]`, `att["lam_pred"]`); VERDICT REPORT cell, print "[P3] lam_min=... pred=..." |
| Prediction log (Step 1), row "Padding threshold f∗ = 0.445" (P4) | predicted padding threshold f* vs empirical f at power 0.5 | 0.445 predicted; Partial (0.408) | Part C cell (`fstar`); VERDICT REPORT cell, `fc = interp_cross(...)`, print "[P4] empirical f(power=0.5)" |
| Prediction log (Step 1), row "Outlier independent of K" (P5) | spread (max − min) of attacked λ_min over K ∈ {5, 10, 20, 40, 80} at f = 0.3 | Pass (spread 0.0017) | Part D cell; VERDICT REPORT cell, print "[P5] spread(max-min)=" |

#### Step 1 — [`step01_missing_mode_toy.py`](experiments/01_controlled_studies/step01_missing_mode_toy.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| Prediction log (Step 1), row "Union S(k) matches jittered-grid theory" (P1) | relative error of the colluders' union structure factor vs jittered-grid theory | Pass (≤ 2% error) | not isolated (Part A shell table: compare columns S_union and theory) |
| Prediction log (Step 1), row "Colluder pair correlation −(1 − S̄)/K" (P2) | mean colluder-pair C_ab, measured vs predicted | Pass (−0.0451 vs −0.0451) | Part B, print "colluder-pair mean C_ab ... pred" |
| Prediction log (Step 1), row "BBP outlier below the bulk" (P3); App. E "Spectral detection condition" ("on a 2-D torus the measured lower outlier matched the BBP prediction to four decimals") | attacked λ_min, measured vs BBP prediction | Pass (0.0702 vs 0.0701) | Part B, prints "lam_min  ATTACK  measured" and "lam_min  ATTACK  BBP prediction" |
| Prediction log (Step 1), row "Padding threshold f∗ = 0.445" (P4) | predicted padding threshold f* vs empirical f at power 0.5 | 0.445 predicted; Partial (0.408) | Part C, print "predicted threshold f*"; empirical value not isolated (read off the Part C power column; this script's padding grid is coarser and it does not interpolate) |
| Prediction log (Step 1), row "Outlier independent of K" (P5) | spread (max − min) of attacked λ_min over K ∈ {5, 10, 20, 40, 80} at f = 0.3 | Pass (spread 0.0017) | not isolated (Part D table, lam_min column) |

#### Step 2 — [`step02_theory_scaling_zoo.py`](experiments/01_controlled_studies/step02_theory_scaling_zoo.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| Prediction log (Step 2), row "Tracy–Widom calibration to 10−3"; App. E "Spectral detection condition" ("over 20,000 null worlds analytic Tracy–Widom thresholds ... matched false-alarm rates from 10−1 to 10−4") | false-alarm rate of analytic TW thresholds on λ_min / λ_max, α = 0.1 ... 0.0001, N_NULL = 20,000 null worlds (GPU) | Pass (to 10−4); "from 10−1 to 10−4" | Part A cell, `rowsA` loop over alpha; VERDICT REPORT print "[A] FPR (min / max) at alpha" |
| Prediction log (Step 2), row "Raw fails, whitened calibrated under non-uniform traffic" | false alarms of the raw vs intensity-whitened statistic on a non-uniform benign null (A3) | Pass (100% vs 0.53%) | Part A cell, A3 prints "RAW" / "WHITENED"; VERDICT REPORT print "[A3]" |
| Prediction log (Step 2), row "TW-corrected f50 at unseen sizes" | gap between measured and TW-predicted f50 across L, γ | Pass (≤ 0.033) | Part B cell, print "f50 meas=... pred=..."; VERDICT REPORT "[B]" lines (max gap not isolated) |
| Prediction log (Step 2), row "Optimal window peak" | predicted vs measured optimal window n_max (C1) | Pass (37 vs 36) | Part C cell, print "predicted optimal nmax = ...; measured best = ..." |
| Prediction log (Step 2), row "N∗ ∝ A" | log-log slope of critical campaign size N* vs A (C2) | Pass (slope 0.95) | Part C cell, `slope_m`, print "log-log slope N* vs A: measured"; VERDICT REPORT "[C2] log-log slope" |
| Prediction log (Step 2), row "Same-marginal stratified attacker detectable" | whitened clean-λ_min rate for S2 `warped_jitter` (Part D) | Pass (1.00) | Part D cell, row "S2 warped_jitter", column WHT cln; VERDICT REPORT "[D] S2 warped_jitter" |

#### Step 3 — [`step03_highdim_rff.py`](experiments/01_controlled_studies/step03_highdim_rff.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| Prediction log (Step 3), row "High-dimensional analytic calibration" (P-A) | analytic TW false alarms in d = 2/8/32; fixed by self-normalised rows and simulation-calibrated thresholds | Fail, then fixed (consequence: row normalisation) | config cell `ROWNORM = True`; `evaluate(...)` row normalisation; Part A cell, print "analytic-TW FPR" and empirical thresholds `emp_tmin`/`emp_tmax` |
| Prediction log (Step 3), row "Herding detectable in every dimension" (P-B2) | whitened union S̄ of `herding` campaigns | Fail (S̄ 0.70–0.89) | Part B cell (S per bandwidth) and Part C summary, column S_union, design `herding`; VERDICT REPORT "[B]"/"[C]" lines |
| Prediction log (Step 3), row "Mismatched herding detectable" (P-B4) | whitened union S̄ of `herding_mismatched` | Fail (S̄ ≈ 1) | same cells, design `herding_mismatched` |
| Prediction log (Step 3), row "k-means++ shows λmax" (P-B3) | λ_max rate of `kmeanspp` | Fail (looks i.i.d.) | same cells, design `kmeanspp`, column max |

#### Step 4 — [`step04_linkage_price_law.py`](experiments/01_controlled_studies/step04_linkage_price_law.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| Prediction log (Step 4), row "One curve, x50 ∈ [1.0, 1.35]" (P4.1) | logistic-fit x50 of clean-λ_min power vs x; configurations classified by x > x50 | Pass (1.052; 47/48) | Part C cell, `x50`, prints "logistic fit: x50" and "configs classified correctly"; VERDICT REPORT "[C]" |
| Prediction log (Step 4), row "Herding S̄ falls with N" (P4.2) | whitened union S̄ of `herding` as N grows | Fail (rises) | Part B cell, column S for design `herding` across N; VERDICT REPORT "[B]" lines |
| Prediction log (Step 4), row "Aligned herding detected almost everywhere" (P4.3) | clean-λ_min power of `herding_aligned` vs x | Partial (exactly where x > 1) | Part B cell, columns x and clean for `herding_aligned` |
| Prediction log (Step 4), row "Spectral linkage AUC ≥ 0.9" (P4.4) | colluder-identification AUC, spectral vs pairwise, where clean λ_min fires | Pass (0.986 vs 0.481) | Part C cell, print "P4.4 linkage where clean_min fires" |
| Prediction log (Step 4), row "Oracle-only band exists" (P4.5) | power in the band 2.3 < z_oracle < √A, oracle vs spectral | Pass (0.75 vs 0.03) | Part C cell, print "P4.5 oracle-only band" |

#### Step 5 — [`step05_real_extraction_cifar10.py`](experiments/01_controlled_studies/step05_real_extraction_cifar10.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| Prediction log (Step 5), row "Victim ≥ 85%; random looks null" (P5.0) | victim test accuracy; random campaigns look null | Pass (88.5–89.1%) | Cell 2 print "victim test acc" / VERDICT "victim test acc"; Part 5 print "P5.0 random-campaign false alarms" |
| Prediction log (Step 5), row "Active strategies beat random by ≥ 1 point" (P5.1) | fidelity gain of entropy / k-center over random | Partial (k-center yes, entropy no) | Cell 4 print "mean fidelity" (Δ pt column) |
| Prediction log (Step 5), row "No x > 1; concentration caught by λmax" (P5.2) | x and λ_max power per (strategy, N, A) | Pass | Part 4 detection print (`x=`, `max=`) / VERDICT `[det]` lines |
| Prediction log (Step 5), row "Fidelity gain tracks concentration" (P5.3) | Spearman(Δfidelity, concentration) | Fail (ρ = −0.15) | Part 5 print "P5.3 Spearman" |
| Prediction log (Step 5), row "Herding undetected because S̄ ≈ 1" (P5.5) | herding union S̄ in the defender embedding | Partial (S̄ 1.1–1.5) | Part 4 detection print `S=` for `herding` rows |
| Prediction log (Step 5), row "Spectral ≥ per-account, gap grows with A" (P5.6) | mean power spectral λ_max vs per-account max, by A | Pass on mean power; not cell by cell | Part 5 print "P5.6 A=..." |
| Prediction log (Step 5), row "Law holds on real attacks" (P5.4) | Step-4 law on real attacks (MAE, x range) | Pass, uninformative (all x < 0.25) | Part 5 print "P5.4 law on real attacks"; `x=` in Part 4 print |
| App. E 'CIFAR-10 regime map' | every active strategy concentrated in the victim's embedding and was detected at the upper edge | qualitative (no number) | Part 4 detection print (`S=` > 1, `max=`) |
| Table 16 (App. O, "Main settings"), CIFAR-10 rows | embedding / modes / bandwidths / whitening floor / reference / calibration | victim penultimate layer, PCA-32; m = 400; h ∈ {2, 4, 8}; 10−4; 300,000 benign queries; 1,000 null worlds per configuration | Cell 1 configuration (`D_DEF`, `M_MODES`, `HS`, `FLOOR`, `N_REF`, `N_NULL`); Bonferroni over bandwidths is added in Step 6 |

#### Step 6 — [`step06_baselines.py`](experiments/01_controlled_studies/step06_baselines.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| Prediction log (Step 6), row "Spectral vs global crossover at K ≈ 2.3 √A" (P6.1) | informative cells where the predicted winner (spectral below, global above the crossover) wins | Pass (7 of 8 cells; 2/2) | Part 5 print "P6.1 informative cells" |
| App. E 'CIFAR-10 regime map' | regime map: spectral dominates when many colluders send few queries each and K ≲ 2.3 √A; per-account tests and Higher Criticism suffice with few heavy colluders; global stream wins with many colluders | 2.3 | Part 5 head-to-head table (`winner` column) / VERDICT `[d]` lines |
| Prediction log (Step 6), row "HC ≥ per-account; spectral ≥ HC at N = 2,000" (P6.2) | cell counts | Pass (29/30; 8/10) | Part 5 print "P6.2 HC≥acct in ... cells; spectral≥HC ..." |
| Prediction log (Step 6), row "Random campaigns ≤ 2%; linkage AUC ≈ 0.5" (P6.3) | random-campaign false alarms; spectral linkage AUC | Pass (1.9%; 0.49–0.51) | Part 5 prints "P6.3 random-campaign false alarms" and "spectral linkage AUC on random campaigns by A" |
| Prediction log (Step 6), row "No real strategy triggers the lower edge" (P6.4); App. E 'Spectral detection condition' ("No real extraction strategy we tested triggered the lower edge") | missing-mode rate on real strategies | Pass (0.002) | Part 5 print "P6.4 missing-mode rate" |
| Table 16 (App. O, "Main settings"), CIFAR-10 rows | bandwidths with Bonferroni correction; calibration | h ∈ {2, 4, 8} with Bonferroni correction; 1,000 null worlds per configuration, Bonferroni over bandwidths | Cell 1 configuration (`HS`, `N_NULL`); Cell 5 `ALPHA_B` |

#### Step 7 — [`step07_robustness.py`](experiments/01_controlled_studies/step07_robustness.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| Prediction log (Step 7), row "Only whole-window resampling calibrates" (P7.1) | suite false-positive rate on realistic null windows, window-resampled calibration | Pass (suite 0.0–0.6%) | Part A print "calibrated by window_resample" (`suite=`) / VERDICT `[A]` lines |
| App. E 'The dose–response study' | window-resampled suite false alarms at s = 1; spectral and global tests under independent-account resampling | 0.6%; 31%; 100% | Part A prints (`window_resample` suite; `acct_resample` spec / glob). Plausibly from here: Step 7's realistic windows have the structure of Step 8's s = 1, and account resampling exists only in this script |
| Prediction log (Step 7), row "Evading the suite needs padding f ≥ 0.75" (P7.2) | suite detection vs padding fraction f | Fail (no detection even unpadded) | Part B print "suite: equal=..." at f=0.00 / VERDICT `[B]` lines |
| Prediction log (Step 7), row "Heterogeneous volumes cost ≤ 0.2 power" (P7.3) | suite power, heterogeneous colluder volumes | Untestable (power 0) | Part B `hetero` variant |
| Prediction log (Step 7), row "Time-splitting evades; aggregation restores" (P7.4) | any-window suite detection, split / aggregated | Untestable (power 0) | Part B `split2` / `split4` / `agg4` variants |
| Prediction log (Step 7), row "Upper-edge rule predicts power at A ≤ 10^4" (P7.5) | agreement of rule (S̄ − 1)√m_eff/√A > 1.36 with spectral power > 0.5 | Fail (10/24 cells) | Part C print "P7.5 rule ... agreement" (fraction of the 24 cells) |

#### Step 8 — [`step08_dose_response_taskaware.py`](experiments/01_controlled_studies/step08_dose_response_taskaware.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| Table 4, row "100, 20 / entropy / geometry" | power vs s | 1.00 / 1.00 / 1.00 / 0.00 / 0.00 / 0.00 | Part B print "geo/dir by s" (first value), A=100 entropy N=10000; VERDICT `[B]` `geo_suite=` |
| Table 4, row "100, 20 / entropy / task-aware" | power vs s | 1.00 / 1.00 / 1.00 / 1.00 / 0.12 / 0.17 | Part B print "geo/dir by s" (second value), A=100 entropy N=10000; VERDICT `[B]` `dir_suite=` |
| Table 4, row "100, 20 / k-center / geometry" | power vs s | 1.00 / 1.00 / 1.00 / 0.00 / 0.00 / 0.00 | Part B, A=100 kcenter N=10000, `geo_suite` |
| Table 4, row "100, 20 / k-center / task-aware" | power vs s | 1.00 / 1.00 / 1.00 / 0.79 / 0.04 / 0.08 | Part B, A=100 kcenter N=10000, `dir_suite` |
| Table 4, row "1000, 100 / entropy / geometry" | power vs s | 1.00 / 0.67 / 0.04 / 0.00 / 0.04 / 0.04 | Part B, A=1000 entropy N=10000, `geo_suite` |
| Table 4, row "1000, 100 / entropy / task-aware" | power vs s | 1.00 / 0.96 / 0.67 / 0.33 / 0.04 / 0.04 | Part B, A=1000 entropy N=10000, `dir_suite` |
| Table 4, row "1000, 100 / k-center / geometry" | power vs s | 1.00 / 0.38 / 0.04 / 0.00 / 0.04 / 0.04 | Part B, A=1000 kcenter N=10000, `geo_suite` |
| Table 4, row "1000, 100 / k-center / task-aware" | power vs s | 1.00 / 0.21 / 0.25 / 0.08 / 0.04 / 0.04 | Part B, A=1000 kcenter N=10000, `dir_suite` |
| Fig. 3 (App. B, "The barrier in a controlled setting") | geometry detection power vs strength of benign coordination, 100 and 1,000 accounts | power erased at strength 0.25 (100 accounts); falls at 0.05 and vanishes at 0.1 (1,000 accounts) | Part B `geo_suite` at N=10000 (same cells as Table 4 geometry rows); the paper figure is drawn by a separate figure script, and how the two strategies are combined per curve is not isolated here |
| App. E 'The dose–response study' | window-resampled suite false alarms at s = 1 | 0.6% | plausibly Part A print "null FPR" at s=1.00 (`geo_suite`); the companion 31% / 100% account-resampling figures come from Step 7 Part A |
| App. E 'The dose–response study' | union S̄ of benign ten-account clusters vs attacks | 83–159, against 26–30 for the attacks | Part C prints "benign 10-account clusters ... union S" and "attacks at N=10000: union S" |
| App. E 'The dose–response study' | task-aware suite tolerated one cluster at 100 accounts and a few at 1,000, then failed | qualitative (see Table 4) | Part B `dir_suite` |
| Prediction log (Step 8), row "Geometry breaks at s ∈ [0.05, 0.5]" (P8.1) | interpolated 50% point s50 | Partial (0.175 at A = 100; ≤ 0.06 at 1,000) | prediction print "P8.1/P8.2 ... (s50 ...)" |
| Prediction log (Step 8), row "Task-aware suite ≥ 0.8 at s = 1" (P8.2) | directional-suite power at s = 1 | Fail (≤ 0.17) | prediction print "P8.1/P8.2 ... dir s=1" |
| Prediction log (Step 8), row "Calibration ≤ 1.5× nominal" (P8.3) | null / random-campaign false-positive rate | Partial (mean 1.1%, worst 3.3%) | worst: print "P8.3 max null FPR ... random-campaign max"; mean: not isolated (summary over Part A / Part B random rows) |
| Prediction log (Step 8), row "Clusters out-shine attacks in geometry, not margin" (P8.4) | benign-cluster vs attack union S̄ and low-margin share | Partial (geometry yes, margin no) | Part C diagnostic prints |

### `experiments/02_development_weeks/`

#### Step 9 — [`step09_wildchat_llm.py`](experiments/02_development_weeks/step09_wildchat_llm.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| §5 "Calibration breaks first"; App. D.1; Prediction log (Step 9, "Fixed calibration transfers to later weeks") | false alarms on later (held-out) null weeks of thresholds set once on early weeks, nominal 1% | 12% (Prediction log: "Fail (12%)") | Part A cell, `row["suite"]` in print "held-out-week FPR ... suite="; Part C print "P9.1" |
| Prediction log (Step 9, "Real users structured: λmax real/shuffled ≥ 2") | median λ_max of real windows / same windows with prompts shuffled across accounts | Pass (4.3–5.1) | Part A cell, print "(ratio ...)" from `lm_real`/`lm_shuf`; Part C print "P9.2" |
| Prediction log (Step 9, "Capability campaigns detected at 5,000 and 20,000") | suite power of `capability`/`capability_kcenter` campaigns | Partial (20,000 via global only) | Part B cell, `suite` and `glob` columns of the per-row print; Part C print "P9.3" |
| Prediction log (Step 9, "Random campaigns ≤ 2× nominal") | suite power of `random` campaigns | Fail (8–21% floor) | Part B cell, `random` rows; Part C print "P9.4" |
| Prediction log (Step 9, "Padding costs ≥ 0.3 power at 5,000") | power cost of 50% padding, `capability` vs `capability_pad50` | Fail as specified; 0.79 at 20,000 | Part C print "P9.5" (N = 5,000); the N = 20,000 value from Part B rows `capability` vs `capability_pad50` at N = 20000 (not printed as a difference) |
| §2 "Data" | development data: hashed IPs and weeks of WildChat-1M | 98,331 hashed IPs over 56 weeks | plausibly the account/week counts printed at the end of Cell 2 (same loading code in Steps 9–12); not isolated |
| §2 "Data"; App. Table 16 (Embedding, Fourier modes, Whitening floor) | attacker pool share; MiniLM → PCA rescaled to total variance 4; m = 400 complex modes, h = 4, floor 10−4 | 20%; "rescaled to total variance 4"; "m = 400"; "h = 4"; "10−4" | Cell 1 configuration (`POOL_FRAC`, `D_DEF`, `H_BW`, `M_MODES`, `FLOOR`) and the PCA step in Cell 2 |

#### Step 10 — [`step10_wildchat_drift_granularity.py`](experiments/02_development_weeks/step10_wildchat_drift_granularity.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| Fig. 4a (App. D.1); App. Table 5, last row "Held-out false alarms at A = 100 / 500" | held-out suite false alarms at nominal 5%, schemes fixed / rolling / rolling + lagged ref. | 18.8% / 56.2%; 5.0% / 24.7%; 3.4% / 28.7% | Part A cell, print "held-out suite FPR" (`rowsA[...]["fpr"]`); Part C print "P10.1"; verdict "[A]" lines |
| App. Table 5, row "100, 20 random natural 20,000" | share; paired detection Fixed / Rolling / Rolling + lag | 0.94; 0.00 / 0.00 / 0.00 | Part B cell, `share` and `{scheme}_attrib` in the per-row print; verdict "[B]" lines (`att`) |
| App. Table 5, row "100, 20 broad capability (1 of 10) 5,000" | share; Fixed / Rolling / Rolling + lag | 0.81; 0.03 / 0.04 / 0.21 | Part B cell, as above (`capability`, N = 5000) |
| App. Table 5, row "100, 20 broad capability 20,000" | share; Fixed / Rolling / Rolling + lag | 0.94; 0.82 / 0.90 / 0.93 | Part B cell, as above (`capability`, N = 20000) |
| App. Table 5, row "100, 20 narrow capability (1 of 30) 1,000" | share; Fixed / Rolling / Rolling + lag | 0.46; 0.00 / 0.00 / 0.03 | Part B cell, as above (`capability_narrow`, N = 1000) |
| App. Table 5, row "100, 20 narrow capability 5,000"; App. F "Small monitored populations" | share; Fixed / Rolling / Rolling + lag | 0.81; 0.78 / 0.93 / 0.94 ("caught 94% of the time in 100-account windows") | Part B cell, as above (`capability_narrow`, N = 5000) |
| App. Table 5, row "100, 20 broad, 50% natural padding 20,000" | share; Fixed / Rolling / Rolling + lag | 0.94; 0.25 / 0.04 / 0.38 | Part B cell, as above (`capability_pad50`, N = 20000) |
| App. Table 5, row "500, 100 broad capability 5,000" | share; Fixed / Rolling / Rolling + lag | 0.46; 0.10 / 0.07 / 0.29 | Part B cell, as above (A = 500, `capability`, N = 5000) |
| App. Table 5, row "500, 100 narrow capability 5,000" | share; Fixed / Rolling / Rolling + lag | 0.46; 0.44 / 0.78 / 0.71 | Part B cell, as above (A = 500, `capability_narrow`, N = 5000) |
| App. F "Small monitored populations" | equal-volume colluders' fraction of the window | 46–94% | Part B cell, `share` |
| Prediction log (Step 10) | locked predictions P10.1–P10.5 | P10.1 Partial (A = 100 yes; A = 500 no); P10.2 Partial (lagged 0.21, 0.29); P10.3 Pass (0.94; 0.71); P10.4 Pass; P10.5 Partial (A = 100 yes) | Part C "predictions" cell (prints "P10.1"–"P10.5") |
| App. Table 5 caption; App. Table 16 (Calibration) | 8 held-out weeks, 3 campaigns × 3 windows per week, α = 5%; 250 windows from the 4 preceding weeks (small populations) | as quoted | Cell 1 configuration (`N_TW`, `TRIALS9`, `W_TW`, `ALPHA`, `N_CAL_ROLL`, `LAG`) |

#### Step 11 — [`step11_wildchat_dimension_robustness.py`](experiments/02_development_weeks/step11_wildchat_dimension_robustness.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| App. F "Small monitored populations" | broad capability, 5,000 prompts: paired (attributable) detection from d = 32 to d = 384 | 0.18 to 0.35 | Cell 6 print "P11.3" (`capability`, N = 5000) and verdict "[B]" lines; plausibly A = 100, `rolling_lag` (paper does not name the cell) |
| App. F "Small monitored populations" | broad, 50% natural padding, 20,000 prompts: paired detection from d = 32 to d = 384 | 0.36 to 0.60 | verdict "[B]" lines, `capability_pad50` N = 20000 (`att`); plausibly A = 100, `rolling_lag` |
| Prediction log (Step 11, "Lagged calibration nominal at 100, not 500, every d") | `rolling_lag` held-out FPR at A = 100 / A = 500 over d | Pass (3.4–4.4%; 22–39%) | Cell 5 print "held-out FPR"; Cell 6 print "P11.1"; verdict "[A]" lines |
| Prediction log (Step 11, "Narrow at 5,000 ≥ 0.8 at every d") | narrow capability, N = 5,000, A = 100, paired detection | Pass (0.94–0.96) | Cell 6 print "P11.2" |
| Prediction log (Step 11, "Broad at 5,000 ≤ 0.35 at every d and A") | broad capability, N = 5,000 | Partial (up to 0.44 at A = 500) | Cell 6 print "P11.3" |
| Prediction log (Step 11, "Random ≤ 0.05") | max paired detection of random campaigns per d | Pass (≤ 0.01) | Cell 6 print "P11.4" |
| Prediction log (Step 11, "Padding drops ≥ 0.4 at every d") | drop `capability` → `capability_pad50`, N = 20,000, A = 100, `rolling_lag` | Partial (0.32 at d = 384) | Cell 6 print "P11.5" |
| App. Table 16 (Embedding) | MiniLM, PCA to d ∈ {32, 64, 128, 384}, rescaled to total variance 4 | as quoted | Cell 1 `DIMS`; Cell 2 `make_Z` |

#### Step 12 — [`step12_realistic_deployment.py`](experiments/02_development_weeks/step12_realistic_deployment.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| Table 1, row "MetaMathQA (generated math)"; Prediction log (Step 12, "Two-scale beats spectral on a generated set") | weekly paired detection: Two-scale / Global / MMD / Spectral / Per-account, HC, Fraudar | 0.93 / 0.71 / 0.60 / 0.19 / ≤0.01; Prediction log: Pass (0.93 vs 0.19) | Cell 8 per-row print (`metamath`, K = 50); verdict "[B]" lines; Cell 10 print "P12.3/4" |
| Table 1, row "Magicoder (generated code)" | same columns | 0.62 / 0.46 / 0.32 / 0.00 / 0.00 | Cell 8 print (`magicoder`, K = 50); verdict "[B]" |
| Table 1, row "Alpaca (generated, broad)" | same columns | 0.01 / 0.00 / 0.01 / 0.00 / 0.00 | Cell 8 print (`alpaca`, K = 50); verdict "[B]" |
| Table 1, row "Narrow cluster of real prompts"; Prediction log (Step 12, "Natural campaigns ≤ 0.2 on every detector") | same columns | 0.12 / 0.04 / 0.27 / 0.00 / 0.00; Prediction log: Fail (MMD 0.27) | Cell 8 print (`wild_narrow`, K = 50); Cell 10 print "P12.2" |
| Table 1, row "Random real prompts" | same columns | 0.01 / 0.00 / 0.01 / 0.00 / 0.00 | Cell 8 print (`random`, K = 50); verdict "[B]" |
| Table 1, last row "Realised false alarms"; Fig. 4b; §5 "Calibration breaks first"; App. D.1; Prediction log (Step 12, "Every detector's false alarms ≤ 2%") | held-out false alarms at nominal 1%, rolling calibration, week-clustered intervals | Two-scale 0.7%, Global 4.1%, MMD 11.6%, Spectral 7.8%, Per-account / HC / Fraudar 5.3 / 7.5 / 2.9%; volume test 14.2% (§5: "2.9–14.2%"; D.1: "4.1–7.8%"); Prediction log: Fail (only two-scale) | Cell 7 print "held-out FPR" (`FPR[d]` with `boot_ci`); verdict "[FPR]" lines; Cell 10 print "P12.7" |
| Prediction log (Step 12, "Volume-share test ≤ 0.1") | max paired detection of the volume test | Pass (≤ 0.02) | Cell 10 print "P12.1" (`vol` column of Cell 8) |
| Prediction log (Step 12, "Generated sets at K = 200: MMD or global ≥ 0.5") | — | Partial (K = 200 infeasible) | Cell 10 print "cells skipped" and "P12.3/4" (largest evaluated K) |
| Prediction log (Step 12, "Partial linkage raises per-account or HC by ≥ 0.1") | — | Untestable (base ≈ 0) | Cell 8 "linked(rho=0.3)" part of the print; Cell 10 print "P12.5" |
| Prediction log (Step 12, "Sharding: spectral ≤ 0.1"); Corrections (docs/corrections.md), "Corrections made during the study" (Sharding) | — | Untestable (1–5 shards; unpaired) | Part 5 (Cell 9) print "sharded ..."; Cell 10 print "P12.6" |
| Table 1 caption; §5; App. D | K = 50 volume-matched colluders, 24 development weeks; "about 1,640 queries a week" | as quoted | Cell 1 (`KS`, `N_TEST_W`); 1,640 plausibly the `N~` printed for K = 50 in Cell 8 (protocol value reused by later Steps); not isolated |
| §2 "Data" | monitored accounts have at least 10 queries in the week; windows hold 64–364 accounts | as quoted | Cell 1 `MIN_PER_ACCT`; window sizes plausibly the "accounts per test-week window" print in Cell 10 (not isolated) |
| App. Table 16 (Fourier modes, Reference, Calibration, MMD baseline, Fraudar baseline) | m = 300 modes with augmented real whitening; h = 4 coarse with h = 0.5 fine; up to 100,000 benign turns from the four preceding weeks; 40 null windows per week, thresholds from the 8 preceding weeks; MMD window sample of 2,000 against 20 reference batches of 500; 100 k-means topics, weights 1/ log(degree + 5) | as quoted | Cell 1 configuration; Cell 5 (`make_bank_aug`, `mmd2`, `fraudar`); Cell 6 (lagged reference) |
| Corrections (docs/corrections.md), "Corrections made during the study" (Padding construction) | in Steps 12–13 the padded campaign assigned natural prompts to separate colluder accounts | (segregated construction; two-scale robustness 0.62 holds only for it) | Cell 8 `campaign_points` + `labc`: `metamath_pad50` here is the segregated construction; the 0.62 value is not isolated (may come from Step 13) |

#### Step 13 — [`step13_twoscale_mechanism_scale.py`](experiments/02_development_weeks/step13_twoscale_mechanism_scale.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| §5 'Coordinated, dataset-like campaigns are catchable'; App. D.2 | two-scale flagged-group precision (share of the group that are colluders) | 0.94–1.00 (§5); 1.00 at K = 50, 0.94–0.96 at K = 10 (App. D.2) | Cell 6 `run_block`, `prec` list -> `ts_precision`; printed "two-scale group precision" (weekly, `metamath_unique`/`metamath_rephrase`, K = 10/50) and verdict `ts_prec=` |
| §5 'Coordinated, dataset-like campaigns are catchable'; App. D.2 | two-scale flagged-group recall (share of colluders in the group) | 12–31% | Cell 6 `run_block`, `rec` list -> `ts_recall`; printed "recall" and verdict `ts_rec=` (range plausibly over weekly sources and K) |
| Prediction log (Step 13, P13.4) | group precision; recall (metamath_unique, weekly K = 50) | Pass (1.00; recall 0.17) | Cell 8, print "P13.4" |
| Table 7 rows "MetaMathQA, one prompt per source problem" / "four variants per source problem"; App. D.3; Prediction log (Step 13, P13.1) | weekly paired detection, K = 50: two-scale / global / MMD | 0.88 / 0.68 / 0.57 (one prompt); 0.80 / 0.71 / 0.60 (four variants); App. D.3 "from 0.88 to 0.80"; Prediction log "Fail (0.88 vs 0.80)" | Cell 7 Part A (weekly), rows `metamath_unique` / `metamath_rephrase` K=50, columns `twoscale`, `glob`, `mmd`; Cell 8 print "P13.1" |
| Table 7 rows "WildChat broad cluster, random prompts" / "farthest-point deduplicated"; Prediction log (Step 13, P13.2) | weekly paired detection, K = 50: two-scale / global / MMD | 0.01 / 0.01 / 0.07 (random); 0.04 / 0.02 / 0.07 (deduplicated); Prediction log "Fail (0.04 vs 0.01)" | Cell 7 Part A, rows `wild_cluster` / `wild_cluster_dedup` K=50; Cell 8 print "P13.2" |
| Prediction log (Step 13, P13.3) | two-scale held-out false alarms, weekly; 4-week | Partial (0.6%; 9.8%) | Cell 6 `FPR[...]` ("held-out FPR" print); Cell 8 print "P13.3" |
| Prediction log (Step 13, P13.5) | 4-week populations, K = 200: two-scale on metamath_unique; on random | Pass (0.86; 0.00) | Cell 7 Part B rows; Cell 8 print "P13.5" |
| App. D.5 'Smaller colluder shares'; Prediction log (Step 13, P13.6); App. L item 4; Corrections (docs/corrections.md), 'Corrections' ('Sharding') | paired attacker-aware sharding, one colluder per 100-account shard, per-shard power of every detector | "defeated every detector in all 352 shard tests" (App. D.5); "Pass (0.00 for all)" (Prediction log); "Sharding defeats every detector" (App. L) | Cell 6 `run_block(..., shard=True)` shard branch, printed "sharded (one colluder per 100-account shard)" with `n=`; Cell 8 print "P13.6" (352 plausibly the total of `n` over the two sources) |

#### Step 15 — [`step15_query_cluster_fair_baselines.py`](experiments/02_development_weeks/step15_query_cluster_fair_baselines.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| §5 'Coordinated, dataset-like campaigns are catchable'; App. D.2 'Its edge is calibration, not ranking'; Table 6 row "MetaMathQA (generated)"; Prediction log (Step 15, P15.5) | oracle-matched 1% TPR, two-scale vs global stream vs MMD (weekly, K = 50) | 0.80 vs 0.70 vs 0.58; Prediction log "Partial (0.80 vs 0.58; group MMD fails)" | Cell 8 `roc(null_v[d], att_v[d])` -> `row[d + "_roc"]["tpr1"]`; Cell 12 print "P15.5" and verdict "ROC:" lines (`twoscale`, `glob`, `mmd`) |
| §5 'Coordinated, dataset-like campaigns are catchable'; App. D.2 | AUC, global stream vs two-scale | 0.97 against 0.88 | Cell 8 `roc(...)["auc"]`; Cell 12 verdict "ROC:" line (`glob`, `twoscale`) |
| Table 6 row "MetaMathQA (generated)", cols Two-scale / Global stream / MMD | AUC / TPR at 1% / TPR at 5% | 0.88 / 0.80 / 0.81; 0.97 / 0.70 / 0.74; 0.94 / 0.58 / 0.62 | Cell 8 `roc(...)`; Cell 12 verdict "ROC:" line (`auc`, `tpr1`, `tpr5`), source `metamath_unique` K=50 |
| Table 6 row "Human GSM8K questions", cols Two-scale / Global stream / MMD | AUC / TPR at 1% / TPR at 5% | 0.96 / 0.95 / 0.95; 0.99 / 0.88 / 0.94; 0.97 / 0.73 / 0.80 | same, source `gsm8k_human` K=50 |
| Table 6 row "Half natural, mixed into every account", cols Two-scale / Global stream / MMD | AUC / TPR at 1% / TPR at 5% | 0.68 / 0.24 / 0.39; 0.75 / 0.05 / 0.11; 0.71 / 0.07 / 0.14 | same, source `metamath_mixed` K=50 |
| Table 6 row "Real follow-up after every prompt", cols Two-scale / Global stream / MMD | AUC / TPR at 1% / TPR at 5% | 0.68 / 0.24 / 0.40; 0.75 / 0.06 / 0.11; 0.72 / 0.11 / 0.20 | same, source `metamath_followup` K=50 |
| Table 6 row "Random natural prompts", cols Two-scale / Global stream / MMD | AUC / TPR at 1% / TPR at 5% | 0.48 / 0.00 / 0.07; 0.39 / 0.00 / 0.04; 0.39 / 0.02 / 0.02 | same, source `random` K=50 (Table 6's QC column is the repaired detector of Step 18, not this script) |
| §5 'It detects topic shift, not extraction'; App. D.3; Prediction log (Step 15, P15.4) | weekly paired two-scale detection, human GSM8K vs MetaMathQA (K = 50) | 0.94 vs 0.79; Prediction log "Pass (0.94 vs 0.79)" | Cell 8 rows `gsm8k_human` / `metamath_unique` (`twoscale`); Cell 12 print "P15.4" |
| App. D.3 | weekly paired two-scale detection, MetaMathQA behind a fixed template | 0.94 | Cell 8 row `metamath_template` K=50 (`twoscale`); verdict "[W]" line |
| App. D.3; Prediction log (Step 15, P15.3) | two-scale, unique prompts vs exact repeats | "from 0.79 to 0.73"; Prediction log "Fail (0.79 to 0.73)" | Cell 8 rows `metamath_unique` / `metamath_repeat`; Cell 12 print "P15.3" |
| App. F 'Mechanism probes'; Prediction log (Step 15, P15.2) | distance between rephrasings of one problem (median); share within h = 0.5 | median 0.67; only 35% within the fine bandwidth; Prediction log "Pass (median 0.67)" | Cell 9 `MECH["rephrase_median"]`, `MECH["frac_below_h"]` ("[M] distance between rephrasings" print); Cell 12 print "P15.2" |
| Prediction log (Step 15, P15.1); Corrections (docs/corrections.md), 'Corrections' ('Query-cluster calibration and the population protocol') | first query-cluster (QC) detector: false alarms; AUC under mimicry | Fail (detector miscalibrated, 64% false alarms; AUC 0.89–0.90) | Cell 8 `FPR["qc"]` ("held-out FPR" print), `row["qc_roc"]["auc"]`; Cell 12 print "P15.1" |
| Prediction log (Step 15, P15.6) | camouflage (most natural-looking 20% of MetaMathQA), two-scale | Fail (two-scale rose to 0.94) | Cell 8 row `metamath_camouflage`; Cell 12 print "P15.6" |
| Prediction log (Step 15, P15.7); Corrections (docs/corrections.md), 'Corrections' ('Query-cluster calibration and the population protocol') | realistic share, best calibrated detector | Fail (0.00 at A = 1,000) | Cell 10 `RESP`; Cell 12 print "P15.7" |
| Prediction log (Step 15, P15.8) | mixed padding caught within 12 weeks; P(>= 1 false alarm in 12 weeks), two-scale alone | Partial (two-scale alone: 0.96, 9.6%) | Cell 11 `LIFE[("metamath_mixed", "twoscale")][12]` and the two-scale "at least one false alarm in 12 weeks" print |

#### Step 17 — [`step17_calibrated_query_cluster.py`](experiments/02_development_weeks/step17_calibrated_query_cluster.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| Prediction log (Step 17, P17.1); App. F 'Query-level pooling' ("robust per-cell calibration fired on 28% of null weeks"); App. D.4 and §5 'Padding, and a fix that looks at directions' (first of the "four repairs that pool topic cells across accounts") | corrected query-cluster (QC) held-out false alarms, weekly | Fail (28%) | Cell 8 `FPR["qc"]` ("held-out FPR" print); Cell 12 print "P17.1" |
| Prediction log (Step 17, P17.2) | QC paired power under mimicry (mixed padding, follow-up); AUC | Fail (0.38–0.40; AUC 0.80–0.82) | Cell 8 rows `metamath_mixed` / `metamath_followup` (`qc`, `qc_roc`); Cell 12 print "P17.2" |
| Prediction log (Step 17, P17.3) | QC on random natural campaigns; on human GSM8K questions | Partial (0.11; 0.58) | Cell 8 rows `random` / `gsm8k_human` (`qc`); Cell 12 print "P17.3" |
| Prediction log (Step 17, P17.4) | rolling 4-week populations (A = 1,000): held-out false alarms, two-scale; QC | Fail (24%, 69%) | Cell 10 `FPRP_CI` ("held-out FPR (position-clustered)" print); Cell 12 print "P17.4/5" (`fpr` values) |
| Prediction log (Step 17, P17.5) | rolling populations, K = 10 (1%): power of every detector | Pass (≤ 0.08) | Cell 10 `RESP` rows K=10 ("[pop] K=10" prints); Cell 12 print "P17.4/5" |
| §5 'Scale: selection fails before the barrier does' | colluder shares at which no detector was both calibrated and powerful | 1–2.5% | Cell 10 rolling populations, K = 10 and 25 of A = 1,000 (plausibly combined with Step 15 Part P; summary statement, not isolated) |

#### Step 18 — [`step18_qc_repair_scale_evals.py`](experiments/02_development_weeks/step18_qc_repair_scale_evals.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| Prediction log (Step 18 rows) | locked predictions P18.1-P18.6 | Pass (conformal 0.0%; lagged 15%); Fail (0.20 vs 0.21; AUC 0.86); Fail (12%; 0.24); Partial (0.84; 0.35); Pass (0.05 vs 0.35); Pass (79–80%; MMLU 2–7%) | final cell "Predictions, figure, VERDICT REPORT", prints "P18.1" ... "P18.6" |
| §5 'It detects topic shift, not extraction'; App. D.3 | benign groups sweeping the GSM8K test set from 3 or 10 accounts: two-scale alarm rate | 79–80% of weeks | Part Q/E weekly loop (Cell 8), rows `eval_gsm8k(3x300)` / `eval_gsm8k(10x100)`; print "P18.6 benign pipeline" |
| App. D.3 | broad MMLU sweeps: two-scale alarm rate | 2–7% | Part Q/E weekly loop (Cell 8), rows `eval_mmlu(3x300)` / `eval_mmlu(10x100)`; print "P18.6 benign pipeline" |
| App. F 'Query-level pooling' | null-week false alarms of per-cell lagged scoring with shrinkage (`qc_ewma`) and of gating behind the two-scale test (`qc_gate`) | 15%; 12% | Cell 8, `FPR[...]` print "[weekly] held-out FPR" |
| App. F 'Query-level pooling' | full-history conformal threshold (`qc_conf`): reached calibration; power on padded campaigns | 0.20 of the time—no better than the two-scale test (Prediction log: 0.20 vs 0.21) | Cell 8 `FPR["qc_conf"]` and row `metamath_mixed`; print "P18.2/3 qc_conf" |
| App. D.4 ('AUC up to 0.86'); App. F 'Query-level pooling' (AUC 0.82–0.86, range shared with Step 17) | QC repairs rank padded campaigns | AUC up to 0.86 | Cell 8, `qc_ewma_roc` AUC on `metamath_mixed` / `metamath_followup`; print "P18.2/3 ... AUC mixed" |
| Table 6, col "QC" (plausibly from this run: the column is the repaired per-cell lagged QC and has no TPR at 5%) | AUC / TPR at 1%: MetaMathQA; human GSM8K; half natural mixed; real follow-up; random natural | 0.94 / 0.64; 1.00 / 0.93; 0.86 / 0.31; 0.85 / 0.27; 0.52 / 0.10 | Cell 8 `roc(...)` for `qc_ewma`; verdict "ROC:" lines `qc_ewma:auc=...,tpr1=...` |
| App. D.5 'Smaller colluder shares'; Prediction log (Step 18, P18.4) | two-scale oracle detection, campaign fixed at 1,640 queries, A = 250 / 500 / 1,000 | 0.84, 0.60, 0.35 at 20%, 10%, 5% | Part S (Cell 10), `RESS` field `oracle`; print "P18.4/5" |
| App. D.5 'Smaller colluder shares' | deployed false alarms in multi-week populations | 14–90% | Part S (Cell 10), `RESS` field `fpr`; "[S]" verdict lines |
| Prediction log (Step 18, P18.5) | at A = 1,000: deployed vs oracle two-scale power | 0.05 vs 0.35 | Part S (Cell 10), `RESS` fields `deployed` / `oracle`; print "P18.4/5 A=1000" |

#### Step 20 — [`step20_baselines_adaptive_scaling.py`](experiments/02_development_weeks/step20_baselines_adaptive_scaling.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| Prediction log (Step 20 rows) | locked predictions P20.1-P20.9 | Fail (0.00 vs 0.81, 0.95); Pass (0.8%; others 2.5–12%); Fail (0.00 vs 0.24); Fail (0.31; 0.00); Fail (slope −0.17 vs +0.62); Fail (502 vs 2,081); Fail (ξ = −0.16); Fail (1 of 4 inside band); Pass (all of it) | final cell "Predictions, figure, VERDICT REPORT", prints "P20.1" ... "P20.9" |
| §5 'Attacks on the monitor'; App. D.6 'Optimising against the statistic' | weekly two-scale detection, prompts ordered greedily against the statistic, no padding | from 0.80 to 0.31 | Part B/D weekly loop (Cell 8): 0.31 = deployed two-scale on `metamath_adaptive` (print "P20.4"); 0.80 plausibly the `metamath_unique` row of the same run |
| §5 'It detects topic shift, not extraction'; App. D.3 | dataset-versus-chat classifier and topic-concentration statistic on the selected group | ranked campaigns moderately (AUC 0.69–0.84) but caught none at a 1% false-alarm rate | Part B/D weekly loop (Cell 8), `style_grp` / `topic_conc` `_roc` (auc, tpr1); verdict "oracle" lines; print "P20.1" |
| Table 10, cols "Null median", "Null 99%", "Detector TPR at 1%" (A = 250 / 1,000 / 2,500 / 5,000 / 10,000) | two-scale null median; null 99%; detector oracle TPR at 1% | 1.27, 1.21, 1.05, 0.94, 0.61; 3.28, 3.16, 4.35, 3.18, 2.95; 1.00, 0.80, 0.42, 0.15, 0.00 | Part L (Cell 10), print "[L] A=...: null median ..., 99% ... oracle TPR@1%" (`LAW`) |
| Table 10, col "True colluder set above threshold"; Fig. 5 series "true colluders scored (oracle)" | true colluder set scored against the same null threshold | 1.00 at every A | Part L (Cell 10), `EVT` field `tpr_oracle_group`; print "[EVT] ... true colluder set" |
| Fig. 5 series "top eigenvector only"; §5 'Scale: selection fails before the barrier does'; App. D.5 'Up to 10^4 accounts' | two-scale power as colluder share goes from 20% to 0.5% | fell from 1.00 to 0.00 | Part L (Cell 10), `LAW` field `tpr` / `EVT` field `tpr_selected` |
| App. D.5 'Up to 10^4 accounts' | null threshold of the two-scale score across A | stayed flat between 2.9 and 4.3 | Part L (Cell 10), `LAW` field `q99`; print "[L] ... 99%" |

#### Step 21 — [`step21_sparse_boundary_adversary.py`](experiments/02_development_weeks/step21_sparse_boundary_adversary.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| Prediction log (Step 21 rows) | locked predictions P21.1-P21.7 | Fail (0.03 vs 0.03); Pass (1.00; random-group null); Pass (uninformative: both ≈ 0); Fail (0.035 vs 0.447); Pass (0.92; 0%); Pass (0.95 to 0.12); Pass (0.88 vs 0.70) | final cell "Predictions, figure, VERDICT REPORT", prints "P21.1/2" ... "P21.7" |
| §5 'Attacks on the monitor'; App. D.6 'Choosing who gets flagged' | sacrificial scheduling (30% most distinctive queries on five accounts): detection on C; other colluders in the flagged group | 0.92; none of the other 45 colluders | Part X, `SCHED["sacrificial"]` (`paired`, `recall_nonsac`); print "[X] sacrificial" |
| §5 'Attacks on the monitor'; App. D.6 'Teaching the monitor that the campaign is normal' | campaign ramped through the eight calibration weeks vs abrupt: deployed two-scale detection | from 0.95 to 0.12 | Part T, `TEMP["abrupt"]` / `TEMP["ramped"]`; print "[T] deployed two-scale paired detection" |
| App. D.2 'Its edge is calibration, not ranking' | oracle TPR at 10−3 false alarms, MetaMathQA: two-scale; global stream; MMD | 0.88; 0.70; 0.39 | Part R, `LOWFPR[("metamath_unique", d)][1e-3]`; print "[R] metamath_unique ..." |
| App. F 'Scale' paragraph; App. D.5 'Why this is not yet a production result' (sparse scans did not help) | sparse scans (truncated power iteration up to 100 iterations x 20 restarts; greedy forward scan) vs an oracle told the colluders | caught 0–4% of campaigns at every colluder fraction from 1% to 28%; oracle caught all of them | Part S, `SPARSE_ROWS` (`tp_*`, `greedy_*`, `oracle`); print "[S] weekly/pooled K=..." |
| App. F 'Statistics that failed' | mixing a fraction f of natural prompts (lam here): oracle two-scale detection; MMD2 where it crosses 0.5; benign top groups' distance | 0.81 (f = 0) to 0.00 (f = 0.75); about 0.035; median 0.145 | Part B, `BOUND` (`tpr`, `eps`, `eps_star`) and `GEPS`; print "[B] weekly: lam=..." |

#### Step 22 — [`step22_hollowed_profile_sequential.py`](experiments/02_development_weeks/step22_hollowed_profile_sequential.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| App. Table 8 (col "Selection on C"; rows MetaMathQA, Half natural mixed, Real follow-up, Human GSM8K questions, Random natural prompts, GSM8K evaluation sweep 3 accounts) | oracle TPR at matched 1% (AUC) | 0.86 (0.92); 0.21 (0.66); 0.23 (0.66); 0.93 (0.97); 0.02 (0.52); 0.75 (0.87) | Cell 10, `row["tsC_roc"]` from `roc(...)`; verdict-cell "oracle tsC:auc=...,tpr1=..." lines |
| App. Table 8 (col "Selection on R", same rows) | oracle TPR at matched 1% (AUC) | 1.00 (1.00); 0.94 (1.00); 0.93 (1.00); 1.00 (1.00); 0.00 (0.51); 0.34 (0.61) | Cell 10, `row["tsR_roc"]` |
| App. Table 8 (col "R on topic cores", same rows) | oracle TPR at matched 1% (AUC) | 1.00 (1.00); 0.95 (1.00); 0.93 (1.00); 1.00 (1.00); 0.00 (0.48); 0.02 (0.54) | Cell 10, `row["coreR_roc"]` |
| App. Table 8 caption; App. F 'Hollowing' | deployed false alarms for C / R / core variant | 0.8% / 8.4% / 7.6% | Cell 9, `FPR["tsC"]`, `FPR["tsR"]`, `FPR["coreR"]` ("[weekly] held-out FPR" print) |
| Prediction log (Step 22, "Hollowing keeps calibration (≤ 1.5%) and power (≥ 0.75)") | hollowed (R) false alarms; MetaMathQA deployed power | Partial (8.4%; 0.90) | Cell 13 print "P22.4" (`FPR["tsR"]`, deployed `tsR` in `RES`) |
| Prediction log (Step 22, "Hollowing: real/shuffled λmax (R) ≤ 2") | real / shuffled λmax(R) | Partial (0.35: common drift mode) | Cell 9, `DIAG["ratioR"]` ("[C1]" print); print "P22.1" |
| Prediction log (Step 22, "Hollowed selection halves bulk-user groups") | templated bulk-user share of selected groups, C to R | Fail (0.08 to 0.06) | Cell 9, `DIAG["bulkC"]`, `DIAG["bulkR"]`; print "P22.2" |
| Prediction log (Step 22, "Group MMD on hollowed groups AUC ≥ 0.6") | group-MMD AUC on MetaMathQA, hollowed groups | Fail (0.18) | Cell 10, `row["gmmdR_roc"]["auc"]`; print "P22.3" |
| Prediction log (Step 22, "Profile score ≥ two-scale + 0.1 under padding"); App. F 'Statistics that failed' (profile "detected nothing under padding") | oracle TPR at 1% on mixed padding, profile vs two-scale | Fail (0.00 vs 0.21) | Cell 10, `profC_roc` / `tsC_roc` `tpr1` for metamath_mixed; print "P22.5" |
| Prediction log (Step 22, "Profile score flags GSM8K sweeps ≥ 50%") | deployed profile score on GSM8K sweep (3 x 300) | Fail (43%) | Cell 13 print "P22.6" (`RES` deployed `profC`) |
| Prediction log (Step 22, "Weeks to detection within 2× of ln(1/α)/D") | martingale median weeks vs ln(1/α)/D | Partial (15 vs 14.7, 16 vs 13.3; Proposition 3 holds unpadded at the floor) | Cell 11, `SEQ[src]["median_martingale"]` vs `SEQ[src]["predicted"]` ("[SEQ]" print; "P22.7/8") |
| Prediction log (Step 22, "Martingale median ≤ 6 weeks under padding, all-time FA ≤ 1%") | martingale weeks under padding; null-run all-time false alarms | Fail (15 weeks; FA 0/33) | Cell 11, `SEQ["metamath_mixed"]["median_martingale"]`; `FA_M` over `len(starts)` null runs |
| Table 2 (row "Two-scale / Half padding", col "Anytime-valid", development weeks) | week of first anytime-valid alarm (GSM8K gain) | week 15 (15.5) | Cell 11 martingale median (metamath_mixed); the gain in parentheses combines this with the Step 19 gain curves (not computed here) |
| Table 2 (row "Two-scale / Dataset", col "Anytime-valid", development weeks); App. B 'Sequential detection in practice' ("about three weeks") | week of first anytime-valid alarm (GSM8K gain) | week 3 (12.7) | plausibly Cell 11 `SEQ["metamath_unique"]["median_martingale"]` (the conformal "floor" of Prediction log); gain from Step 19 |
| Prediction log, corrections ('Sequential prediction') | slow-down factor predicted by Proposition 3 | about 18 | not isolated (plausibly the ratio of `SEQ[...]["predicted"]` for padded vs unpadded sources) |
| Prediction log (Step 22, "Core-plus-halo calibrated power ≥ 0.4") | deployed power on mixed padding | Fail (halo 0.02; hollowed cores 0.90 at 7.6%) | Cell 13 print "P22.9" (`RES` `halo`, `coreR`; `FPR["coreR"]`) |
| App. F 'Statistics that failed' | core-plus-halo statistic AUC | at chance (AUC 0.48) | Cell 10, `row["halo_roc"]["auc"]` ("[W] ... AUC ... halo=" print); campaign row not stated in the paper |
| Prediction log (Step 22, "Prefilter keeps ≥ 95% of colluders at 10% pass"); App. F 'Statistics that failed' ("kept none of the colluders") | colluder retention of the behavioural prefilter | Fail (0%) | Cell 12, `RETAIN` ("[PF]" print); print "P22.10" |
| Prediction log (Step 22, "Prefiltering restores power at 5% share") | filtered oracle TPR at 1% | Fail (0.00) | Cell 12, `PF["filtered"]["tpr"]`; print "P22.11" |

#### Step 23 — [`step23_hollowed_calibration_scale.py`](experiments/02_development_weeks/step23_hollowed_calibration_scale.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| §5 'Padding, and a fix that looks at directions'; App. D.4; Prediction log (Step 23, "Common-mode removal: real/shuffled λmax (Rc) ≥ 1") | top eigenvalue of Rc, real / shuffled weeks | 1.14 (Prediction log: Pass (1.14)) | Cell 9, `RATIO` ("[C1]" print); print "P23.1" |
| §5 'Padding, and a fix that looks at directions'; App. D.4; App. Table 9 caption; Prediction log (Step 23, "Conformal hollowed selection: false alarms ≤ 2%") | held-out false alarms, conformal thresholds (Rc and C) | no false alarm in 960 windows; Pass (0/960 windows; week-level bound 12%) | Cell 9, `FPR[("conf","tsRc")]`, `FPR[("conf","tsC")]`; print "P23.2" |
| App. Table 9 (top, row "Selection on Rc"; cols MetaMathQA, Mixed, Follow-up, Random, Sweep 3 accts, Sweep 10 accts) | weekly deployed detection (conformal) | 1.00; 0.48; 0.46; 0.00; 0.42; 0.97 | Cell 9, `RES` key `"conf/tsRc"` ("[W]" print) |
| App. Table 9 (top, row "Selection on C", same cols) | weekly deployed detection (conformal) | 0.82; 0.00; 0.02; 0.00; 0.80; 0.82 | Cell 9, `RES` key `"conf/tsC"` |
| §5 'Padding, and a fix that looks at directions'; App. D.4; Table 3 caption (development-week hollowed values); Prediction log (Step 23, "Calibrated hollowed selection ≥ 0.6 under padding": Fail (0.48)) | hollowed vs C selection on padded campaigns; hollowed on dataset | 0.46–0.48 against 0.00–0.02; dataset 1.00 | Cell 9, `"conf/tsRc"` / `"conf/tsC"` for metamath_mixed, metamath_followup, metamath_unique; print "P23.3" |
| Table 2 (row "Hollowed", col "Weekly detection", development weeks); App. Table 15 (development weeks, col "Hollowed") | weekly detection, dataset; half padding | 1.00; 0.44–0.48 | plausibly Cell 9 `"conf/tsRc"` (this script gives 1.00 and 0.46–0.48; the 0.44 end plausibly from the Steps 27–31 run) |
| App. D.4; Prediction log (Step 23, "Sweeps: 3 accounts ≤ 0.4, 10 accounts ≥ 0.5") | benign GSM8K evaluation sweep flagged (Rc, conformal) | 0.97 (ten accounts), 0.42 (three accounts); Partial (0.42; 0.97) | Cell 9, `"conf/tsRc"` for `eval_gsm8k(10x100)` and `eval_gsm8k(3x300)`; print "P23.5" |
| App. Table 9 (bottom, pooled oracle TPR at 1%, A = 250 / 1,000 / 2,500 / 5,000 / 10,000; rows "C, unpadded", "Rc, unpadded", "Rc, mixed padding") | oracle TPR at 1% | 0.97 0.75 0.28 0.12 0.05; 1.00 1.00 0.00 0.00 0.03; 1.00 0.10 0.00 0.00 0.03 | Cell 10, `SCALE` rows, `row["tsC"]`, `row["tsRc"]` ("[S]" print) |
| App. Table 9 (bottom, row "Colluders in Rc's group"); Prediction log (Step 23, "Hollowed selection ≥ 0.5 at 1% colluders") | share of colluders in the Rc-selected group; Rc TPR at A = 5,000 | 1.00 1.00 0.00 0.00 0.00; Fail (0.00; group has no colluders) | Cell 10, `row["recall_Rc"]`, `row["tsRc"]`; print "P23.4" |
| App. Table 15 (row "Pooled, 1,000 accounts, 5% colluders / Dataset", col "Two-scale") | oracle detection | 0.75–0.80 | plausibly Cell 10 `SCALE` `tsC` at A = 1,000 (0.75; the 0.80 is App. Table 10 from a separate run) |

#### Step 27–31 — [`step27_31_bulk_edge_multioutlier_tilted.py`](experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| App. Table 11 (cols "Top eigenvector" unpadded / padded; A = 250, 1,000, 2,500, 5,000, 10,000) | oracle TPR at matched 1% | 1.00/1.00; 1.00/0.07; 0.00/0.00; 0.00/0.00; 0.03/0.03 | Cell 12, `SC` rows, `row["tsRc"]` for metamath_unique / metamath_mixed ("[27/28]" print) |
| App. Table 11 (cols "Tmax" unpadded / padded, same A) | oracle TPR at matched 1% | 1.00/1.00; 1.00/1.00; 1.00/0.83; 0.93/0.77; 1.00/0.17 | Cell 12, `row["Tmax"]` |
| App. Table 11 (col "Colluder rank (unp./pad.)"); App. F 'Scale' ("at rank 1–8 rather than first") | rank of the Rc eigenvector most aligned with the colluders | 0/0; 0/1; 1/2; 2/3; 3/8 | Cell 12, `row["jstar_med"]` |
| App. Table 11 (col "ρ vs needed"); App. F 'Scale' ("pairwise cosine 0.49–0.61 against a requirement of at most 0.082") | colluders' mean pairwise cosine vs BBP requirement | 0.49 vs 0.013; 0.58 vs 0.026; 0.58 vs 0.041; 0.60 vs 0.058; 0.61 vs 0.082 | Cell 12, `row["rho"]`, `row["bbp_needed"]` |
| §5 'Scale: selection fails before the barrier does'; App. D.5 | multi-outlier selection (Tmax), oracle thresholds | 93–100% of unpadded campaigns up to 10^4 accounts; padded 77–100% up to 5,000 accounts and 17% at 10^4 | Cell 12, `row["Tmax"]` |
| Fig. 5 | curves "every outlier eigenvector", "top eigenvector only", "true colluders scored (oracle)" vs accounts monitored | (plotted curves) | plausibly Cell 12 `row["Tmax"]`, `row["tsRc"]`, `row["orc"]` (unpadded); not isolated: the "top eigenvector only" curve may be the two-scale (C) curve of App. Table 10 |
| App. Table 15 (rows "Pooled, 1,000 accounts" and "Pooled, 10,000 accounts", cols "Hollowed" and "Multi-outlier", Dataset / Half padding) | oracle detection | Hollowed 1.00 / 0.07 and 0.03 / 0.03; Multi-outlier 1.00 / 1.00 and 1.00 / 0.17 | Cell 12, `row["tsRc"]`, `row["Tmax"]` |
| App. F 'Scale'; Prediction log (Step 27, "r ≥ 1,500 lifts unpadded TPR at A = 2,500 to ≥ 0.5") | top-eigenvector selection at A = 2,500, 600 to 1,500 components | 0.00 to 0.57 | Cell 12, `HI["metamath_unique"]["tsRc"]` ("[27d]" print; "P27d") |
| App. F 'Scale'; Prediction log (Step 28, "k(10,000) ≤ 5 k(1,000)") | number of retained outlier eigenvectors | cap of 20 from 1,000 accounts upward; Pass (uninformative: cap of 20 reached) | Cell 12, `row["k_med"]`; print "P28c" |
| Prediction log (Step 27, "Campaign below the bulk edge at A ≥ 2,500") | colluder eigenvalue vs null 99% λmax | Pass by the letter (uninformative: compared with null λmax; campaign far above the edge) | Cell 12, `row["above_edge"]`; print "P27a/b" |
| Prediction log (Step 27, "Null-selected groups not coordinated") | null-selected group cosine vs random groups | Fail (cosine 0.43–0.57 vs 0.00; selection-biased) | Cell 12, `row["cos_sel"]`, `row["cos_rand"]`; print "P27a/b" |
| Prediction log (Step 27, "Colluder cosine 0.025–0.045; padded ratio 0.4–0.6") | colluder cosine (unpadded); padded/unpadded ratio | Partial (0.57; 0.51) | print "P27c" (mean of `row["rho"]`) |
| Prediction log (Step 28, "Tmax calibrated (conformal)") | held-out false alarms of Tmax | Pass (0/960 windows; week-level bound 12%) | Cell 10, `FPR["Tmax"]`; print "P28a" |
| Prediction log (Step 28, "Multi-outlier ≥ top eigenvector + 0.2 at A = 2,500") | oracle TPR, Tmax vs top eigenvector | Pass (1.00 vs 0.00) | Cell 12, `SC` row A = 2,500; print "P28b" |
| Prediction log (Step 28, "GEV predicts Tmax threshold at ≥ 3 of 4 sizes") | GEV-predicted 99% Tmax threshold inside band | Fail (2 of 4) | Cell 12, `GEV` ("[28d]" print); print "P28d" |
| §4 (discussion after Prop. 4); App. B 'The least-favourable campaign'; Prediction log (Step 29, "Task region rare: p ≤ 5%") | benign share of the task region | p = 1.48%; Pass (1.48%) | Cell 9, `P_TASK` ("[29a]" print; "P29a"); the "about 51" extra queries over twelve weeks of 1,640 is derived from p analytically in the paper, not computed here |
| Prediction log (Step 29, "Tilted campaign evades: two-scale ≤ 0.10, hollowed ≤ 0.25") | weekly deployed detection of the tilted (c = 0.5) campaign | Pass (0.00; 0.01) | Cell 10, `RES` row tilted_c05, `tsC` / `tsRc`; print "P29b" |
| Table 2 (row "Any / Least detectable", development weeks); App. G.2 ("at most 0.01 on development weeks"); App. Table 15 (development weeks, "Least detectable": Two-scale / Hollowed / Multi-outlier); §6 'What the attacker gets' ('The covert end': evaded every detector) | weekly detection of the least-detectable campaign | ≈0.01; 0.00 / 0.01 / 0.01 | Cell 10, `RES` row tilted_c05 (`tsC`, `tsRc`, `Tmax`) |
| Prediction log (Step 29, "Detectors realise ≤ 1% of Nw d(c∥p)") | share of available nats per week realised by the best detector | Pass (0.03%) | Cell 10, `best / AVAIL` ("[29c]" print; "P29c") |
| Table 2 (row "Multi-outlier", col "Weekly detection", development weeks); Table 3 caption; App. Table 15 (development weeks, col "Multi-outlier") | weekly detection, dataset; half padding | 1.00; 0.36–0.42 | plausibly Cell 10, `RES` `Tmax` for metamath_unique; metamath_mixed / metamath_followup |
| Table 2 (row "Hollowed / Half padding", col "Weekly detection", development weeks); App. Table 15 | weekly detection under half padding | 0.44–0.48 | plausibly in part Cell 10, `RES` `tsRc` for metamath_mixed / metamath_followup, combined with Step 23 (0.46–0.48) |
| Prediction log (Step 30, "Cell drift low-rank (≥ 50% in q ≤ 5)"); App. F 'Query-level pooling' ("two factors carried only 31% of the variance") | variance share of the first q cell-drift factors | Fail (31%) | Cell 11, median `QCR[b]["share"]` ("[30a]" print; "P30a") |
| Prediction log (Step 30, "Factor adjustment calibrates (≤ 2%)"); App. F 'Query-level pooling' ("a factor model of cell drift on 23%") | held-out false alarms, factor-adjusted query-cluster | Fail (23%) | Cell 11, `FPRQ["qc_fac"]` ("[30b/d]" print; "P30b") |
| Prediction log (Step 30, "Factor QC ≥ two-scale + 0.1 under padding") | mixed padding, factor QC vs two-scale | Partial (0.48 vs 0.01 at 23% false alarms) | Cell 11, `QRES["metamath_mixed"]["qc_fac"]` vs Cell 10 `tsC`; print "P30c" |
| Prediction log (Step 30, "Factor model ≥ 5 points below forecast") | false alarms, factor model vs EWMA forecast | Fail (23% vs 14.5%) | Cell 11, `FPRQ["qc_fac"]` vs `FPRQ["qc_fc"]`; print "P30d" |
| Prediction log (Step 31, "Two-sided prefilter keeps ≥ 80% of colluders" and "It keeps evaluation sweeps too"); App. F 'Statistics that failed' ("kept 13% of colluders and every benign evaluation account") | retention at 20% benign pass rate, colluders; GSM8K-sweep accounts | Fail (13%); Pass (100%) | Cell 13, `RET_COL`, `RET_SWP` ("[31]" print; "P31a") |
| §7 'The test that matters: one frozen evaluation on untouched weeks'; App. I | multi-outlier design constants | bulk-edge factor of 1.05, minimum group spread of 5 accounts, cap of 20 eigenvectors | Cell 8, `stats27` (`1.05 * e_hat`, participation number `>= 5`, `min(20, ...)`) |

### `experiments/03_capability_frontier/`

#### Step 14 — [`step14_mechanism_distillation.py`](experiments/03_capability_frontier/step14_mechanism_distillation.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| App. Table 12, row "MetaMathQA, one prompt per source problem", cols "Two-scale" / "Global" / "MMD" | paired detection at 1% (weekly, K = 50, 1,641 queries) | 0.85 [0.74, 0.94] / 0.65 / 0.56 | Cell 9 `run_block(PART_M, SRC_M, "weekly")`, source `metamath_unique` (`twoscale`, `glob`, `mmd`); Cell 11 "[M3]" lines |
| App. Table 12, row "Half natural prompts mixed into every account" | paired detection: two-scale / global / MMD | 0.21 [0.10, 0.37] / 0.12 / 0.08 | Cell 9, source `metamath_pad50`; Cell 11 "[M3]" lines |
| App. Table 12, row "Each prompt followed by a real follow-up turn" | paired detection: two-scale / global / MMD | 0.24 [0.11, 0.38] / 0.12 / 0.08 | Cell 9, source `metamath_followup`; Cell 11 "[M3]" lines |
| App. Table 12, row "Natural WildChat cluster" | paired detection: two-scale / global / MMD | 0.01 / 0.01 / 0.07 | Cell 9, source `wild_cluster`; Cell 11 "[M3]" lines |
| App. Table 12, caption | base Qwen2.5-0.5B GSM8K accuracy (first 500 test problems) | 34.4% | Cell 10, `ACC[("base", 0)]` |
| App. Table 12, col "GSM8K gain, 1,641 queries" | gain: one prompt per problem / half natural / follow-up | +11.4 / +12.0 / +13.4 | Cell 10 loop over `CONFIGS` (budget `BUDGETS[0]`); Cell 11 "[D]" lines |
| App. Table 12, col "GSM8K gain, 7,550 queries" | gain: one prompt per problem / half natural / natural cluster | +12.6 / +11.0 / +6.0 | Cell 10 loop over `CONFIGS` (budget `BUDGETS[1]`); Cell 11 "[D]" lines |
| §5 "Padding, and a fix that looks at directions"; App. D.4 | two-scale detection cut by padding / follow-up mimicry | from 0.85 to 0.21–0.24 | Cell 9 (`metamath_unique` vs `metamath_pad50`, `metamath_followup`) |
| Corrections (docs/corrections.md), "Corrections made during the study" ("Padding construction") | two-scale detection with padding mixed into every account | 0.21 when mixed | Cell 9, source `metamath_pad50` (construction: Cell 7 `campaign_points`) |
| Table 2, Table 15 (row "Half padding", two-scale, development weeks); Table 3 caption | two-scale weekly detection of half padding, development weeks | 0.21 | plausibly Cell 9, source `metamath_pad50` (the same value is also logged in Steps 18 and 22) |
| App. D ("A note on uncertainty"); Table 2 / Table 15 ("0.79–0.85") | one of four runs of two-scale detection of MetaMathQA campaigns | 0.85 | plausibly Cell 9, source `metamath_unique` |
| App. F "Mechanism probes" (text); Prediction log (Step 14), "Removing follow-ups closes ≥ 50% of the clumping gap" | P14.1 (fine-scale clumping, all turns vs first turns) | Fail (moved the wrong way) | Cell 8 (`M1`); Cell 11 "P14.1" print |
| Prediction log (Step 14), "Follow-up mimicry cuts two-scale power by ≥ 0.3" | P14.2 | Pass (0.85 to 0.24) | Cell 11 "P14.2" print |
| Prediction log (Step 14), "Unique at 7,550 gains ≥ 5 points; natural ≤ 1" | P14.3 | Partial (+12.6; natural +6.0) | Cell 11 "P14.3" print |
| Prediction log (Step 14), "Evading variants gain ≤ 60% of unique" | P14.4 (ratio of gains at 1,641 queries) | Fail (105% and 118%) | Cell 11 "P14.4" print (ratio) |

#### Step 16 — [`step16_price_of_stealth.py`](experiments/03_capability_frontier/step16_price_of_stealth.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| App. Table 13, col "All math" (Qwen2.5-0.5B, three seeds, 400 problems) | GSM8K gain at 100 / 300 / 900 queries | +9.8 ± 2.1 / +7.5 ± 0.5 / +8.6 ± 1.2 | Cell 6, `summary("unique", n)`; print "n=...: gains" |
| App. Table 13, col "Half natural, mixed" | GSM8K gain at 100 / 300 / 900 queries | +9.8 ± 1.8 / +10.2 ± 2.1 / +10.1 ± 1.0 | Cell 6, `summary("mixed", n)` |
| App. Table 13, col "Camouflaged (natural-looking math)" | GSM8K gain at 100 / 300 / 900 queries | +9.3 ± 0.7 / +8.7 ± 2.3 / +11.2 ± 1.9 | Cell 6, `summary("camouflage", n)` |
| §6 "Two regimes"; App. G.1; Fig. 6b (elicitation, Qwen2.5-0.5B, three seeds) | gain from 100 math prompts, and no more from 900; padding costs nothing measurable | 9.8 points | Cell 6 (`G`); figure in Cell 8 |
| Prediction log (Step 16), "All-math gains grow ≥ 3 points from 100 to 900 queries" | P16.1 | Fail (+9.8 to +8.6) | Cell 8 "P16.1" print |
| Prediction log (Step 16), "Mixed gains ≤ all-math − 2 at 300 and 900" | P16.2 (paired mixed − unique) | Fail (+2.7, +1.5) | Cell 6 `paired_diff("mixed", "unique", n)`; Cell 8 "P16.2" print |
| Prediction log (Step 16), "Camouflage gains ≤ 80% of all-math at 900" | P16.3 | Fail (130%) | Cell 8 "P16.3" print (ratio) |
| Prediction log (Step 16), "MATH-500, 1.5B student: price of stealth" | P16.4 | Untestable (not run) | Cell 7 (Part H; `RUN_MATH = False`); MATH-500 was measured in Step 19 Part B instead |

#### Step 19 — [`step19_nonsaturating_distillation.py`](experiments/03_capability_frontier/step19_nonsaturating_distillation.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| App. Table 14 caption; §6 "Two regimes"; App. G.1; Fig. 6a caption; Prediction log (Step 19), "SmolLM2-360M lacks the skill" | base SmolLM2-360M GSM8K accuracy (first 400 problems); P19.1 | 3.5%; Pass (3.5%) | Cell 7, `BASE_A`; Cell 9 "P19.1" print |
| App. G.2 ("the student solves 14 of 400 problems at baseline") | baseline problems solved | 14 of 400 | Cell 7, `BASE_A` (fraction of 400) |
| App. Table 14, row "SmolLM2-360M, GSM8K", col "All math"; §6 "Two regimes"; App. G.1; Fig. 6a | gain at 1,000 / 3,000 / 10,000 queries | +7.5 ± 0.0 / +10.4 ± 0.1 / +15.9 ± 0.9 (text: 7.5, 10.4 and 15.9 points) | Cell 7, summary print "[A] n=...: unique" |
| App. Table 14, row "SmolLM2-360M, GSM8K", col "Half natural, mixed"; Fig. 6a | gain at 1,000 / 3,000 / 10,000 queries | +3.9 ± 0.6 / +7.6 ± 0.6 / +12.1 ± 1.1 | Cell 7, summary print "[A] n=...: mixed" |
| App. Table 14, col "Paired cost"; §6 "Two regimes"; App. G.1 | cost of padding half of every campaign (and share of the gain) | −3.6 / −2.8 / −3.8 (text: 3.6, 2.8 and 3.8 points; 48%, 27% and 24% of the gain) | Cell 7, "paired mixed−unique"; the percentages are cost / all-math gain (not printed) |
| Fig. 1a (points "dataset prompts", "half padding", 3,000 queries) | capability gained at 3,000 queries | plotted (= App. Table 14 +10.4 ± 0.1 / +7.6 ± 0.6) | Cell 7, n = 3000 |
| §6 "The covert end"; App. G.2 | dataset-campaign gain at 3,000 queries used as the reference (tilted 2.2 against 10.4) | 10.4 | Cell 7, unique n = 3000 (the tilted 2.2 comes from Step 29d) |
| Table 2 (gains in parentheses, cols "First weekly alarm" / "Anytime-valid") | GSM8K gain held at the week of first detection, 1,640 queries a week | 8.8, 12.7 (dataset); 9.5, 15.5, 7.9, 5.6 (half padding) | not isolated: derived from this script's Part A unique / mixed gain curves (Cell 7) at the cumulative queries of the detection week; that step and the detection weeks are not computed here |
| App. Table 14, row "Qwen2.5-1.5B, MATH-500" and caption | base accuracy; gain all math / mixed / paired cost at 1,000 and 4,000 queries | base 22.8%; +12.0 / +11.4 / −0.6 (1,000); +11.6 / +9.6 / −2.0 (4,000) | Cell 8 (Part B), `BASE_B`, `RB` |
| §6 "Two regimes"; App. G.1 | Qwen2.5-1.5B on MATH-500 | flat from 1,000 to 4,000 queries | Cell 8 (Part B) |
| Prediction log (Step 19), "Extraction regime: all-math gains rise ≥ 5 points, 1k to 10k" | P19.2 | Pass (+7.5 to +15.9) | Cell 9 "P19.2" print |
| Prediction log (Step 19), "Price of stealth: mixed ≤ all-math − 2 at 3k and 10k" | P19.3 | Pass (−2.8, −3.8) | Cell 9 "P19.3" print |
| Prediction log (Step 19), "MATH-500 (1.5B): all-math ≥ +3, mixed ≤ 70% of it" | P19.4 | Fail (+11.6; 83%) | Cell 9 "P19.4" print |
| Prediction log Table 16, rows "Extraction student" and "Elicitation students" (Qwen2.5-1.5B) | training settings | learning rate 5 × 10−5, effective batch 16 (micro-batch 4); micro-batch 2 with 4 accumulation steps | Cell 1 (`LR_A`, `BS`), Cell 6 `train_full`, Cell 8 `train_lora_lowmem` |

#### Step 29d — [`step29d_tilted_distillation.py`](experiments/03_capability_frontier/step29d_tilted_distillation.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| App. Table 14, row "SmolLM2-360M, tilted (least detectable)", col "Half natural, mixed" | tilted_c05 gain at 3,000 / 10,000 queries | +2.2 ± 0.2 / +1.5 ± 0.2 | Cell 7, summary print "[A] tilted_c05 n=..." |
| App. Table 14, tilted rows, col "Paired cost" | cost against the all-math campaign of the same size | −8.2 / −14.4 | not isolated: tilted gain here minus Step 19's all-math gain (+10.4, +15.9) |
| §6 "The covert end"; App. G.2; Fig. 1a and Fig. 6a ("least detectable (tilted)") | gain at 3,000 queries, against 10.4; covertness cost | 2.2 points; 79% of the gain | Cell 7, tilted_c05 n = 3000; 79% = 1 − 2.2/10.4 with Step 19's reference (not computed here) |
| §6 and App. G.2 "The frontier, in one sentence" | share of gain given up by the tilted campaign, one of the covert campaigns behind "gave up 58% or more" | 79% | 79% = 1 − 2.2/10.4, as above; the 58% bound comes from Step 32 |
| App. H (text) | on-task prompts in the attacker's pool; on-task share of the 10,000-query tilted campaign | 2,933; 29% on task | Cell 4, `len(ON_PAIRS)` in the "task region" print; the 29% follows from `tilted()` capping on-task draws at `len(ON_PAIRS)` |
| Prediction log (Step 29), "Base-rate on-task prompts distil (≥ 8 points at 10k)" | P29d | Fail (+1.5) | Cell 9 "P29d" print |

#### Step 32 — [`step32_written_attacker.py`](experiments/03_capability_frontier/step32_written_attacker.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| §6 'The covert end'; App. G.2; Prediction log (Step 32, "Iterative rewriting reaches AUC ≤ 0.65"); App. N 'Representation' | held-out discriminator AUC, real on-task prompts vs written rewrites (first pass, then two regeneration rounds) | "AUC 0.97–0.98" (§6); "AUC 0.98 … after two more rounds … 0.97" (G.2); "Fail (0.98, 0.97, 0.97)" (Prediction log); "AUC 0.98" (App. N) | Cell V, `crossfit(...)` → `VOI_AUCS`; prints "[W] round …" and "[W] discriminator AUC by round"; Cell Z "P32j" |
| §6 'The covert end'; App. G.2 | rewrites that passed as user-like (posterior < 0.5 and in S_v) | "7.9%"; "395 of 5,000 rewrites (7.9%)" | Cell V, `keep` → `VOI_KEEP`, `len(VOICED)`; print "[W] discriminator AUC by round … kept"; Cell Z "P32b" |
| Prediction log (Step 32, "Written prompts can be made user-like") | P32b, first run | "Fail (AUC 0.98; 4.3% pass)" | first run; same computation as above (Cell V; Cell Z "P32b") |
| §6 'The covert end'; App. G.2; Prediction log (Step 32, "Distribution-matched attacker evades and gains ≥ 6") | importance-resampled (distribution-matched) campaign `matched_c05`: detection and gain at 3,000 queries | "evaded every detector and gained 4.4 points—42% of the dataset gain"; "Fail (evades; +4.4, ESS 151)" | Cell W `[W] matched_c05` row (`RES`); Cell D `DRES[("matched_c05", sd)]`; Cell Z "P32k". The 42% is relative to Step 19's all-math +10.4 |
| §6 'The covert end'; App. G.2; Prediction log (P32k row) | effective sample size of the resampled rewrites | "effective sample of only 151 distinct rewrites"; "ESS 151" | Cell V, `ESS = 1/sum(wts**2)`; print "[W] … effective sample size"; Cell Z "P32k" |
| §6 'The frontier, in one sentence'; App. G.2 | share of the detectable campaign's gain given up by covert campaigns | "58% or more" | not isolated (summary): 58% = 1 − 4.4/10.4 uses this script's `matched_c05` gain (the best covert campaign); the others gave up more (e.g. 79% for Step 29d's tilted campaign) |
| §6; App. G.2; App. N 'Capability' | covert gains and their size in problems | "covert gains of 1.1–4.4 points are differences of 4–18 of 400 problems" (§6); "differences are 4–18 problems out of 400" (App. N) | Cell D (`tilted_v_c05` +1.1 to `matched_c05` +4.4; 400 = `S32["N_EVAL"]`) |
| App. G.2 | student baseline | "solves 14 of 400 problems at baseline" | Cell D, `BASE`; print "[D] base GSM8K" |
| App. G.2 | rewrites passing the first filter, distilled on their own | "The 213 rewrites that passed the first filter gained 3.5 points on their own" | first run of Step 32; plausibly the all-written campaign (`voiced_c1`, Cell D) |
| Prediction log (Step 32, "Written campaign keeps capability (≥ 6)") | P32d, `voiced_c05` gain at 3,000, first run | "Fail (+2.2; 213 unique prompts)" | first run; recomputed in Cell D `voiced_c05`; Cell Z "P32d" |
| Prediction log (Step 32, "Written campaign evades") | P32c, `voiced_c05` / `voiced_c1` weekly detection, first run | "Pass (0.00; 0.10 all on task)" | first run; recomputed in Cell W `[W] voiced_c05`, `[W] voiced_c1`; Cell Z "P32c" |
| Fig. 1a (points "written in users' voice", "written, distribution-matched") | hollowed weekly detection (development weeks) vs gain at 3,000 queries | plotted (caption: "kept at most 42% of the dataset campaign's gain") | Cell W `tsRc` for `voiced_c05`, `matched_c05`; Cell D gains |
| Prediction log (Step 32, "Value-defined task region rare") | P32a, benign share p_v of the value-defined region | "Pass (0.53%)" | Cell V, `P_V`; print "[V] value-defined region"; Cell Z "P32a" |
| App. H; Prediction log (Step 32, "Value-defined tilted campaign ≥ 4 points") | value-defined on-task pool prompts; `tilted_v_c05` gain at 3,000 | "only 972 on-task prompts … gained 1.1 points at 3,000 queries"; "Fail (+1.1)" | Cell V `ONV_KEYS` (print "[V] … attacker-pool on-task"); Cell D `tilted_v_c05`; Cell Z "P32e" |
| App. H; Prediction log (Step 32, "Personas evade query detectors, type count ≥ 0.9") | persona campaign: detection, type-count detector, gain | "evaded every query-level detector and gained 3.5 points from 1,891 queries … caught it 0.58 of the time"; "Fail (0.00; type count 0.58)" | Cell W `[W] persona` (`typecount`); Cell D `persona` (print "[D] persona … pairs"); Cell Z "P32l" |
| Cor. 1 (§4); App. B 'The least-favourable campaign'; Prediction log (Step 32, "Majority-on-task accounts rare") | benign share of majority-on-task accounts q | "q = 0.25%"; "Pass (0.25%)" (the paper derives "about 14 posing accounts" at A = 10^5 from it; K* is not computed here) | Cell W, `Q_TYPE`; print "[types]"; Cell Z "P32m" |
| Prediction log (Step 32, "Per-account topicality sees tilted campaigns") | P32f, held-out false alarms of the topicality HC test | "Fail (25% false alarms)" | Cell W, `FPR["topic"]`; print "[weekly] held-out FPR"; Cell Z "P32f" |
| Prediction log (Step 32, "Pooling artefact ≥ 2× random") | P32g, same-week share of null-selected groups vs random | "Partial (1.9–2.1×)" | Cell P, `same` / `expct`; print "[P] … same-week share"; Cell Z "P32g" |
| Prediction log (Step 32, "Per-week centring keeps multi-outlier") | P32h, per-week-centred Tmax oracle TPR at 10^4 | "Pass (1.00)" | Cell P, rows with `centre_week=True`; Cell Z "P32h" |
| Prediction log (Step 32, "Multi-outlier deployable at 2,500") | P32i, deployed Tmax false alarms | "Fail (30% false alarms)" | Cell P, `DEP["fpr"]["Tmax"]`; print "[P-dep]"; Cell Z "P32i/n" |
| §5 'Scale: selection fails before the barrier does'; App. D.5 'Why this is not yet a production result'; Prediction log (Step 32, "Multi-outlier deployable at 10^4") | best deployed calibration of multi-outlier selection | "5% false alarms, not 1%"; "at 5,000–7,552 account-weeks"; "Fail (5% false alarms at 7,552)" | Cell P, `DEP["fpr"]["Tmax"]` (A = 10,000 capped to 0.8 × test account-weeks); print "[P-dep]"; Cell Z "P32i/n" |
| App. D.5 'Why this is not yet a production result' | deployed multi-outlier power at 5,000–7,552 | "caught unpadded campaigns 0.53–1.00 of the time and padded ones 0.03–0.10" | Cell P, `DEP[src]["Tmax"]`; print "[P-dep] … unique … mixed" |
| App. K | deployed precision of the flagged group at 5,000–7,552 account-weeks | "(0.73 and 0.54)" | Cell P, `DEP["metamath_unique"]["precision"]`; print "[P-dep] … precision" (the "16–42 legitimate accounts" is derived in App. K) |
| Prediction log (Step 32, "≤ 100 benign accounts flagged per week at 10^5") | P32p, reviewer workload | "Pass (44–59 at A ≥ 5,000)" | Cell P, `DEP["benign_flagged_per_week_at_1e5"]`; print "[P-dep] … benign accounts flagged"; Cell Z "P32i/n" |
| §5 'Scale: selection fails before the barrier does'; App. D.5 'For providers'; App. L item 6; Prediction log (Step 32, "Linked 30%: expansion ≥ 0.8 …") | seeded hollowed expansion from 30% linked colluders (15 of 50, +50 accounts) at 2,500–10,000 | "recovered 96–100% of the rest"; "Partial (1.00, 0.96; metadata 0.67)" | Cell P, `seeded(...)` → `link_seeded`, `link_recall`, `link_meta_only`; print "[P] … linked colluders"; Cell Z "P32o" |
| App. F Table 11 (cols "Tmax unpadded / padded", rows A = 2,500 and 10,000) | multi-outlier oracle TPR in pooled populations | "1.00 / 0.83" and "1.00 / 0.17" | plausibly replicated by Cell P rows with `centre_week=False` (`row["Tmax"]`); the table's primary source is the Step 27–31 script |

### `experiments/04_frozen_holdout/`

#### Step 35 — [`step35_frozen_holdout.py`](experiments/04_frozen_holdout/step35_frozen_holdout.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| Table 3 (MiniLM, 64–252 accounts, row "False alarms"); Fig. 1b ("false alarms (1.8–1.9%)"); §7 'What held' and App. I ('1.8–1.9%'); Prediction log (Step 35, frozen calibration: 'Pass (1.8%, 1.9%, 1.9%') | held-out false-alarm rate, Two-scale / Hollowed / Multi-outlier / Volume scan (volume scan = rolling 8-week `vol_roll`) | 1.8% / 1.9% / 1.9% / 1.5% | Cell 8, `FPR_M = fpr_table(...)`, print "[P35a] held-out FPR" (tsC / tsRc / Tmax / vol_roll); Cell 14 P35a print and "[FPR matched]" |
| App. I 'What held' | weeks with any false alarm (MiniLM) | two or three of 17 weeks | Cell 8 print "[P35a]" (`weeks with FA`) |
| Prediction log (Step 35, frozen calibration) | week-unit 95% upper bound on false alarms | week-level upper 4–6% | `week_upper(...)` in Cell 4; printed in Cell 8 "[P35a]" and Cell 14 P35a |
| Table 3 (MiniLM, 64–252 accounts, row "Dataset (MetaMathQA)"); Fig. 1b; Table 2 (col "Weekly detection", untouched values); App. Table 15 (untouched, "Dataset"); Prediction log (Step 35, dataset campaigns: 'Pass (1.00, 0.97; 0.86)') | weekly paired detection, Two-scale / Hollowed / Multi-outlier / Volume scan | 0.86 / 1.00 / 0.97 / 0.61 | Cell 8, `RES_M = run_sources(...)`, print "[matched] metamath_unique" (tsC / tsRc / Tmax / vol_roll); Cell 14 P35b |
| Abstract ('0.86–1.00 of weeks'); §1 item 1 ('86–100% of weeks'); §7 and App. I ('0.86–1.00') | dataset-campaign detection range over both encoders | 0.86–1.00 | Cell 8 `RES_M` and Cell 13 `RES_E`, source metamath_unique |
| Table 3 (MiniLM, 64–252 accounts, row "Half padding"); Fig. 1b (0.57); Table 2 (untouched: two-scale 0.00, multi-outlier 0.00); App. Table 15 (untouched, "Half padding"); Prediction log (Step 35, mimicry: 'hollowed 0.57; Tmax 0.00; two-scale 0.00'); App. I ('0.37 against 0.00') | weekly paired detection under half padding, Two-scale / Hollowed / Multi-outlier / Volume scan | 0.00 / 0.57 / 0.00 / 0.37 | Cell 8, `RES_M`, print "[matched] metamath_mixed"; Cell 14 P35c |
| §7 'What held'; App. I; Table 2 (Hollowed, Half padding, untouched); App. Table 15 (untouched, Hollowed) | hollowed detection of padded campaigns across three campaign seeds | 0.50–0.57 | Cell 8, `RES_M` (seed 0) and `RUNS[sd]` (seeds 1, 2); Cell 14 verdict rows "[matched]", "[seed1]", "[seed2]" metamath_mixed tsRc |
| Abstract ('0.50–0.96 of weeks'); §1 item 2 ('50–96% of untouched weeks') | hollowed detection of padded campaigns, over seeds, E5 and whole weeks | 0.50–0.96 | Cell 8 `RES_M`/`RUNS`, Cell 13 `RES_E`, Cell 12 `RES_L` (metamath_mixed, tsRc) |
| App. I 'What held'; Prediction log (Step 35, run-to-run spread: 'Pass (≤ 0.07)') | run-to-run spread across campaign seeds | at most 0.07 | Cell 14, `spread`, print "P35l" |
| Table 3 (MiniLM, 64–252 accounts, row "Least detectable (tilted)"); Fig. 1b (0.04); Table 2 (row "Least detectable", '≤0.04'); App. Table 15 (untouched, "Least detectable"); §7 and App. I ('at or below 0.04'); App. G.2 ('at most 0.04 on untouched weeks'); Prediction log (Step 35, tilted: 'Pass (≤ 0.04)') | weekly paired detection of the tilted campaign, Two-scale / Hollowed / Multi-outlier / Volume scan | 0.00 / 0.04 / 0.00 / 0.10 | Cell 8, `RES_M`, print "[matched] tilted_c05"; Cell 14 P35d |
| Table 3 (MiniLM, 64–252 accounts, row "Random real prompts"); Prediction log (Step 35, random natural: 'Pass (0.00)') | weekly paired detection, all four detectors | 0.00 / 0.00 / 0.00 / 0.00 | Cell 8, `RES_M`, print "[matched] random"; Cell 14 P35e |
| Table 3 (MiniLM, 64–252 accounts, row "GSM8K sweep, 10 / 3 accounts") | weekly paired detection of benign evaluation sweeps, Two-scale / Hollowed / Multi-outlier / Volume scan | 0.87 / 0.88, 1.00 / 0.34, 0.97 / 0.93, 0.46 / 0.46 | Cell 8, `RES_M`, prints "[matched] eval_gsm8k(10x100)" and "eval_gsm8k(3x300)" (from `R["EVALS"]`) |
| Table 3 (E5-small-v2, row "False alarms"); §7 and App. I ('0–1.3%') | held-out false-alarm rate, Two-scale / Hollowed / Multi-outlier / Volume scan | 1.3% / 0.3% / 0.0% / 6.2% | Cell 13, `FPR_E = fpr_table(...)`, print "[E5] held-out FPR" |
| Table 3 (E5-small-v2, row "Dataset"); Prediction log (Step 35, E5: 'Pass (1.00, 0.00, 0.00; ...') | weekly paired detection, Two-scale / Hollowed / Multi-outlier / Volume scan | 0.86 / 1.00 / 1.00 / 0.48 | Cell 13, `RES_E`, print "[matched] metamath_unique" (second occurrence); Cell 14 P35f |
| Table 3 (E5-small-v2, row "Half padding"); §7 and App. I ('0.89 with E5') | weekly paired detection, Two-scale / Hollowed / Multi-outlier / Volume scan | 0.00 / 0.89 / 0.00 / 0.09 | Cell 13, `RES_E`, print "[matched] metamath_mixed" (second occurrence); Cell 14 P35f |
| Table 3 (E5-small-v2, row "Least detectable") | weekly paired detection, Two-scale / Hollowed / Multi-outlier / Volume scan | 0.00 / 0.03 / 0.00 / 0.02 | Cell 13, `RES_E`, print "[matched] tilted_c05" (second occurrence); Cell 14 P35f |
| Table 3 (MiniLM, whole weeks, row "False alarms"); Prediction log (Step 35, whole weeks: 'Fail (2.5%; 1.00)'); App. I ('fired on 6–11% of null windows elsewhere', with E5 6.2%) | held-out false-alarm rate, Two-scale / Hollowed / Multi-outlier / Volume scan | 1.0% / 0.5% / 2.5% / 11.3% | Cell 12, `FPR_L = fpr_table(...)`, print "[P35k] held-out FPR"; Cell 14 P35k |
| Table 3 (MiniLM, whole weeks, row "Dataset") | weekly paired detection, Two-scale / Hollowed / Multi-outlier / Volume scan | 0.90 / 1.00 / 1.00 / 0.84 | Cell 12, `RES_L`, print "[large] metamath_unique"; Cell 14 P35k |
| Table 3 (MiniLM, whole weeks, row "Half padding"); §7 and App. I ('0.96 in whole weekly populations') | weekly paired detection, Two-scale / Hollowed / Multi-outlier / Volume scan | 0.00 / 0.96 / 0.00 / 0.51 | Cell 12, `RES_L`, print "[large] metamath_mixed" |
| Table 3 (row label "(260–469 accounts)"); App. I 'What could not be tested'; Prediction log (Step 35, whole weeks: 'weeks of 260–469 accounts only') | size of whole-week populations | 260–469 | plausibly Cell 12 print "[large] holdout weeks ... window sizes" (`sizes_L`); not otherwise isolated |
| App. I 'What could not be tested' | benign accounts flagged per week by multi-outlier selection at 10^5 accounts | about 160 | Cell 12, `FLAGGED_1E5`, print "[large] benign accounts flagged per week at 1e5" |
| App. I 'What held' | share of accounts flagged by multi-outlier selection that are colluders | 94% | `Tmax_precision` in `run_sources` (Cell 7); printed as "Tmax prec" in Cell 14 verdict rows (paper does not say which row) |
| App. I 'What held' ('changed nothing'); Prediction log (Step 35, addition vs replacement: 'Pass (0.00)') | change in random-campaign detection, addition vs replacement | 0.00 | Cell 8, `RES_ADD`; Cell 14 print "P35m" |
| §7 'What did not'; App. I; App. D.4 ('On the untouched weeks the ratio was 2.22'); Prediction log (Step 35, E5: '3.06 vs 2.22'; 'Census ratio 2.22, not 1.14') | real-to-shuffled top-eigenvalue ratio of R_c, MiniLM / E5 | 2.22 / 3.06 | Cell 9 `CEN["minilm"]` and Cell 13 `CEN["e5"]` (`census`), prints "[census]"; Cell 14 P35f |
| App. I ('calibrated but nearly powerless (0.03 on dataset campaigns)'); Prediction log (Step 35, volume scan: 'Fail (0.0%; 0.03; both 0.00)') | conformal volume scan: false alarms; dataset detection; padded detection (volume scan and T_max) | 0.0%; 0.03; both 0.00 | `vol_conf` detector: Cell 8 `FPR_M`/`RES_M`; Cell 14 print "P35g" |
| §7 'What held' and App. I ('49–59% of the non-sacrificial colluders'; 'both still caught every campaign'; selection on C 'again 0.00 here'); Prediction log (Step 35, sacrificial: 'Fail (1.00; recall 0.49)') | sacrificial scheduling: detection and non-sacrificial recall | 49–59%; 1.00; 0.00 | Cell 10, `SCHED`, print "[X] sacrificial" (and "[X] random" for comparison); Cell 14 P35h and "[SCHED]" |
| §7 'What did not' and App. I (ramp); Prediction log (Step 35, ramp: 'Fail (0.89 to 0.00)'); App. D.6 ('On the untouched weeks the attack defeated every detector') | abrupt -> ramped detection: multi-outlier; conformal two-scale; hollowed; rolling two-scale | 0.89 to 0.00; 0.80 to 0.02; 0.98 to 0.32; 0.79 to 0.32 | Cell 10, `TEMP`, print "[T] week" (cumulative; last line = final); Cell 14 P35i and "[TEMP]" |
| §7 'Covert is not invisible'; App. I; Prediction log (Step 35, tilted: 'Ranked by hollowed (AUC 0.97)') | AUC of hollowed selection: tilted (MiniLM), tilted (E5), random real prompts (MiniLM) | 0.97, 0.92, 0.88 | `roc(...)` in Cell 4 via `run_sources`; Cell 14 verdict rows "[matched] tilted_c05", "[e5] tilted_c05", "[matched] random", tsRc `auc=` |
| Table 2 (col "First weekly alarm", untouched values: Two-scale and Multi-outlier); Prediction log (Step 35, martingale: 'never') | median week of first weekly alarm: dataset / half padding | week 1 / none (both detectors) | Cell 11, `SEQ[('all', d, src)]['first_alarm']`, print "[SEQ] all"; Cell 14 P35j. GSM8K gains in parentheses come from the distillation steps, not this script |
| Table 2 (col "Anytime-valid", untouched values: Two-scale and Multi-outlier); Prediction log (Step 35, martingale: 'Fail (25% of runs in 16 weeks; ...') | median week of conformal-martingale alarm: dataset / half padding | week 3 / none (two-scale); week 3 / 25% of runs (multi-outlier) | Cell 11, `SEQ[('all', d, src)]['martingale']` and `['frac_m']`, print "[SEQ] all"; Cell 14 P35j |
| §6 and App. G.3 ('null runs alarmed within 16 weeks a quarter of the time on untouched weeks') | share of null runs with a first weekly alarm within 16 weeks | a quarter of the time | Cell 11, `SEQ[('all', d, 'null')]['frac_f']`; Cell 14 P35j ("first-alarm Tmax") |
| Table 2 (Hollowed, untouched: 'week 1 (8.8)', 'week 1 (5.6)') | first weekly alarm of hollowed selection | week 1 / week 1 | not isolated (Cell 11 runs the sequential rules for tsC and Tmax only) |
| §2 'Data'; §7; App. I 'Design' | untouched data: test weeks; history weeks; English user turns; hashed IPs | 17 weeks; 31 weeks; 757,000; 76,000 | Cell 2 print (turns, hashed IPs, history/holdout weeks); Cell 8 print "[matched] holdout test weeks" |
| Table 3 (row label "MiniLM, 64–252 accounts"); App. Table 15 ("Untouched weeks, 64–252 accounts"); App. I 'Design' | window sizes, matched regime | 64–252 | Cell 8 print "[matched] ... window sizes" |
| Prediction log (Step 35 rows) | locked predictions P35a-P35m | pass/fail as listed | Cell 14 ("predictions and verdict") |

### `experiments/05_randomized_interventions/`

#### Step 34 — [`step34_randomized_interventions.py`](experiments/05_randomized_interventions/step34_randomized_interventions.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| App. K (first paragraph) | design: pooled populations, weekly blocks, benign failover team | 250, 2,500 and 10,000 account-weeks; 12 weekly blocks; team of 20 accounts, half of it flagged | Cell 1, `S34` (`AS`, `BLOCKS`, `TEAM`); Cell R, team `G` |
| App. K | admissible throttle schedules | 4,094 | Cell R, `ASSIGN` |
| §8 'The provider's own randomness'; App. J (Prop. 5 paragraph); App. K; Prediction log (Step 34, P34a) | randomisation null on real, drifting traffic: rejection at 1% on null runs | one rejection at 1% in 120 null runs; Partial (1/40 at 250; 0/40 at 2,500 and 10,000) | Cell R, scenario `null` print "[R] A=... null"; Cell Z "P34a" print |
| §8 'The provider's own randomness'; App. J; App. K; Prediction log (Step 34, P34b/P34c) | failover campaign (reassigned within the block / one block later) detected | caught/detected in every run; 1.00 at 250 and 2,500 | Cell R, scenarios `fail0`, `fail1`; Cell Z "P34b/c" print |
| App. K; Prediction log (Step 34, P34b) | throttles at A = 10,000: flagged group already holds all 50 colluders | Fail (0.00: all 50 colluders already flagged) | Cell R, `fail0` at A=10000, `reject` and `colluders_in_G`; Cell Z "P34b/c" print |
| App. J; App. K | throughput lost by a campaign that dropped throttled work | 40% | Cell R, scenario `nofail`, `throughput_lost` |
| §8 'The provider's own randomness'; App. J; App. K; Prediction log (Step 34, P34d) | benign shared-queue team with failover fires the test | 82–98% of runs; Pass (0.82–0.98) | Cell R, scenario `team`; Cell Z "P34d" print |
| App. K | legitimate accounts throttled per detection (at deployed precision 0.73 and 0.54 measured at 5,000–7,552 account-weeks) | about 16–42 | not computed here (paper combines this design with the precision measured in Step 32) |
| App. K; Prediction log (Step 34, P34e) | toy answer marks: adaptive campaign at 10% uptake within four weeks; fixed-prompt campaign; answer-sharing benign team | 0.62; never (0.00); as often (identical) | Cell M, `RESM` / print "[M] ..."; Cell Z "P34e" print |
| Prediction log (Step 34, P34f); App. F 'Statistics that failed' | dual-reference test (slope of frozen − lagged score) detecting ramps | 0.03 | Cell D, `det_slope`, print "[D] ramp detection"; Cell Z "P34f" print |
| Prediction log (Step 34, P34f); App. D.6 'Teaching the monitor that the campaign is normal' | reference frozen from early weeks, alone: ramp detection and false alarms | 0.44 at 5.2% | Cell D, `det_frz` with `FA_F` / `FA_F_DRIFT` in print "[D] ramp detection" (which false-alarm figure is the 5.2%: not isolated) |

### `experiments/future_work_not_run/`

#### Step 36 — [`step36_covert_corner.py`](experiments/future_work_not_run/step36_covert_corner.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| none — not run | none — not run (future work; implements open tests in §11 'Limitations and open tests') | none — not run | none — not run |

#### Step 37 — [`step37_live_teacher_distillation.py`](experiments/future_work_not_run/step37_live_teacher_distillation.py)

| Paper location | Quantity | Value | Where in the script |
|---|---|---|---|
| none — not run | none — not run (future work; implements open tests in §11 'Limitations and open tests') | none — not run | none — not run |

