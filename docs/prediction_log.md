# Prediction log

Every prediction below was written into the header of its notebook before the run. The predictions carry no external timestamp, so we call them *locked* rather than pre-registered. This log accompanies the paper (its Reproducibility appendix gives the totals: 75 passed, 30 partially, 74 failed, 5 untestable); with each Step linked to the script that tests it; inside the script, the prediction IDs (e.g. `P35a`) appear in the header cell and in the final scorecard cell.

| Step | Script | Prediction | Outcome | Consequence |
|---|---|---|---|---|
| 1 | [`step01_missing_mode.py`](../experiments/01_controlled_studies/step01_missing_mode.py) | Union S(k) matches jittered-grid theory | Pass (≤ 2% error) | — |
| 1 | [`step01_missing_mode.py`](../experiments/01_controlled_studies/step01_missing_mode.py) | Colluder pair correlation -(1-S̄)/K | Pass (-0.0451 vs -0.0451) | — |
| 1 | [`step01_missing_mode.py`](../experiments/01_controlled_studies/step01_missing_mode.py) | BBP outlier below the bulk | Pass (0.0702 vs 0.0701) | — |
| 1 | [`step01_missing_mode.py`](../experiments/01_controlled_studies/step01_missing_mode.py) | Padding threshold f^*=0.445 | Partial (0.408) | TW finite-size correction |
| 1 | [`step01_missing_mode.py`](../experiments/01_controlled_studies/step01_missing_mode.py) | Outlier independent of K | Pass (spread 0.0017) | Core claim |
| 2 | [`step02_theory_scaling_zoo.py`](../experiments/01_controlled_studies/step02_theory_scaling_zoo.py) | Tracy–Widom calibration to 10^-3 | Pass (to 10^-4) | Analytic thresholds |
| 2 | [`step02_theory_scaling_zoo.py`](../experiments/01_controlled_studies/step02_theory_scaling_zoo.py) | Raw fails, whitened calibrated under non-uniform traffic | Pass (100% vs 0.53%) | Whitening adopted |
| 2 | [`step02_theory_scaling_zoo.py`](../experiments/01_controlled_studies/step02_theory_scaling_zoo.py) | TW-corrected f_50 at unseen sizes | Pass (≤ 0.033) | — |
| 2 | [`step02_theory_scaling_zoo.py`](../experiments/01_controlled_studies/step02_theory_scaling_zoo.py) | Optimal window peak | Pass (37 vs 36) | — |
| 2 | [`step02_theory_scaling_zoo.py`](../experiments/01_controlled_studies/step02_theory_scaling_zoo.py) | N^*∝ A | Pass (slope 0.95) | — |
| 2 | [`step02_theory_scaling_zoo.py`](../experiments/01_controlled_studies/step02_theory_scaling_zoo.py) | Same-marginal stratified attacker detectable | Pass (1.00) | — |
| 3 | [`step03_highdim_rff.py`](../experiments/01_controlled_studies/step03_highdim_rff.py) | High-dimensional analytic calibration | Fail, then fixed | Row normalisation |
| 3 | [`step03_highdim_rff.py`](../experiments/01_controlled_studies/step03_highdim_rff.py) | Herding detectable in every dimension | Fail (S̄ 0.70–0.89) | Led to the law |
| 3 | [`step03_highdim_rff.py`](../experiments/01_controlled_studies/step03_highdim_rff.py) | Mismatched herding detectable | Fail (S̄≈ 1) | — |
| 3 | [`step03_highdim_rff.py`](../experiments/01_controlled_studies/step03_highdim_rff.py) | k-means++ shows λ_max | Fail (looks i.i.d.) | — |
| 4 | [`step04_linkage_price_law.py`](../experiments/01_controlled_studies/step04_linkage_price_law.py) | One curve, x_50in[1.0,1.35] | Pass (1.052; 47/48) | Lower-edge law |
| 4 | [`step04_linkage_price_law.py`](../experiments/01_controlled_studies/step04_linkage_price_law.py) | Herding S̄ falls with N | Fail (rises) | Pool floor |
| 4 | [`step04_linkage_price_law.py`](../experiments/01_controlled_studies/step04_linkage_price_law.py) | Aligned herding detected almost everywhere | Partial (exactly where x>1) | — |
| 4 | [`step04_linkage_price_law.py`](../experiments/01_controlled_studies/step04_linkage_price_law.py) | Spectral linkage AUC ≥ 0.9 | Pass (0.986 vs 0.481) | — |
| 4 | [`step04_linkage_price_law.py`](../experiments/01_controlled_studies/step04_linkage_price_law.py) | Oracle-only band exists | Pass (0.75 vs 0.03) | Appendix A.1 |
| 5 | [`step05_real_extraction_cifar10.py`](../experiments/01_controlled_studies/step05_real_extraction_cifar10.py) | Victim ≥ 85%; random looks null | Pass (88.5–89.1%) | — |
| 5 | [`step05_real_extraction_cifar10.py`](../experiments/01_controlled_studies/step05_real_extraction_cifar10.py) | Active strategies beat random by ≥ 1 point | Partial (k-center yes, entropy no) | — |
| 5 | [`step05_real_extraction_cifar10.py`](../experiments/01_controlled_studies/step05_real_extraction_cifar10.py) | No x>1; concentration caught by λ_max | Pass | — |
| 5 | [`step05_real_extraction_cifar10.py`](../experiments/01_controlled_studies/step05_real_extraction_cifar10.py) | Fidelity gain tracks concentration | Fail (ρ=-0.15) | — |
| 5 | [`step05_real_extraction_cifar10.py`](../experiments/01_controlled_studies/step05_real_extraction_cifar10.py) | Herding undetected because S̄≈ 1 | Partial (S̄ 1.1–1.5) | — |
| 5 | [`step05_real_extraction_cifar10.py`](../experiments/01_controlled_studies/step05_real_extraction_cifar10.py) | Spectral ≥ per-account, gap grows with A | Pass on mean power; not cell by cell | — |
| 5 | [`step05_real_extraction_cifar10.py`](../experiments/01_controlled_studies/step05_real_extraction_cifar10.py) | Law holds on real attacks | Pass, uninformative (all x<0.25) | — |
| 6 | [`step06_baselines.py`](../experiments/01_controlled_studies/step06_baselines.py) | Spectral vs global crossover at K≈ 2.3√ A | Pass (7 of 8 cells; 2/2) | Regime map |
| 6 | [`step06_baselines.py`](../experiments/01_controlled_studies/step06_baselines.py) | HC ≥ per-account; spectral ≥ HC at N=2,000 | Pass (29/30; 8/10) | — |
| 6 | [`step06_baselines.py`](../experiments/01_controlled_studies/step06_baselines.py) | Random campaigns ≤ 2%; linkage AUC ≈ 0.5 | Pass (1.9%; 0.49–0.51) | — |
| 6 | [`step06_baselines.py`](../experiments/01_controlled_studies/step06_baselines.py) | No real strategy triggers the lower edge | Pass (0.002) | — |
| 7 | [`step07_robustness.py`](../experiments/01_controlled_studies/step07_robustness.py) | Only whole-window resampling calibrates | Pass (suite 0.0–0.6%) | Calibration protocol |
| 7 | [`step07_robustness.py`](../experiments/01_controlled_studies/step07_robustness.py) | Evading the suite needs padding f≥ 0.75 | Fail (no detection even unpadded) | Benign structure |
| 7 | [`step07_robustness.py`](../experiments/01_controlled_studies/step07_robustness.py) | Heterogeneous volumes cost ≤ 0.2 power | Untestable (power 0) | — |
| 7 | [`step07_robustness.py`](../experiments/01_controlled_studies/step07_robustness.py) | Time-splitting evades; aggregation restores | Untestable (power 0) | — |
| 7 | [`step07_robustness.py`](../experiments/01_controlled_studies/step07_robustness.py) | Upper-edge rule predicts power at A≤ 10^4 | Fail (10/24 cells) | No closed-form upper edge |
| 8 | [`step08_dose_response_taskaware.py`](../experiments/01_controlled_studies/step08_dose_response_taskaware.py) | Geometry breaks at sin[0.05,0.5] | Partial (0.175 at A=100; ≤ 0.06 at 1,000) | One cluster suffices |
| 8 | [`step08_dose_response_taskaware.py`](../experiments/01_controlled_studies/step08_dose_response_taskaware.py) | Task-aware suite ≥ 0.8 at s=1 | Fail (≤ 0.17) | Tolerance, not immunity |
| 8 | [`step08_dose_response_taskaware.py`](../experiments/01_controlled_studies/step08_dose_response_taskaware.py) | Calibration ≤ 1.5× nominal | Partial (mean 1.1%, worst 3.3%) | More windows needed |
| 8 | [`step08_dose_response_taskaware.py`](../experiments/01_controlled_studies/step08_dose_response_taskaware.py) | Clusters out-shine attacks in geometry, not margin | Partial (geometry yes, margin no) | Benign-coordination limit |
| 9 | [`step09_wildchat_llm.py`](../experiments/02_development_weeks/step09_wildchat_llm.py) | Fixed calibration transfers to later weeks | Fail (12%) | Drift-aware calibration |
| 9 | [`step09_wildchat_llm.py`](../experiments/02_development_weeks/step09_wildchat_llm.py) | Real users structured: λ_max real/shuffled ≥ 2 | Pass (4.3–5.1) | Read as topicality |
| 9 | [`step09_wildchat_llm.py`](../experiments/02_development_weeks/step09_wildchat_llm.py) | Capability campaigns detected at 5,000 and 20,000 | Partial (20,000 via global only) | Detection needs dominance |
| 9 | [`step09_wildchat_llm.py`](../experiments/02_development_weeks/step09_wildchat_llm.py) | Random campaigns ≤ 2× nominal | Fail (8–21% floor) | — |
| 9 | [`step09_wildchat_llm.py`](../experiments/02_development_weeks/step09_wildchat_llm.py) | Padding costs ≥ 0.3 power at 5,000 | Fail as specified; 0.79 at 20,000 | — |
| 10 | [`step10_wildchat_drift_granularity.py`](../experiments/02_development_weeks/step10_wildchat_drift_granularity.py) | Rolling ≤ 10%; lagged ≤ 7.5% | Partial (A=100 yes; A=500 no) | Small populations |
| 10 | [`step10_wildchat_drift_granularity.py`](../experiments/02_development_weeks/step10_wildchat_drift_granularity.py) | Broad at 5,000 ≤ 0.2 under all schemes | Partial (lagged 0.21, 0.29) | Failure is not drift |
| 10 | [`step10_wildchat_drift_granularity.py`](../experiments/02_development_weeks/step10_wildchat_drift_granularity.py) | Narrow at 5,000, lagged ≥ 0.5 | Pass (0.94; 0.71) | Granularity matters |
| 10 | [`step10_wildchat_drift_granularity.py`](../experiments/02_development_weeks/step10_wildchat_drift_granularity.py) | ≥ 0.5 needs ≥ 50% share, except narrow | Pass | — |
| 10 | [`step10_wildchat_drift_granularity.py`](../experiments/02_development_weeks/step10_wildchat_drift_granularity.py) | Padding drops ≥ 0.5 at 20,000, all schemes | Partial (A=100 yes) | — |
| 11 | [`step11_wildchat_dimension_robustness.py`](../experiments/02_development_weeks/step11_wildchat_dimension_robustness.py) | Lagged calibration nominal at 100, not 500, every d | Pass (3.4–4.4%; 22–39%) | Not a projection artefact |
| 11 | [`step11_wildchat_dimension_robustness.py`](../experiments/02_development_weeks/step11_wildchat_dimension_robustness.py) | Narrow at 5,000 ≥ 0.8 at every d | Pass (0.94–0.96) | — |
| 11 | [`step11_wildchat_dimension_robustness.py`](../experiments/02_development_weeks/step11_wildchat_dimension_robustness.py) | Broad at 5,000 ≤ 0.35 at every d and A | Partial (up to 0.44 at A=500) | More dimensions help |
| 11 | [`step11_wildchat_dimension_robustness.py`](../experiments/02_development_weeks/step11_wildchat_dimension_robustness.py) | Random ≤ 0.05 | Pass (≤ 0.01) | — |
| 11 | [`step11_wildchat_dimension_robustness.py`](../experiments/02_development_weeks/step11_wildchat_dimension_robustness.py) | Padding drops ≥ 0.4 at every d | Partial (0.32 at d=384) | — |
| 12 | [`step12_realistic_deployment.py`](../experiments/02_development_weeks/step12_realistic_deployment.py) | Volume-share test ≤ 0.1 | Pass (≤ 0.02) | Volume matching works |
| 12 | [`step12_realistic_deployment.py`](../experiments/02_development_weeks/step12_realistic_deployment.py) | Natural campaigns ≤ 0.2 on every detector | Fail (MMD 0.27) | — |
| 12 | [`step12_realistic_deployment.py`](../experiments/02_development_weeks/step12_realistic_deployment.py) | Generated sets at K=200: MMD or global ≥ 0.5 | Partial (K=200 infeasible) | — |
| 12 | [`step12_realistic_deployment.py`](../experiments/02_development_weeks/step12_realistic_deployment.py) | Two-scale beats spectral on a generated set | Pass (0.93 vs 0.19) | Headline detector |
| 12 | [`step12_realistic_deployment.py`](../experiments/02_development_weeks/step12_realistic_deployment.py) | Partial linkage raises per-account or HC by ≥ 0.1 | Untestable (base ≈ 0) | — |
| 12 | [`step12_realistic_deployment.py`](../experiments/02_development_weeks/step12_realistic_deployment.py) | Sharding: spectral ≤ 0.1 | Untestable (1–5 shards; unpaired) | Redone in Step 13 |
| 12 | [`step12_realistic_deployment.py`](../experiments/02_development_weeks/step12_realistic_deployment.py) | Every detector's false alarms ≤ 2% | Fail (only two-scale) | — |
| 13 | [`step13_twoscale_mechanism_scale.py`](../experiments/02_development_weeks/step13_twoscale_mechanism_scale.py) | Unique ≥ rephrased + 0.3 | Fail (0.88 vs 0.80) | Not deduplication |
| 13 | [`step13_twoscale_mechanism_scale.py`](../experiments/02_development_weeks/step13_twoscale_mechanism_scale.py) | Deduplicated natural ≥ raw natural + 0.2 | Fail (0.04 vs 0.01) | — |
| 13 | [`step13_twoscale_mechanism_scale.py`](../experiments/02_development_weeks/step13_twoscale_mechanism_scale.py) | Two-scale false alarms ≤ 2%, weekly and 4-week | Partial (0.6%; 9.8%) | Needs long history |
| 13 | [`step13_twoscale_mechanism_scale.py`](../experiments/02_development_weeks/step13_twoscale_mechanism_scale.py) | Two-scale group precision ≥ 0.5 | Pass (1.00; recall 0.17) | Manual review |
| 13 | [`step13_twoscale_mechanism_scale.py`](../experiments/02_development_weeks/step13_twoscale_mechanism_scale.py) | 4-week, K=200: ≥ 0.8; random ≤ 0.05 | Pass (0.86; 0.00) | — |
| 13 | [`step13_twoscale_mechanism_scale.py`](../experiments/02_development_weeks/step13_twoscale_mechanism_scale.py) | Sharding: per-shard two-scale ≤ 0.2 | Pass (0.00 for all) | Do not shard |
| 14 | [`step14_mechanism_distillation.py`](../experiments/03_capability_frontier/step14_mechanism_distillation.py) | Removing follow-ups closes ≥ 50% of the clumping gap | Fail (moved the wrong way) | Mechanism open |
| 14 | [`step14_mechanism_distillation.py`](../experiments/03_capability_frontier/step14_mechanism_distillation.py) | Follow-up mimicry cuts two-scale power by ≥ 0.3 | Pass (0.85 to 0.24) | Mimicry evades |
| 14 | [`step14_mechanism_distillation.py`](../experiments/03_capability_frontier/step14_mechanism_distillation.py) | Unique at 7,550 gains ≥ 5 points; natural ≤ 1 | Partial (+12.6; natural +6.0) | Gains partly generic |
| 14 | [`step14_mechanism_distillation.py`](../experiments/03_capability_frontier/step14_mechanism_distillation.py) | Evading variants gain ≤ 60% of unique | Fail (105% and 118%) | No price of stealth |
| 15 | [`step15_query_cluster_fair_baselines.py`](../experiments/02_development_weeks/step15_query_cluster_fair_baselines.py) | Query-cluster power on padded campaigns ≥ 0.6 | Fail (detector miscalibrated, 64% false alarms; AUC 0.89–0.90) | Recalibrate |
| 15 | [`step15_query_cluster_fair_baselines.py`](../experiments/02_development_weeks/step15_query_cluster_fair_baselines.py) | Rephrasings lie farther apart than h=0.5 | Pass (median 0.67) | Earlier ablation uninformative |
| 15 | [`step15_query_cluster_fair_baselines.py`](../experiments/02_development_weeks/step15_query_cluster_fair_baselines.py) | Exact repeats cut two-scale power by ≥ 0.3 | Fail (0.79 to 0.73) | Not deduplication |
| 15 | [`step15_query_cluster_fair_baselines.py`](../experiments/02_development_weeks/step15_query_cluster_fair_baselines.py) | Human GSM8K questions detected ≥ 0.7× MetaMathQA | Pass (0.94 vs 0.79) | Source-shift detector |
| 15 | [`step15_query_cluster_fair_baselines.py`](../experiments/02_development_weeks/step15_query_cluster_fair_baselines.py) | Matched 1%: two-scale ≥ MMD + 0.15; group MMD ≥ MMD | Partial (0.80 vs 0.58; group MMD fails) | — |
| 15 | [`step15_query_cluster_fair_baselines.py`](../experiments/02_development_weeks/step15_query_cluster_fair_baselines.py) | Camouflage cuts two-scale and QC power by ≥ 0.3 | Fail (two-scale rose to 0.94) | Camouflage does not help |
| 15 | [`step15_query_cluster_fair_baselines.py`](../experiments/02_development_weeks/step15_query_cluster_fair_baselines.py) | Realistic share: best calibrated detector ≥ 0.5 at 1% | Fail (0.00 at A=1,000) | Scale problem |
| 15 | [`step15_query_cluster_fair_baselines.py`](../experiments/02_development_weeks/step15_query_cluster_fair_baselines.py) | Mixed padding caught in 12 weeks, ≤ 15% false alarms | Partial (two-scale alone: 0.96, 9.6%) | — |
| 16 | [`step16_price_of_stealth.py`](../experiments/03_capability_frontier/step16_price_of_stealth.py) | All-math gains grow ≥ 3 points from 100 to 900 queries | Fail (+9.8 to +8.6) | Saturated by 100 |
| 16 | [`step16_price_of_stealth.py`](../experiments/03_capability_frontier/step16_price_of_stealth.py) | Mixed gains ≤ all-math - 2 at 300 and 900 | Fail (+2.7, +1.5) | No price of stealth |
| 16 | [`step16_price_of_stealth.py`](../experiments/03_capability_frontier/step16_price_of_stealth.py) | Camouflage gains ≤ 80% of all-math at 900 | Fail (130%) | — |
| 16 | [`step16_price_of_stealth.py`](../experiments/03_capability_frontier/step16_price_of_stealth.py) | MATH-500, 1.5B student: price of stealth | Untestable (not run) | Open |
| 17 | [`step17_calibrated_query_cluster.py`](../experiments/02_development_weeks/step17_calibrated_query_cluster.py) | Robustly calibrated query-cluster FPR ≤ 2% | Fail (28%) | Not operable |
| 17 | [`step17_calibrated_query_cluster.py`](../experiments/02_development_weeks/step17_calibrated_query_cluster.py) | Query-cluster power under mimicry ≥ 0.5, within 0.25 of unpadded | Fail (0.38–0.40; AUC 0.80–0.82) | Best ranker only |
| 17 | [`step17_calibrated_query_cluster.py`](../experiments/02_development_weeks/step17_calibrated_query_cluster.py) | Query-cluster: random ≤ 0.05; human GSM8K ≥ 0.5 | Partial (0.11; 0.58) | — |
| 17 | [`step17_calibrated_query_cluster.py`](../experiments/02_development_weeks/step17_calibrated_query_cluster.py) | Rolling populations: two-scale and QC calibrated; QC ≥ two-scale + 0.2 at 2.5% | Fail (24%, 69%) | Scale problem |
| 17 | [`step17_calibrated_query_cluster.py`](../experiments/02_development_weeks/step17_calibrated_query_cluster.py) | Rolling populations: every detector ≤ 0.3 at 1% | Pass (≤ 0.08) | — |
| 18 | [`step18_qc_repair_scale_evals.py`](../experiments/02_development_weeks/step18_qc_repair_scale_evals.py) | Lagged or conformal QC reaches FPR ≤ 2% | Pass (conformal 0.0%; lagged 15%) | Conservative only |
| 18 | [`step18_qc_repair_scale_evals.py`](../experiments/02_development_weeks/step18_qc_repair_scale_evals.py) | Calibrated QC on mixed padding ≥ two-scale + 0.15 | Fail (0.20 vs 0.21; AUC 0.86) | Ranking, not detection |
| 18 | [`step18_qc_repair_scale_evals.py`](../experiments/02_development_weeks/step18_qc_repair_scale_evals.py) | Gated QC: FPR ≤ 1.5%, mixed ≥ 0.3 | Fail (12%; 0.24) | — |
| 18 | [`step18_qc_repair_scale_evals.py`](../experiments/02_development_weeks/step18_qc_repair_scale_evals.py) | Fixed N: oracle ≥ 0.6 at 20%, ≤ 0.3 at 5% | Partial (0.84; 0.35) | Smooth decay with share |
| 18 | [`step18_qc_repair_scale_evals.py`](../experiments/02_development_weeks/step18_qc_repair_scale_evals.py) | At A=1,000, deployed ≤ oracle - 0.2 | Pass (0.05 vs 0.35) | Calibration bottleneck |
| 18 | [`step18_qc_repair_scale_evals.py`](../experiments/02_development_weeks/step18_qc_repair_scale_evals.py) | GSM8K evaluation sweep flagged in ≥ 50% of weeks | Pass (79–80%; MMLU 2–7%) | Benign false alarms |
| 19 | [`step19_nonsaturating_distillation.py`](../experiments/03_capability_frontier/step19_nonsaturating_distillation.py) | SmolLM2-360M lacks the skill: base GSM8K ≤ 10% | Pass (3.5%) | — |
| 19 | [`step19_nonsaturating_distillation.py`](../experiments/03_capability_frontier/step19_nonsaturating_distillation.py) | Extraction regime: all-math gains rise ≥ 5 points, 1k to 10k | Pass (+7.5 to +15.9) | Not saturated |
| 19 | [`step19_nonsaturating_distillation.py`](../experiments/03_capability_frontier/step19_nonsaturating_distillation.py) | Price of stealth: mixed ≤ all-math - 2 at 3k and 10k | Pass (-2.8, -3.8) | Cost is real |
| 19 | [`step19_nonsaturating_distillation.py`](../experiments/03_capability_frontier/step19_nonsaturating_distillation.py) | MATH-500 (1.5B): all-math ≥ +3, mixed ≤ 70% of it | Fail (+11.6; 83%) | Elicitation again |
| 20 | [`step20_baselines_adaptive_scaling.py`](../experiments/02_development_weeks/step20_baselines_adaptive_scaling.py) | A simple baseline matches two-scale (oracle 1%, within 0.05) | Fail (0.00 vs 0.81, 0.95) | Normalisation matters |
| 20 | [`step20_baselines_adaptive_scaling.py`](../experiments/02_development_weeks/step20_baselines_adaptive_scaling.py) | Only two-scale has deployed FPR ≤ 2% | Pass (0.8%; others 2.5–12%) | — |
| 20 | [`step20_baselines_adaptive_scaling.py`](../experiments/02_development_weeks/step20_baselines_adaptive_scaling.py) | Per-account classifier beats two-scale under padding | Fail (0.00 vs 0.24) | — |
| 20 | [`step20_baselines_adaptive_scaling.py`](../experiments/02_development_weeks/step20_baselines_adaptive_scaling.py) | Adaptive attacker: two-scale ≤ 0.3; classifier ≥ 0.5 | Fail (0.31; 0.00) | Optimisation helps attacker |
| 20 | [`step20_baselines_adaptive_scaling.py`](../experiments/02_development_weeks/step20_baselines_adaptive_scaling.py) | Null median grows with ln A at the Gumbel rate | Fail (slope -0.17 vs +0.62) | Not a max statistic |
| 20 | [`step20_baselines_adaptive_scaling.py`](../experiments/02_development_weeks/step20_baselines_adaptive_scaling.py) | Predicted breaking point within a factor of 2 | Fail (502 vs 2,081) | — |
| 20 | [`step20_baselines_adaptive_scaling.py`](../experiments/02_development_weeks/step20_baselines_adaptive_scaling.py) | Heavy benign tail (ξ>0) | Fail (ξ=-0.16) | Light tail |
| 20 | [`step20_baselines_adaptive_scaling.py`](../experiments/02_development_weeks/step20_baselines_adaptive_scaling.py) | GEV fit predicts larger-A thresholds | Fail (1 of 4 inside band) | Threshold flat |
| 20 | [`step20_baselines_adaptive_scaling.py`](../experiments/02_development_weeks/step20_baselines_adaptive_scaling.py) | True colluder set recovers ≥ half the lost power | Pass (all of it) | Selection fails |
| 21 | [`step21_sparse_boundary_adversary.py`](../experiments/02_development_weeks/step21_sparse_boundary_adversary.py) | Sparse scan ≥ spectral + 0.2 at K/A≤ 5% | Fail (0.03 vs 0.03) | Self-terms capture scans |
| 21 | [`step21_sparse_boundary_adversary.py`](../experiments/02_development_weeks/step21_sparse_boundary_adversary.py) | Oracle told the colluders detects at 1% share | Pass (1.00; random-group null) | Gap is wide |
| 21 | [`step21_sparse_boundary_adversary.py`](../experiments/02_development_weeks/step21_sparse_boundary_adversary.py) | Sparse scans add ≤ 0.1 where spectral works | Pass (uninformative: both ≈ 0) | — |
| 21 | [`step21_sparse_boundary_adversary.py`](../experiments/02_development_weeks/step21_sparse_boundary_adversary.py) | Boundary ε^* within 2× of benign top-group ε | Fail (0.035 vs 0.447) | Wrong distance |
| 21 | [`step21_sparse_boundary_adversary.py`](../experiments/02_development_weeks/step21_sparse_boundary_adversary.py) | Sacrificial schedule: detection ≥ 0.6, other colluders ≤ 10% flagged | Pass (0.92; 0%) | Attacker picks who is flagged |
| 21 | [`step21_sparse_boundary_adversary.py`](../experiments/02_development_weeks/step21_sparse_boundary_adversary.py) | Temporal camouflage cuts detection by ≥ 0.3 | Pass (0.95 to 0.12) | Calibration poisoning |
| 21 | [`step21_sparse_boundary_adversary.py`](../experiments/02_development_weeks/step21_sparse_boundary_adversary.py) | TPR at 10^-3 ≥ 0.5 and ≥ global | Pass (0.88 vs 0.70) | Thin null tail |
| 22 | [`step22_hollowed_profile_sequential.py`](../experiments/02_development_weeks/step22_hollowed_profile_sequential.py) | Hollowing: real/shuffled λ_max(R)≤ 2 | Partial (0.35: common drift mode) | Not coordination |
| 22 | [`step22_hollowed_profile_sequential.py`](../experiments/02_development_weeks/step22_hollowed_profile_sequential.py) | Hollowed selection halves bulk-user groups | Fail (0.08 to 0.06) | — |
| 22 | [`step22_hollowed_profile_sequential.py`](../experiments/02_development_weeks/step22_hollowed_profile_sequential.py) | Group MMD on hollowed groups AUC ≥ 0.6 | Fail (0.18) | — |
| 22 | [`step22_hollowed_profile_sequential.py`](../experiments/02_development_weeks/step22_hollowed_profile_sequential.py) | Hollowing keeps calibration (≤ 1.5%) and power (≥ 0.75) | Partial (8.4%; 0.90) | Uncalibrated |
| 22 | [`step22_hollowed_profile_sequential.py`](../experiments/02_development_weeks/step22_hollowed_profile_sequential.py) | Profile score ≥ two-scale + 0.1 under padding | Fail (0.00 vs 0.21) | — |
| 22 | [`step22_hollowed_profile_sequential.py`](../experiments/02_development_weeks/step22_hollowed_profile_sequential.py) | Profile score flags GSM8K sweeps ≥ 50% | Fail (43%) | — |
| 22 | [`step22_hollowed_profile_sequential.py`](../experiments/02_development_weeks/step22_hollowed_profile_sequential.py) | Weeks to detection within 2× of ln(1/α)/D | Partial (15 vs 14.7, 16 vs 13.3; unpadded at the floor) | Proposition 3 holds |
| 22 | [`step22_hollowed_profile_sequential.py`](../experiments/02_development_weeks/step22_hollowed_profile_sequential.py) | Martingale median ≤ 6 weeks under padding, all-time FA ≤ 1% | Fail (15 weeks; FA 0/33) | Slow but valid |
| 22 | [`step22_hollowed_profile_sequential.py`](../experiments/02_development_weeks/step22_hollowed_profile_sequential.py) | Core-plus-halo calibrated power ≥ 0.4 | Fail (halo 0.02; hollowed cores 0.90 at 7.6%) | — |
| 22 | [`step22_hollowed_profile_sequential.py`](../experiments/02_development_weeks/step22_hollowed_profile_sequential.py) | Prefilter keeps ≥ 95% of colluders at 10% pass | Fail (0%) | Wrong tail |
| 22 | [`step22_hollowed_profile_sequential.py`](../experiments/02_development_weeks/step22_hollowed_profile_sequential.py) | Prefiltering restores power at 5% share | Fail (0.00) | — |
| 23 | [`step23_hollowed_calibration_scale.py`](../experiments/02_development_weeks/step23_hollowed_calibration_scale.py) | Common-mode removal: real/shuffled λ_max(R_c)≥ 1 | Pass (1.14) | Benign coordination is weak |
| 23 | [`step23_hollowed_calibration_scale.py`](../experiments/02_development_weeks/step23_hollowed_calibration_scale.py) | Conformal hollowed selection: false alarms ≤ 2% | Pass (0/960 windows; week-level bound 12%) | Conservative |
| 23 | [`step23_hollowed_calibration_scale.py`](../experiments/02_development_weeks/step23_hollowed_calibration_scale.py) | Calibrated hollowed selection ≥ 0.6 under padding | Fail (0.48) | Half, against none |
| 23 | [`step23_hollowed_calibration_scale.py`](../experiments/02_development_weeks/step23_hollowed_calibration_scale.py) | Hollowed selection ≥ 0.5 at 1% colluders | Fail (0.00; group has no colluders) | Barrier returns |
| 23 | [`step23_hollowed_calibration_scale.py`](../experiments/02_development_weeks/step23_hollowed_calibration_scale.py) | Sweeps: 3 accounts ≤ 0.4, 10 accounts ≥ 0.5 | Partial (0.42; 0.97) | — |
| 27 | [`step27_31_bulk_edge_multioutlier_tilted.py`](../experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | Campaign below the bulk edge at A≥ 2,500 | Pass by the letter (uninformative: compared with null λ_max; campaign far above the edge) | Out-ranked, not submerged |
| 27 | [`step27_31_bulk_edge_multioutlier_tilted.py`](../experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | Null-selected groups not coordinated | Fail (cosine 0.43–0.57 vs 0.00; selection-biased) | — |
| 27 | [`step27_31_bulk_edge_multioutlier_tilted.py`](../experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | Colluder cosine 0.025–0.045; padded ratio 0.4–0.6 | Partial (0.57; 0.51) | — |
| 27 | [`step27_31_bulk_edge_multioutlier_tilted.py`](../experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | r≥ 1,500 lifts unpadded TPR at A=2,500 to ≥ 0.5 | Pass (0.00 to 0.57) | Rank depends on r |
| 28 | [`step27_31_bulk_edge_multioutlier_tilted.py`](../experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | T_max calibrated (conformal) | Pass (0/960 windows; week-level bound 12%) | — |
| 28 | [`step27_31_bulk_edge_multioutlier_tilted.py`](../experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | Multi-outlier ≥ top eigenvector + 0.2 at A=2,500 | Pass (1.00 vs 0.00) | Selection gap closed |
| 28 | [`step27_31_bulk_edge_multioutlier_tilted.py`](../experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | k(10,000)≤ 5 k(1,000) | Pass (uninformative: cap of 20 reached) | — |
| 28 | [`step27_31_bulk_edge_multioutlier_tilted.py`](../experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | GEV predicts T_max threshold at ≥ 3 of 4 sizes | Fail (2 of 4) | Threshold rises |
| 29 | [`step27_31_bulk_edge_multioutlier_tilted.py`](../experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | Task region rare: p≤ 5% | Pass (1.48%) | — |
| 29 | [`step27_31_bulk_edge_multioutlier_tilted.py`](../experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | Tilted campaign evades: two-scale ≤ 0.10, hollowed ≤ 0.25 | Pass (0.00; 0.01) | Evades every detector |
| 29 | [`step27_31_bulk_edge_multioutlier_tilted.py`](../experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | Detectors realise ≤ 1% of N_w d(c‖p) | Pass (0.03%) | — |
| 30 | [`step27_31_bulk_edge_multioutlier_tilted.py`](../experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | Cell drift low-rank (≥ 50% in q≤ 5) | Fail (31%) | Per-cell drift |
| 30 | [`step27_31_bulk_edge_multioutlier_tilted.py`](../experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | Factor adjustment calibrates (≤ 2%) | Fail (23%) | — |
| 30 | [`step27_31_bulk_edge_multioutlier_tilted.py`](../experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | Factor QC ≥ two-scale + 0.1 under padding | Partial (0.48 vs 0.01 at 23% false alarms) | Miscalibrated |
| 30 | [`step27_31_bulk_edge_multioutlier_tilted.py`](../experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | Factor model ≥ 5 points below forecast | Fail (23% vs 14.5%) | — |
| 31 | [`step27_31_bulk_edge_multioutlier_tilted.py`](../experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | Two-sided prefilter keeps ≥ 80% of colluders | Fail (13%) | — |
| 31 | [`step27_31_bulk_edge_multioutlier_tilted.py`](../experiments/02_development_weeks/step27_31_bulk_edge_multioutlier_tilted.py) | It keeps evaluation sweeps too | Pass (100%) | Catches pipelines |
| 29 | [`step29d_tilted_distillation.py`](../experiments/03_capability_frontier/step29d_tilted_distillation.py) | Base-rate on-task prompts distil (≥ 8 points at 10k) | Fail (+1.5) | Covert but useless |
| 32 | [`step32_written_attacker.py`](../experiments/03_capability_frontier/step32_written_attacker.py) | Value-defined task region rare (p_v≤ 5%) | Pass (0.53%) | — |
| 32 | [`step32_written_attacker.py`](../experiments/03_capability_frontier/step32_written_attacker.py) | Written prompts can be made user-like | Fail (AUC 0.98; 4.3% pass) | Rewriter too weak |
| 32 | [`step32_written_attacker.py`](../experiments/03_capability_frontier/step32_written_attacker.py) | Written campaign evades | Pass (0.00; 0.10 all on task) | — |
| 32 | [`step32_written_attacker.py`](../experiments/03_capability_frontier/step32_written_attacker.py) | Written campaign keeps capability (≥ 6) | Fail (+2.2; 213 unique prompts) | Supply-limited |
| 32 | [`step32_written_attacker.py`](../experiments/03_capability_frontier/step32_written_attacker.py) | Value-defined tilted campaign ≥ 4 points | Fail (+1.1) | — |
| 32 | [`step32_written_attacker.py`](../experiments/03_capability_frontier/step32_written_attacker.py) | Per-account topicality sees tilted campaigns | Fail (25% false alarms) | Uncalibrated |
| 32 | [`step32_written_attacker.py`](../experiments/03_capability_frontier/step32_written_attacker.py) | Pooling artefact ≥ 2× random | Partial (1.9–2.1×) | Not the cause |
| 32 | [`step32_written_attacker.py`](../experiments/03_capability_frontier/step32_written_attacker.py) | Per-week centring keeps multi-outlier (≥ 0.8 at 10^4) | Pass (1.00) | — |
| 32 | [`step32_written_attacker.py`](../experiments/03_capability_frontier/step32_written_attacker.py) | Multi-outlier deployable at 2,500 | Fail (30% false alarms) | Oracle only |
| 32 | [`step32_written_attacker.py`](../experiments/03_capability_frontier/step32_written_attacker.py) | Iterative rewriting reaches AUC ≤ 0.65 | Fail (0.98, 0.97, 0.97) | Writer too weak |
| 32 | [`step32_written_attacker.py`](../experiments/03_capability_frontier/step32_written_attacker.py) | Distribution-matched attacker evades and gains ≥ 6 | Fail (evades; +4.4, ESS 151) | Frontier holds |
| 32 | [`step32_written_attacker.py`](../experiments/03_capability_frontier/step32_written_attacker.py) | Personas evade query detectors, type count ≥ 0.9 | Fail (0.00; type count 0.58) | Weak personas |
| 32 | [`step32_written_attacker.py`](../experiments/03_capability_frontier/step32_written_attacker.py) | Majority-on-task accounts rare (q≤ 3%) | Pass (0.25%) | — |
| 32 | [`step32_written_attacker.py`](../experiments/03_capability_frontier/step32_written_attacker.py) | Multi-outlier deployable at 10^4 | Fail (5% false alarms at 7,552) | Improving with size |
| 32 | [`step32_written_attacker.py`](../experiments/03_capability_frontier/step32_written_attacker.py) | Linked 30%: expansion ≥ 0.8, recall ≥ 0.5; metadata alone ≤ 0.5 | Partial (1.00, 0.96; metadata 0.67) | Complements linkage |
| 32 | [`step32_written_attacker.py`](../experiments/03_capability_frontier/step32_written_attacker.py) | ≤ 100 benign accounts flagged per week at 10^5 | Pass (44–59 at A≥5,000) | — |
| 34 | [`step34_randomized_interventions.py`](../experiments/05_randomized_interventions/step34_randomized_interventions.py) | Randomization p-values valid on real traffic (≤ 1.5% at every A) | Partial (1/40 at 250; 0/40 at 2,500 and 10,000) | Exact by design |
| 34 | [`step34_randomized_interventions.py`](../experiments/05_randomized_interventions/step34_randomized_interventions.py) | Throttles detect failover at A=10,000 (≥ 0.5) | Fail (0.00: all 50 colluders already flagged; 1.00 at 250 and 2,500) | Selection complete |
| 34 | [`step34_randomized_interventions.py`](../experiments/05_randomized_interventions/step34_randomized_interventions.py) | Delayed failover still visible (≥ 0.2) | Pass (1.00 at 250 and 2,500) | — |
| 34 | [`step34_randomized_interventions.py`](../experiments/05_randomized_interventions/step34_randomized_interventions.py) | Benign failover teams fire the test (≥ 0.5) | Pass (0.82–0.98) | Expected limit |
| 34 | [`step34_randomized_interventions.py`](../experiments/05_randomized_interventions/step34_randomized_interventions.py) | Answer marks: adaptive ≥ 0.8 in 4 weeks; fixed ≤ 0.02; teams as often | Fail (0.62; 0.00; identical) | Toy |
| 34 | [`step34_randomized_interventions.py`](../experiments/05_randomized_interventions/step34_randomized_interventions.py) | Dual reference: ramp detection ≥ 0.5, false alarms ≤ 10% | Fail (slope 0.03; frozen alone 0.44 at 5.2%) | Frozen reference helps |
| 35 | [`step35_frozen_holdout.py`](../experiments/04_frozen_holdout/step35_frozen_holdout.py) | Frozen calibration on untouched weeks: two-scale, hollowed, T_max ≤ 2% | Pass (1.8%, 1.9%, 1.9%; week-level upper 4–6%) | About twice nominal |
| 35 | [`step35_frozen_holdout.py`](../experiments/04_frozen_holdout/step35_frozen_holdout.py) | Dataset campaigns replicate (≥ 0.9 hollowed/T_max; ≥ 0.7 two-scale) | Pass (1.00, 0.97; 0.86) | — |
| 35 | [`step35_frozen_holdout.py`](../experiments/04_frozen_holdout/step35_frozen_holdout.py) | Mimicry replicates: hollowed and T_max 0.25–0.65; two-scale ≤ 0.35 | Partial (hollowed 0.57; T_max 0.00; two-scale 0.00) | Only hollowed holds |
| 35 | [`step35_frozen_holdout.py`](../experiments/04_frozen_holdout/step35_frozen_holdout.py) | Tilted campaign ≤ 0.05 on every calibrated detector | Pass (≤ 0.04) | Ranked by hollowed (AUC 0.97) |
| 35 | [`step35_frozen_holdout.py`](../experiments/04_frozen_holdout/step35_frozen_holdout.py) | Random natural ≤ 0.05 | Pass (0.00) | — |
| 35 | [`step35_frozen_holdout.py`](../experiments/04_frozen_holdout/step35_frozen_holdout.py) | E5: T_max dataset ≥ 0.7, padded ≤ 0.6, tilted ≤ 0.1; census within pm50% | Pass (1.00, 0.00, 0.00; 3.06 vs 2.22) | Census ratio 2.22, not 1.14 |
| 35 | [`step35_frozen_holdout.py`](../experiments/04_frozen_holdout/step35_frozen_holdout.py) | Volume scan (conformal) calibrated, dataset ≥ 0.5, trails T_max under padding | Fail (0.0%; 0.03; both 0.00) | Rolling form competitive |
| 35 | [`step35_frozen_holdout.py`](../experiments/04_frozen_holdout/step35_frozen_holdout.py) | Sacrificial schedule: T_max ≥ 0.8, non-sacrificial recall ≤ 0.10 | Fail (1.00; recall 0.49) | Hollowing not steered |
| 35 | [`step35_frozen_holdout.py`](../experiments/04_frozen_holdout/step35_frozen_holdout.py) | Ramp costs T_max ≤ 0.3 | Fail (0.89 to 0.00) | Poisoning sets the quantile |
| 35 | [`step35_frozen_holdout.py`](../experiments/04_frozen_holdout/step35_frozen_holdout.py) | Martingale: T_max padded ≤ 6 weeks; first alarm ≤ 3 for both | Fail (25% of runs in 16 weeks; never) | — |
| 35 | [`step35_frozen_holdout.py`](../experiments/04_frozen_holdout/step35_frozen_holdout.py) | Whole weeks: T_max ≤ 2% and dataset ≥ 0.5 | Fail (2.5%; 1.00); weeks of 260–469 accounts only | Scale untested |
| 35 | [`step35_frozen_holdout.py`](../experiments/04_frozen_holdout/step35_frozen_holdout.py) | Run-to-run spread ≤ 0.10 | Pass (≤ 0.07) | — |
| 35 | [`step35_frozen_holdout.py`](../experiments/04_frozen_holdout/step35_frozen_holdout.py) | Addition vs replacement changes random-campaign detection ≤ 0.03 | Pass (0.00) | No injection artefact |
