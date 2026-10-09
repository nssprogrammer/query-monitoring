# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Paper numbers reproduced by this script
# Paper: "The Limits of Query Monitoring for Distributed Model Extraction".
# Run on Kaggle (see "Kaggle:" line below for hardware/time). Seeds are fixed in the configuration cell;
# re-runs vary by a few points (the paper reports run-to-run spread up to ~0.07-0.14 for weekly detection).
#
# This is the expanded rerun of Step 32. App. P records P32a–i from the first run of Step 32 (this script repeats
# those measurements as a replication, so its printed values for P32a–i may differ from the first-run outcomes quoted
# below) and P32j–p from this rerun. App. P ("A mislabelled source in Step 32") explains why the first run's
# `tilted_c05` source was discarded and removed here; the tilted results in the paper are Step 27's.
#
# | Paper location | Quantity | Value in paper | Where computed here |
# |---|---|---|---|
# | §6 'The covert end'; App. G.2; App. P (Step 32, "Iterative rewriting reaches AUC ≤ 0.65"); App. N 'Representation' | held-out discriminator AUC, real on-task prompts vs written rewrites (first pass, then two regeneration rounds) | "AUC 0.97–0.98" (§6); "AUC 0.98 … after two more rounds … 0.97" (G.2); "Fail (0.98, 0.97, 0.97)" (App. P); "AUC 0.98" (App. N) | Cell V, `crossfit(...)` → `VOI_AUCS`; prints "[W] round …" and "[W] discriminator AUC by round"; Cell Z "P32j" |
# | §6 'The covert end'; App. G.2 | rewrites that passed as user-like (posterior < 0.5 and in S_v) | "7.9%"; "395 of 5,000 rewrites (7.9%)" | Cell V, `keep` → `VOI_KEEP`, `len(VOICED)`; print "[W] discriminator AUC by round … kept"; Cell Z "P32b" |
# | App. P (Step 32, "Written prompts can be made user-like") | P32b, first run | "Fail (AUC 0.98; 4.3% pass)" | first run; same computation as above (Cell V; Cell Z "P32b") |
# | §6 'The covert end'; App. G.2; App. P (Step 32, "Distribution-matched attacker evades and gains ≥ 6") | importance-resampled (distribution-matched) campaign `matched_c05`: detection and gain at 3,000 queries | "evaded every detector and gained 4.4 points—42% of the dataset gain"; "Fail (evades; +4.4, ESS 151)" | Cell W `[W] matched_c05` row (`RES`); Cell D `DRES[("matched_c05", sd)]`; Cell Z "P32k". The 42% is relative to Step 19's all-math +10.4 |
# | §6 'The covert end'; App. G.2; App. P (P32k row) | effective sample size of the resampled rewrites | "effective sample of only 151 distinct rewrites"; "ESS 151" | Cell V, `ESS = 1/sum(wts**2)`; print "[W] … effective sample size"; Cell Z "P32k" |
# | §6 'The frontier, in one sentence'; App. G.2 | share of the detectable campaign's gain given up by covert campaigns | "58–79%" | not isolated (summary): 58% = 1 − 4.4/10.4 uses this script's `matched_c05` gain; 79% is Step 29d's tilted campaign |
# | §6; App. G.2; App. N 'Capability' | covert gains and their size in problems | "covert gains of 1.1–4.4 points are differences of 4–18 of 400 problems" (§6); "differences are 4–18 problems out of 400" (App. N) | Cell D (`tilted_v_c05` +1.1 to `matched_c05` +4.4; 400 = `S32["N_EVAL"]`) |
# | App. G.2 | student baseline | "solves 14 of 400 problems at baseline" | Cell D, `BASE`; print "[D] base GSM8K" |
# | App. G.2 | rewrites passing the first filter, distilled on their own | "The 213 rewrites that passed the first filter gained 3.5 points on their own" | first run of Step 32; plausibly the all-written campaign (`voiced_c1`, Cell D) |
# | App. P (Step 32, "Written campaign keeps capability (≥ 6)") | P32d, `voiced_c05` gain at 3,000, first run | "Fail (+2.2; 213 unique prompts)" | first run; recomputed in Cell D `voiced_c05`; Cell Z "P32d" |
# | App. P (Step 32, "Written campaign evades") | P32c, `voiced_c05` / `voiced_c1` weekly detection, first run | "Pass (0.00; 0.10 all on task)" | first run; recomputed in Cell W `[W] voiced_c05`, `[W] voiced_c1`; Cell Z "P32c" |
# | Fig. 1a (points "written in users' voice", "written, distribution-matched") | hollowed weekly detection (development weeks) vs gain at 3,000 queries | plotted (caption: "kept at most 42% of the dataset campaign's gain") | Cell W `tsRc` for `voiced_c05`, `matched_c05`; Cell D gains |
# | App. P (Step 32, "Value-defined task region rare") | P32a, benign share p_v of the value-defined region | "Pass (0.53%)" | Cell V, `P_V`; print "[V] value-defined region"; Cell Z "P32a" |
# | App. H; App. P (Step 32, "Value-defined tilted campaign ≥ 4 points") | value-defined on-task pool prompts; `tilted_v_c05` gain at 3,000 | "only 972 on-task prompts … gained 1.1 points at 3,000 queries"; "Fail (+1.1)" | Cell V `ONV_KEYS` (print "[V] … attacker-pool on-task"); Cell D `tilted_v_c05`; Cell Z "P32e" |
# | App. H; App. P (Step 32, "Personas evade query detectors, type count ≥ 0.9") | persona campaign: detection, type-count detector, gain | "evaded every query-level detector and gained 3.5 points from 1,891 queries … caught it 0.58 of the time"; "Fail (0.00; type count 0.58)" | Cell W `[W] persona` (`typecount`); Cell D `persona` (print "[D] persona … pairs"); Cell Z "P32l" |
# | Cor. 1 (§4); App. B 'The least-favourable campaign'; App. P (Step 32, "Majority-on-task accounts rare") | benign share of majority-on-task accounts q | "q = 0.25%"; "Pass (0.25%)" (the paper derives "about 14 posing accounts" at A = 10^5 from it; K* is not computed here) | Cell W, `Q_TYPE`; print "[types]"; Cell Z "P32m" |
# | App. P (Step 32, "Per-account topicality sees tilted campaigns") | P32f, held-out false alarms of the topicality HC test | "Fail (25% false alarms)" | Cell W, `FPR["topic"]`; print "[weekly] held-out FPR"; Cell Z "P32f" |
# | App. P (Step 32, "Pooling artefact ≥ 2× random") | P32g, same-week share of null-selected groups vs random | "Partial (1.9–2.1×)" | Cell P, `same` / `expct`; print "[P] … same-week share"; Cell Z "P32g" |
# | App. P (Step 32, "Per-week centring keeps multi-outlier") | P32h, per-week-centred Tmax oracle TPR at 10^4 | "Pass (1.00)" | Cell P, rows with `centre_week=True`; Cell Z "P32h" |
# | App. P (Step 32, "Multi-outlier deployable at 2,500") | P32i, deployed Tmax false alarms | "Fail (30% false alarms)" | Cell P, `DEP["fpr"]["Tmax"]`; print "[P-dep]"; Cell Z "P32i/n" |
# | §5 'Scale: selection fails before the barrier does'; App. D.5 'Why this is not yet a production result'; App. P (Step 32, "Multi-outlier deployable at 10^4") | best deployed calibration of multi-outlier selection | "5% false alarms, not 1%"; "at 5,000–7,552 account-weeks"; "Fail (5% false alarms at 7,552)" | Cell P, `DEP["fpr"]["Tmax"]` (A = 10,000 capped to 0.8 × test account-weeks); print "[P-dep]"; Cell Z "P32i/n" |
# | App. D.5 'Why this is not yet a production result' | deployed multi-outlier power at 5,000–7,552 | "caught unpadded campaigns 0.53–1.00 of the time and padded ones 0.03–0.10" | Cell P, `DEP[src]["Tmax"]`; print "[P-dep] … unique … mixed" |
# | App. K | deployed precision of the flagged group at 5,000–7,552 account-weeks | "(0.73 and 0.54)" | Cell P, `DEP["metamath_unique"]["precision"]`; print "[P-dep] … precision" (the "16–42 legitimate accounts" is derived in App. K) |
# | App. P (Step 32, "≤ 100 benign accounts flagged per week at 10^5") | P32p, reviewer workload | "Pass (44–59 at A ≥ 5,000)" | Cell P, `DEP["benign_flagged_per_week_at_1e5"]`; print "[P-dep] … benign accounts flagged"; Cell Z "P32i/n" |
# | §5 'Scale: selection fails before the barrier does'; App. D.5 'For providers'; App. L item 6; App. P (Step 32, "Linked 30%: expansion ≥ 0.8 …") | seeded hollowed expansion from 30% linked colluders (15 of 50, +50 accounts) at 2,500–10,000 | "recovered 96–100% of the rest"; "Partial (1.00, 0.96; metadata 0.67)" | Cell P, `seeded(...)` → `link_seeded`, `link_recall`, `link_meta_only`; print "[P] … linked colluders"; Cell Z "P32o" |
# | App. F Table 11 (cols "Tmax unpadded / padded", rows A = 2,500 and 10,000) | multi-outlier oracle TPR in pooled populations | "1.00 / 0.83" and "1.00 / 0.17" | plausibly replicated by Cell P rows with `centre_week=False` (`row["Tmax"]`); the table's primary source is the Step 27–31 script |
#
# Not reported in the paper: the rerun's own `voiced_c05` / `voiced_c1` detection and gains (the paper quotes the first
# run for these), the control rows of Cell W (`metamath_unique`, `metamath_mixed`, `random`) and their oracle TPRs,
# per-detector held-out FPRs other than topicality, top-eigenvector (`tsRc`) and two-scale (`tsC`) rows of Cell P,
# null group sizes, the importance-resampled set size, PCA variance and data-loading diagnostics.

# %% [markdown]
# # Step 32 — The decisive attacker (written prompts in the users' voice), a value-defined task region, per-account topicality, and the pooling artefact
# **Kaggle:** GPU (T4/P100) + Internet + `HF_TOKEN` → **Run All** (≈4.5–5 h; reuses earlier caches). `QUICK=True` first (≈30 min).
#
# **Rerun note.** The first run of Step 32 (P32a–i) is recorded in the paper; this expanded version repeats those measurements as a replication and adds P32j–p. Its cache file is `step32b_…` so the first run's two-item cache is not misread, and the mislabelled `tilted_c05` source of the first run is removed (Step 27 holds the valid tilted results).
#
# * **V — task region by value.** S_v is where a logistic classifier trained on human-written **GSM8K** training questions (the target) against WildChat gives posterior ≥ 0.5; p_v is the benign share. A tilted campaign draws its on-task half from attacker-pool prompts in S_v.
# * **W — the written attacker.** `Qwen/Qwen2.5-1.5B-Instruct` rewrites MetaMathQA GSM-type problems as messages from WildChat math users, shown three real on-task user messages as style examples; every rewrite keeps the MetaMathQA answer, so only the prompt's style changes. Rewrites are kept only if a cross-fitted classifier cannot tell them from real on-task WildChat prompts (posterior < 0.5) and they fall in S_v. Campaigns: `voiced_c05` (half rewritten on-task, half off-task pool prompts) and `voiced_c1` (all rewritten).
# * **T — per-account topicality** (calibrated out of sample: reference from weeks before the 8 calibration weeks). Real users are individually topical; Proposition 4 assumes accounts that send draws from pooled traffic. A two-sided Higher-Criticism statistic on per-account log S̄ (calibrated per volume bin on prior weeks) tests whether tilted and written campaigns' accounts look unlike users.
# * **P — the pooling artefact and deployable calibration at scale.** Same-week share of the groups multi-outlier selection picks in pooled null windows (against the random expectation); multi-outlier selection with rows centred **per source week**; and deployed T_max at A = 2,500 with per-week centring and conformal thresholds from an earlier period (weeks 8–35, tested on 36–55).
# * **D — distillation.** SmolLM2-360M, full fine-tuning, GSM8K (400 problems), 3,000 queries, two seeds: voiced_c05, voiced_c1, tilted_v_c05 (Step 19/29d references at 3,000: all-math +10.4, half padding +7.6, tilted +2.2).
#
# | ID | Locked prediction (locked in this notebook; outcomes in App. P (prediction log)) |
# |---|---|
# | **P32a** | The value-defined task region is rare: p_v ≤ 5% |
# | **P32b** | Written prompts can be made user-like: held-out classifier AUC (real on-task vs written) ≤ 0.75, and ≥ 50% of rewrites pass the filter |
# | **P32c** | The written attacker evades: voiced_c05 weekly detection ≤ 0.10 (two-scale) and ≤ 0.25 (hollowed, multi-outlier; conformal) |
# | **P32d** | The written attacker keeps capability: voiced_c05 gain at 3,000 queries ≥ 6 points (the frontier breaks) |
# | **P32e** | Defining the region by value helps the tilted attacker: tilted_v_c05 gain at 3,000 ≥ 4 points (2.2 for the MetaMathQA-defined region) |
# | **P32f** | Per-account topicality detects the tilted campaign: two-sided HC ≥ 0.3 weekly at held-out false alarms ≤ 2% |
# | **P32g** | The pooling artefact is present: same-week share of null-selected outlier groups ≥ 2× the random expectation |
# | **P32h** | Per-source-week centring keeps multi-outlier selection: unpadded oracle TPR at 1% ≥ 0.8 at A = 10,000 |
# | **P32i** | Multi-outlier selection is deployable at A = 2,500: held-out false alarms ≤ 2% and unpadded power ≥ 0.5 with earlier-period conformal thresholds |
# | **P32j** | Iterative rewriting reaches a held-out discriminator AUC ≤ 0.65 by the final round |
# | **P32k** | The importance-resampled upper-bound attacker evades (hollowed and multi-outlier weekly ≤ 0.10) and gains ≥ 6 points at 3,000 queries |
# | **P32l** | Persona campaigns (Corollary 1) evade query-level detectors (≤ 0.25) but are caught by the type-count detector in weekly populations (≥ 0.9) |
# | **P32m** | Majority-on-task accounts are rare: q ≤ 3% |
# | **P32n** | Multi-outlier selection is deployable at A = 10,000: held-out false alarms ≤ 2% and unpadded power ≥ 0.5 |
# | **P32o** | With 30% of colluders linked, seeded hollowed expansion detects padded campaigns at A = 10,000 (oracle ≥ 0.8, colluder recall ≥ 0.5), against ≤ 0.5 for metadata alone |
# | **P32p** | Reviewer workload: at 10⁵ accounts the deployed multi-outlier test flags ≤ 100 benign accounts per week |

# %%
# ===== Cell 1: configuration =====
import math, json, time, os, gc, re, pickle
import numpy as np
QUICK = False
SYNTHETIC_TEST = False
SEED = 9091
OUTDIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
try:
    import torch
    HAS_TORCH = True; DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    HAS_TORCH, DEV = False, None
USE_GPU = HAS_TORCH and DEV.type == "cuda"
if not USE_GPU and not SYNTHETIC_TEST:
    raise RuntimeError("No GPU visible. Kaggle: Settings -> Accelerator -> GPU, then restart.")
CHUNK = 32768 if USE_GPU else 8192
H_COARSE, H_FINE, M_MODES, ALPHA, FLOOR = 4.0, 0.5, 300, 0.01, 1e-4
VOL_CAP, MIN_PER_ACCT, POOL_FRAC, SUB_FRAC = 500, 10, 0.2, 0.7
N_TOPICS, N_REF_BATCH, REF_BATCH, MMD_WIN, MINCELL = 100, 20, 500, 2000, 20
DETS = ["twoscale", "qc", "qcount", "mmd", "gmmd", "glob", "spec", "fraudar"]
SRC_MAIN = ["metamath_unique", "metamath_mixed", "metamath_followup", "gsm8k_human", "random"]
EVALS = [("eval_gsm8k", 3, 300), ("eval_gsm8k", 10, 100), ("eval_mmlu", 3, 300), ("eval_mmlu", 10, 100)]
SRC_EXTRA = ["wild_cluster"]
CAMO_FRAC = 0.2
if QUICK:
    WK = dict(KS=[20], CAL_N=4, LAGB=2, N_TEST=3, NULL_PER=12, REPS=2)
    POP = dict(AS=[40, 70], K=6, N_FIX=200, STEP=4, FIRST=12, N_CAL=12, N_TEST=6, REPS=3)
    EVALS = [("eval_gsm8k", 3, 40), ("eval_mmlu", 3, 40)]
    MMD_WIN, N_REF_BATCH, REF_BATCH = 400, 5, 200
else:
    WK = dict(KS=[50], CAL_N=8, LAGB=4, N_TEST=24, NULL_PER=40, REPS=4)
    POP = dict(AS=[250, 500, 1000], K=50, N_FIX=1640, STEP=4, FIRST=20, N_CAL=40, N_TEST=20, REPS=10)
CACHE11 = f"{OUTDIR}/step11_wildchat_E384_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
CACHE13 = f"{OUTDIR}/step13_distill_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
CACHE14 = f"{OUTDIR}/step14_meta_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
CACHE15 = f"{OUTDIR}/step15_extra_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
N_CONV = 30000 if QUICK else None; DIMS = [384]
print(f"backend {'GPU ' + torch.cuda.get_device_name(0) if USE_GPU else 'CPU'} | QUICK={QUICK}")
CACHE18 = f"{OUTDIR}/step18_evals_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
import contextlib
@contextlib.contextmanager
def modes(m):
    global M_MODES
    old = M_MODES; M_MODES = m
    try: yield
    finally: M_MODES = old
if QUICK: S27 = dict(K=20, REPS=2, AS=[30, 60, 120], NUL={30: 25, 60: 20, 120: 15}, ATT=6, KP=6, NP=200, A_HI=60, M_HI=150, NUL_HI=15, ATT_HI=6, PF_N=300)
else:     S27 = dict(K=50, REPS=4, AS=[250, 1000, 2500, 5000, 10000], NUL={250: 150, 1000: 120, 2500: 100, 5000: 80, 10000: 50}, ATT=30, KP=50, NP=1640, A_HI=2500, M_HI=750, NUL_HI=100, ATT_HI=30, PF_N=4000)
GEN_MODEL, STUDENT = "Qwen/Qwen2.5-1.5B-Instruct", "HuggingFaceTB/SmolLM2-360M"
if QUICK: S32 = dict(ROUNDS=1, N_GEN=96, GEN_BS=16, MAX_NEW_GEN=80, A_P=[60, 120], NUL_P=12, ATT_P=4, A_DEPS=[60], LINKED=2, EXPAND=6, A_DEP=60, CAL_DEP=20, TEST_DEP=10, ATT_DEP=6, BUD=60, SEEDS=[0], N_EVAL=24, MAX_LEN=384, BS=8, GEN_BS_EVAL=16, MAX_NEW=96)
else:     S32 = dict(ROUNDS=2, N_GEN=5000, GEN_BS=48, MAX_NEW_GEN=110, A_P=[2500, 10000], NUL_P=80, ATT_P=30, A_DEPS=[2500, 5000, 10000], LINKED=15, EXPAND=50, A_DEP=2500, CAL_DEP=100, TEST_DEP=60, ATT_DEP=30, BUD=3000, SEEDS=[0, 1], N_EVAL=400, MAX_LEN=512, BS=16, GEN_BS_EVAL=32, MAX_NEW=320)
CACHE32 = f"{OUTDIR}/step32b_voiced_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.pkl"
CACHE19 = f"{OUTDIR}/step19_mm_rows_{'quick' if QUICK else 'full'}.pkl"

# %% [markdown]
# ## Part 0 — caches

# %%
# ===== Cell 2: full 384-d MiniLM embeddings of English user turns (cached as float16) =====
if os.path.exists(CACHE11):
    _c = np.load(CACHE11); E16, ACCT, WEEK = _c["E16"], _c["ACCT"], _c["WEEK"]; print(f"loaded cache: {E16.shape}")
else:
    t0 = time.time()
    if SYNTHETIC_TEST:
        rng = np.random.default_rng(0); n_ip, n_rows = 3000, 60000
        ACCT = rng.integers(0, n_ip, n_rows).astype(np.int32); WEEK = rng.integers(0, 30, n_rows).astype(np.int16)
        topic = rng.integers(0, 40, n_ip)[ACCT]; base = rng.standard_normal((40, 384))
        E = (base[topic] * 0.8 + rng.standard_normal((n_rows, 384)) * 0.6).astype(np.float32)
    else:
        from datasets import load_dataset
        tok = None
        try:
            from kaggle_secrets import UserSecretsClient; tok = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            tok = os.environ.get("HF_TOKEN")
        split = "train" if N_CONV is None else f"train[:{N_CONV}]"
        ds = load_dataset("allenai/WildChat-1M", split=split, token=tok).select_columns(["conversation", "hashed_ip", "timestamp", "language"])
        texts, ips, tss = [], [], []
        for batch in ds.iter(batch_size=2000):
            for conv, ip, ts, lang in zip(batch["conversation"], batch["hashed_ip"], batch["timestamp"], batch["language"]):
                if lang != "English" or ip is None or ts is None: continue
                for turn in conv:
                    if turn.get("role") == "user" and turn.get("content"):
                        texts.append(turn["content"][:1000]); ips.append(ip); tss.append(ts)
        print(f"{len(texts)} English user turns from {len(set(ips))} IPs ({time.time()-t0:.0f}s)")
        _, ACCT = np.unique(np.array(ips), return_inverse=True); ACCT = ACCT.astype(np.int32)
        days = np.array([np.datetime64(t.replace(tzinfo=None) if hasattr(t, "replace") else t, "D") for t in tss])
        WEEK = ((days - days.min()).astype(np.int64) // 7).astype(np.int16); del ds; gc.collect()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            os.system("pip install -q sentence-transformers"); from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda"); enc.max_seq_length = 256
        E = enc.encode(texts, batch_size=512, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        del texts; gc.collect()
    E16 = E.astype(np.float16); del E; gc.collect()
    np.savez(CACHE11, E16=E16, ACCT=ACCT, WEEK=WEEK); print(f"cached ({time.time()-t0:.0f}s)")
n_w = int(WEEK.max()) + 1
# PCA basis on a 100k fit sample (nested: the first d components of the same basis for every d)
rng = np.random.default_rng(SEED); fit = E16[rng.choice(len(E16), min(100000, len(E16)), replace=False)].astype(np.float32)
MU_E = fit.mean(0); ew, U = np.linalg.eigh(np.cov(fit - MU_E, rowvar=False)); EW, UB = ew[::-1], U[:, ::-1].copy()
for d in DIMS: print(f"d={d:3d}: explains {EW[:d].sum()/EW.sum():.3f} of embedding variance")
def make_Z(d):
    W_ = UB[:, :d] / math.sqrt(EW[:d].sum() / 4.0); Z = np.empty((len(E16), d), dtype=np.float32)
    for s0 in range(0, len(E16), 200000): Z[s0:s0 + 200000] = (E16[s0:s0 + 200000].astype(np.float32) - MU_E) @ W_
    return Z


# %%
# ===== Cell 3: conversation ids, turn indices and attacker-pool text pairs (same order as the cache), MetaMathQA pairs =====
import pickle
rng = np.random.default_rng(SEED + 1); n_acct = int(ACCT.max()) + 1; is_pool = rng.random(n_acct) < POOL_FRAC    # identical to Steps 12-13
PAIRS_PKL = CACHE14.replace(".npz", "_pairs.pkl")
if os.path.exists(CACHE14) and os.path.exists(PAIRS_PKL):
    _c = np.load(CACHE14); CONV, TURN, D_MMU = _c["CONV"], _c["TURN"], _c["D_MMU"]
    with open(PAIRS_PKL, "rb") as fh: POOL_TXT, MM_PAIRS = pickle.load(fh)
    print(f"loaded metadata cache: {len(CONV)} turns, {len(POOL_TXT)} pool pairs, {len(MM_PAIRS)} MetaMathQA pairs")
else:
    t0 = time.time()
    if SYNTHETIC_TEST:
        r = np.random.default_rng(5); CONV = np.cumsum(r.random(len(E16)) < 0.4).astype(np.int32); TURN = np.ones(len(E16), np.int16)
        for i in range(1, len(E16)):
            if CONV[i] == CONV[i - 1]: TURN[i] = TURN[i - 1] + 1
        POOL_TXT = {int(i): (f"user prompt {i}", f"assistant reply {i} " * 5) for i in np.flatnonzero(is_pool[ACCT])}
        MM_PAIRS = [(f"What is {a} plus {a+1}?", f"{a} + {a+1} = {2*a+1}. The answer is: {2*a+1}") for a in range(1500)]
        D_MMU = (np.random.default_rng(3).standard_normal(384) * 2 + np.random.default_rng(4).standard_normal((1500, 384)) * 0.7).astype(np.float16)
    else:
        from datasets import load_dataset
        tok = None
        try:
            from kaggle_secrets import UserSecretsClient; tok = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            tok = os.environ.get("HF_TOKEN")
        split = "train" if N_CONV is None else f"train[:{N_CONV}]"
        ds = load_dataset("allenai/WildChat-1M", split=split, token=tok).select_columns(["conversation", "hashed_ip", "timestamp", "language"])
        ips, conv_id, turn_ix, users, replies = [], [], [], [], []; cid = 0
        for batch in ds.iter(batch_size=2000):
            for conv, ip, ts, lang in zip(batch["conversation"], batch["hashed_ip"], batch["timestamp"], batch["language"]):
                if lang != "English" or ip is None or ts is None: continue
                t = 0
                for j, turn in enumerate(conv):
                    if turn.get("role") == "user" and turn.get("content"):
                        t += 1; nxt = next((c.get("content") or "" for c in conv[j + 1:] if c.get("role") == "assistant"), "")
                        ips.append(ip); conv_id.append(cid); turn_ix.append(t); users.append(turn["content"][:1000]); replies.append(nxt[:1500])
                cid += 1
        _, ACCT2 = np.unique(np.array(ips), return_inverse=True)
        assert len(ACCT2) == len(ACCT) and np.array_equal(ACCT2.astype(np.int32), ACCT), "re-read order does not match the Step 11 cache"
        CONV, TURN = np.array(conv_id, np.int32), np.array(turn_ix, np.int16)
        POOL_TXT = {int(i): (users[i], replies[i]) for i in np.flatnonzero(is_pool[ACCT])}
        del users, replies, ds; gc.collect()
        print(f"re-read WildChat: order verified, {len(CONV)} turns in {int(CONV.max())+1} conversations, {len(POOL_TXT)} pool pairs ({time.time()-t0:.0f}s)")
        mm = load_dataset("meta-math/MetaMathQA", split="train", token=tok).select_columns(["query", "response", "original_question"])
        groups = {}
        for q, a, o in zip(mm["query"], mm["response"], mm["original_question"]):
            if q and a and o: groups.setdefault(o, []).append((q.strip()[:1000], a.strip()[:2000]))
        rr = np.random.default_rng(SEED); origs = list(groups); rr.shuffle(origs)
        MM_PAIRS = [groups[o][int(rr.integers(len(groups[o])))] for o in origs]          # one (query, response) per source problem
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            os.system("pip install -q sentence-transformers"); from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda"); enc.max_seq_length = 256
        D_MMU = enc.encode([q for q, _ in MM_PAIRS], batch_size=512, convert_to_numpy=True, normalize_embeddings=True).astype(np.float16)
        print(f"MetaMathQA: {len(MM_PAIRS)} (query, response) pairs, one per source problem ({time.time()-t0:.0f}s)")
    np.savez(CACHE14, CONV=CONV, TURN=TURN, D_MMU=D_MMU)
    with open(PAIRS_PKL, "wb") as fh: pickle.dump((POOL_TXT, MM_PAIRS), fh)
DIST = {"metamath_unique": D_MMU}
print(f"follow-up turns (turn >= 2) among all turns: {np.mean(TURN >= 2):.3f}")

# %%
# ===== Cell 3: MetaMathQA grouped by source problem (unique vs rephrased) + Magicoder, MiniLM, cached =====
if os.path.exists(CACHE13):
    _c = np.load(CACHE13); DIST13 = {k[2:]: _c[k] for k in _c.files if k.startswith("D_")}; MM_GROUP = _c["MM_GROUP"]; print({k: v.shape for k, v in DIST13.items()})
else:
    t0 = time.time(); DIST13 = {}
    if SYNTHETIC_TEST:
        r = np.random.default_rng(3); c = r.standard_normal(384) * 2
        base = c + r.standard_normal((1500, 384)) * 0.7
        DIST13["metamath_unique"] = base.astype(np.float16)
        reph = np.repeat(base[:600], 4, axis=0) + r.standard_normal((2400, 384)) * 0.05; DIST13["metamath_rephrase"] = reph.astype(np.float16)
        MM_GROUP = np.repeat(np.arange(600), 4)
        DIST13["magicoder"] = (r.standard_normal(384) * 2 + r.standard_normal((2000, 384)) * 0.7).astype(np.float16)
    else:
        from datasets import load_dataset
        tok = None
        try:
            from kaggle_secrets import UserSecretsClient; tok = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            tok = os.environ.get("HF_TOKEN")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            os.system("pip install -q sentence-transformers"); from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda"); enc.max_seq_length = 256
        emb = lambda T: enc.encode(T, batch_size=512, convert_to_numpy=True, normalize_embeddings=True).astype(np.float16)
        mm = load_dataset("meta-math/MetaMathQA", split="train", token=tok).select_columns(["query", "original_question"])
        qs, oq = mm["query"], mm["original_question"]; groups = {}
        for q, o in zip(qs, oq):
            if q and o: groups.setdefault(o, set()).add(q.strip()[:1000])
        rng = np.random.default_rng(SEED); origs = list(groups); rng.shuffle(origs)
        uniq = [sorted(groups[o])[int(rng.integers(len(groups[o])))] for o in origs]
        multi = [o for o in origs if len(groups[o]) >= 4][:6000]
        reph, gid = [], []
        for j, o in enumerate(multi):
            for q in rng.permutation(sorted(groups[o]))[:4]: reph.append(q); gid.append(j)
        DIST13["metamath_unique"] = emb(uniq); DIST13["metamath_rephrase"] = emb(reph); MM_GROUP = np.array(gid)
        print(f"MetaMathQA: {len(uniq)} source problems (unique set), {len(multi)} problems x 4 variants (rephrase set) ({time.time()-t0:.0f}s)")
        mg = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", token=tok)
        idx = np.random.default_rng(SEED).choice(len(mg), min(25000, len(mg)), replace=False)
        DIST13["magicoder"] = emb(list(dict.fromkeys(str(mg[int(i)]["problem"])[:1000] for i in idx)))
        print(f"Magicoder: {len(DIST13['magicoder'])} prompts ({time.time()-t0:.0f}s)")
    np.savez(CACHE13, MM_GROUP=MM_GROUP, **{f"D_{k}": v for k, v in DIST13.items()})

# %%
# ===== Cell 4: extra sources — human GSM8K questions, templated MetaMathQA, natural-likeness scores (cached) =====
if os.path.exists(CACHE15):
    _c = np.load(CACHE15); D_GSM, D_TMPL, NAT = _c["D_GSM"], _c["D_TMPL"], _c["NAT"]
else:
    t0 = time.time()
    if SYNTHETIC_TEST:
        r = np.random.default_rng(11); D_GSM = (D_MMU[:1000].astype(np.float32) + r.standard_normal((1000, 384)) * 0.3).astype(np.float16)
        D_TMPL = (D_MMU.astype(np.float32) * 0.7 + 0.3).astype(np.float16)
    else:
        from datasets import load_dataset
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            os.system("pip install -q sentence-transformers"); from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda"); enc.max_seq_length = 256
        emb = lambda T: enc.encode(T, batch_size=512, convert_to_numpy=True, normalize_embeddings=True).astype(np.float16)
        g = load_dataset("openai/gsm8k", "main", split="train")
        D_GSM = emb([q[:1000] for q in g["question"]])                                    # human-written math word problems
        TEMPLATE = "You are a careful math tutor. Solve the following problem step by step and state the final answer.\n\n"
        D_TMPL = emb([TEMPLATE + q for q, _ in MM_PAIRS])                                   # same problems, fixed template prefix
    # natural-likeness of each MetaMathQA prompt: mean cosine similarity to its 10 nearest benign WildChat turns
    ben = np.flatnonzero(~is_pool[ACCT]); smp = np.random.default_rng(3).choice(ben, min(50000, len(ben)), replace=False)
    B_ = E16[smp].astype(np.float32); Q_ = D_MMU.astype(np.float32); NAT = np.empty(len(Q_), np.float32)
    for s0 in range(0, len(Q_), 2048):
        if USE_GPU:
            sims = torch.as_tensor(Q_[s0:s0 + 2048], device=DEV) @ torch.as_tensor(B_, device=DEV).T
            NAT[s0:s0 + 2048] = sims.topk(10, dim=1).values.mean(1).cpu().numpy()
        else:
            sims = Q_[s0:s0 + 2048] @ B_.T; NAT[s0:s0 + 2048] = np.sort(sims, 1)[:, -10:].mean(1)
    np.savez(CACHE15, D_GSM=D_GSM, D_TMPL=D_TMPL, NAT=NAT); print(f"extra sources built ({time.time()-t0:.0f}s)")
CAMO = np.flatnonzero(NAT >= np.quantile(NAT, 1 - CAMO_FRAC))
DIST = {"metamath_unique": D_MMU, "metamath_rephrase": DIST13["metamath_rephrase"], "gsm8k_human": D_GSM, "metamath_template": D_TMPL}
print(f"sources: " + ", ".join(f"{k}:{len(v)}" for k, v in DIST.items()) + f" | camouflage pool {len(CAMO)} (natural-likeness >= {np.quantile(NAT, 1 - CAMO_FRAC):.3f}; all MetaMathQA median {np.median(NAT):.3f})")
# ---- benchmark sweeps for the benign evaluation-pipeline test (GSM8K test questions, MMLU questions)
if os.path.exists(CACHE18):
    _c = np.load(CACHE18); D_EG, D_EM = _c["D_EG"], _c["D_EM"]
else:
    if SYNTHETIC_TEST:
        r = np.random.default_rng(21); D_EG = (D_GSM[:300].astype(np.float32) + r.standard_normal((300, 384)) * 0.2).astype(np.float16)
        D_EM = (r.standard_normal(384) * 2 + r.standard_normal((800, 384)) * 0.6).astype(np.float16)
    else:
        from datasets import load_dataset
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            os.system("pip install -q sentence-transformers"); from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda"); enc.max_seq_length = 256
        emb = lambda T: enc.encode(T, batch_size=512, convert_to_numpy=True, normalize_embeddings=True).astype(np.float16)
        D_EG = emb([q[:1000] for q in load_dataset("openai/gsm8k", "main", split="test")["question"]])
        mm = load_dataset("cais/mmlu", "all", split="test")
        D_EM = emb([(q + "\n" + "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(ch)))[:1000] for q, ch in zip(mm["question"], mm["choices"])])
    np.savez(CACHE18, D_EG=D_EG, D_EM=D_EM)
DIST["eval_gsm8k"], DIST["eval_mmlu"] = D_EG, D_EM
print(f"evaluation sweeps: GSM8K test {len(D_EG)}, MMLU {len(D_EM)}")

# %% [markdown]
# ## V/W — value-defined region and the written attacker

# %%
# ===== Cell V: value-defined task region (GSM8K vs chat) and the written attacker =====
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
neg = np.flatnonzero((~is_pool[ACCT]) & (WEEK < 8)); neg = np.random.default_rng(4).choice(neg, min(4 * len(D_GSM), len(neg)), replace=False)
clf_v = LogisticRegression(max_iter=2000, class_weight="balanced").fit(np.vstack([D_GSM.astype(np.float32), E16[neg].astype(np.float32)]), np.r_[np.ones(len(D_GSM)), np.zeros(len(neg))])
POSTV = np.zeros(len(E16), np.float32)
for s0 in range(0, len(E16), 200000): POSTV[s0:s0 + 200000] = clf_v.predict_proba(E16[s0:s0 + 200000].astype(np.float32))[:, 1]
# [PAPER] App. P (Step 32, P32a): value-defined region benign share p_v 0.53% (first run; recomputed here)
SV = POSTV >= 0.5; P_V = float(SV[~is_pool[ACCT]].mean())
POOL_KEYS = np.array(sorted(i for i in POOL_TXT if POOL_TXT[i][0] and POOL_TXT[i][1])); ONV_KEYS = POOL_KEYS[SV[POOL_KEYS]]
if len(ONV_KEYS) < 50:
    ONV_KEYS = POOL_KEYS[np.argsort(-POSTV[POOL_KEYS])[: max(50, len(POOL_KEYS) // 50)]]; print(f"[V] WARNING: few on-task pool prompts; using the top {len(ONV_KEYS)} by posterior")
OFFV_KEYS = np.setdiff1d(POOL_KEYS, ONV_KEYS)
# [PAPER] App. H: the value-defined region left only 972 on-task prompts (len(ONV_KEYS))
print(f"[V] value-defined region: benign share p_v = {P_V:.4f}; attacker-pool on-task {len(ONV_KEYS)}, off-task {len(OFFV_KEYS)}")
if os.path.exists(CACHE32):
    with open(CACHE32, "rb") as fh: VOICED, D_VOI, MATCHED, D_MAT, VOI_AUCS, VOI_KEEP, VOI_RELAXED, ESS = pickle.load(fh)
    VOI_AUC = VOI_AUCS[-1]
else:
    if os.path.exists(CACHE19):
        with open(CACHE19, "rb") as fh: GSM_ROWS, _ = pickle.load(fh)
    elif SYNTHETIC_TEST:
        GSM_ROWS = [(f"Tom has {i} apples and buys {i+3} more. How many apples does he have?", f"He has {2*i+3}. The answer is: {2*i+3}") for i in range(2000)]
    else:
        from datasets import load_dataset
        mmq = load_dataset("meta-math/MetaMathQA", split="train").select_columns(["query", "response", "type"])
        GSM_ROWS = [(q.strip()[:1000], a.strip()[:2000]) for q, a, ty in zip(mmq["query"], mmq["response"], mmq["type"]) if q and a and ty and ty.startswith("GSM")]
    src_rows = [GSM_ROWS[i] for i in np.random.default_rng(32).choice(len(GSM_ROWS), S32["N_GEN"], replace=False)]
    style = [POOL_TXT[k][0][:220].replace("\n", " ") for k in ONV_KEYS] or ["can you help me with this math question"]
    real = E16[np.random.default_rng(8).choice(ONV_KEYS, min(len(ONV_KEYS), max(len(src_rows), 50)), replace=len(ONV_KEYS) < 50)].astype(np.float32)
    def crossfit(Dw):
        iw = np.random.default_rng(9).permutation(len(Dw)); ir = np.random.default_rng(10).permutation(len(real)); P = np.zeros(len(Dw)); au = []
        for f in range(2):
            tw, hw = iw[f::2], iw[1 - f::2]; tr, hr = ir[f::2], ir[1 - f::2]
            dc = LogisticRegression(max_iter=2000, class_weight="balanced").fit(np.vstack([Dw[tw], real[tr]]), np.r_[np.ones(len(tw)), np.zeros(len(tr))])
            P[hw] = dc.predict_proba(Dw[hw])[:, 1]; au.append(roc_auc_score(np.r_[np.ones(len(hw)), np.zeros(len(hr))], np.r_[P[hw], dc.predict_proba(real[hr])[:, 1]]))
        return P, float(np.mean(au))
    if SYNTHETIC_TEST:
        voiced_txt = [q for q, _ in src_rows]
        def embed(T): return (D_MMU[np.arange(len(T)) % len(D_MMU)].astype(np.float32) * 0.6 + real.mean(0) * 0.4)
        def regen(qs, strong): return list(qs)
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            os.system("pip install -q sentence-transformers"); from sentence_transformers import SentenceTransformer
        gtok = AutoTokenizer.from_pretrained(GEN_MODEL); gtok.padding_side = "left"
        gm = AutoModelForCausalLM.from_pretrained(GEN_MODEL, dtype=torch.float16).to(DEV); gm.eval(); rr = np.random.default_rng(7)
        enc_m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
        def embed(T): return enc_m.encode(T, batch_size=512, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        def regen(qs, strong):
            outs = []
            for s0 in range(0, len(qs), S32["GEN_BS"]):
                msgs = []
                for q in qs[s0:s0 + S32["GEN_BS"]]:
                    ex = "\n".join(f"- {style[i]}" for i in rr.choice(len(style), 3, replace=len(style) < 3))
                    extra = " Match their length, casing and punctuation closely; do not sound like a textbook or a worksheet." if strong else ""
                    msgs.append(gtok.apply_chat_template([{"role": "user", "content": "Here are messages that real people typed into a chatbot when asking for help with math:\n" + ex +
                        "\n\nRewrite the following problem as a message from one of these people, in their style and voice (informal, maybe a little context about why they ask). Keep every number and exactly what is being asked." + extra + " Output only the message.\n\nProblem: " + q}], tokenize=False, add_generation_prompt=True))
                enc = gtok(msgs, return_tensors="pt", padding=True).to(DEV)
                with torch.no_grad(): out = gm.generate(**enc, max_new_tokens=S32["MAX_NEW_GEN"], do_sample=True, temperature=0.9 if strong else 0.8, top_p=0.95, pad_token_id=gtok.eos_token_id)
                outs += [gtok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().strip('"').split("\n\n")[0][:1000] for o in out]
                if s0 % (10 * S32["GEN_BS"]) == 0: print(f"[W] generated {len(outs)}/{len(qs)} (strong={strong}); e.g. {outs[-1][:120]!r}")
            return outs
        voiced_txt = regen([q for q, _ in src_rows], False)
    # [PAPER] §6 / App. G.2: discriminator AUC 0.98, then 0.97 after two regeneration rounds; App. P (P32j): 0.98, 0.97, 0.97
    D_all = embed(voiced_txt); P_gen, auc0 = crossfit(D_all); VOI_AUCS = [auc0]
    for rnd in range(S32.get("ROUNDS", 2)):
        bad = np.flatnonzero(P_gen >= 0.5)
        if len(bad) == 0: break
        new_txt = regen([src_rows[i][0] for i in bad], True)
        for j, i in enumerate(bad): voiced_txt[i] = new_txt[j]
        D_all[bad] = embed(new_txt); P_gen, aucr = crossfit(D_all); VOI_AUCS.append(aucr)
        print(f"[W] round {rnd + 1}: regenerated {len(bad)}; held-out discriminator AUC {aucr:.3f}")
    if not SYNTHETIC_TEST:
        del gm, enc_m; gc.collect(); torch.cuda.empty_cache()
    inS = clf_v.predict_proba(D_all)[:, 1] >= 0.5
    # [PAPER] §6 / App. G.2: 395 of 5,000 rewrites (7.9%) passed as user-like (App. P P32b, first run: 4.3% pass)
    keep = (P_gen < 0.5) & inS; VOI_RELAXED = False
    if keep.sum() < 200:
        VOI_RELAXED = True; keep = P_gen <= np.quantile(P_gen, 0.5)
        print(f"[W] FILTER RELAXED: fewer than 200 rewrites passed the registered filter; using the least-distinguishable half ({int(keep.sum())}). P32b fails.")
    VOICED = [(voiced_txt[i], src_rows[i][1]) for i in np.flatnonzero(keep)]; D_VOI = D_all[keep].astype(np.float16)
    # importance-resampled upper bound: resample all in-region rewrites with weights = odds(real | x)
    cand = np.flatnonzero(inS) if inS.sum() >= 50 else np.arange(len(D_all)); wts = np.clip((1 - P_gen[cand]) / np.maximum(P_gen[cand], 1e-3), 0, 50); wts = wts / wts.sum()
    mi = np.random.default_rng(11).choice(cand, len(cand), replace=True, p=wts); MATCHED = [(voiced_txt[i], src_rows[i][1]) for i in mi]; D_MAT = D_all[mi].astype(np.float16)
    # [PAPER] §6 / App. G.2 / App. P (P32k): effective sample of 151 distinct rewrites
    ESS = float(1.0 / np.sum(wts ** 2))
    VOI_AUC, VOI_KEEP = VOI_AUCS[-1], float(keep.mean())
    with open(CACHE32, "wb") as fh: pickle.dump((VOICED, D_VOI, MATCHED, D_MAT, VOI_AUCS, VOI_KEEP, VOI_RELAXED, ESS), fh)
    print(f"[W] discriminator AUC by round {[round(x, 3) for x in VOI_AUCS]}; kept {VOI_KEEP:.3f} ({len(VOICED)}); relaxed {VOI_RELAXED}; importance-resampled set {len(MATCHED)} (effective sample size {ESS:.0f})")
DIST["voiced"] = D_VOI if len(D_VOI) else D_MMU[:10]
DIST["matched"] = D_MAT if len(D_MAT) else D_MMU[:10]
print(f"[W] written on-task prompts available: {len(D_VOI)}")

# %% [markdown]
# ## Statistics

# %%
# ===== Cell 4: projection, topics, natural source cluster with farthest-point order =====
from sklearn.cluster import MiniBatchKMeans
rng = np.random.default_rng(SEED + 1); n_acct = int(ACCT.max()) + 1; is_pool = rng.random(n_acct) < POOL_FRAC; pool_idx = np.flatnonzero(is_pool[ACCT])
Z = make_Z(384); W384 = UB[:, :384] / math.sqrt(EW[:384].sum() / 4.0); EN = E16
DZ = {k: ((v.astype(np.float32) - MU_E) @ W384).astype(np.float32) for k, v in DIST.items()}
t0 = time.time(); ben = np.flatnonzero(~is_pool[ACCT])
km = MiniBatchKMeans(n_clusters=N_TOPICS, random_state=SEED, batch_size=4096, n_init=3).fit(Z[np.random.default_rng(1).choice(ben, min(200000, len(ben)), replace=False)])
TOPIC = km.predict(Z).astype(np.int32)
CL10 = MiniBatchKMeans(n_clusters=10, random_state=SEED + 1, batch_size=4096, n_init=3).fit(Z[pool_idx]).labels_
WILD = pool_idx[CL10 == int(np.argmax(np.bincount(CL10)))]
def fps(X, N, seed=0):
    N = min(N, len(X)); r = np.random.default_rng(seed); i0 = int(r.integers(len(X)))
    if USE_GPU:
        Xt = torch.as_tensor(X, dtype=torch.float32, device=DEV); d2 = ((Xt - Xt[i0]) ** 2).sum(1); sel = torch.empty(N, dtype=torch.long, device=DEV); sel[0] = i0
        for t in range(1, N): i = torch.argmax(d2); sel[t] = i; d2 = torch.minimum(d2, ((Xt - Xt[i]) ** 2).sum(1))
        return sel.cpu().numpy()
    sel = [i0]; d2 = ((X - X[i0]) ** 2).sum(1)
    for _ in range(N - 1): i = int(np.argmax(d2)); sel.append(i); d2 = np.minimum(d2, ((X - X[i]) ** 2).sum(1))
    return np.array(sel)
WILD_FPS = WILD[fps(Z[WILD], min(len(WILD), 20000), seed=SEED)]
_sub = EN[np.random.default_rng(2).choice(len(EN), 2000, replace=False)].astype(np.float32)
_d2 = ((_sub[:1000, None, :] - _sub[None, 1000:, :]) ** 2).sum(-1); MMD_BW = float(np.sqrt(np.median(_d2) / 2)); del _d2
print(f"natural source cluster {len(WILD)} prompts (farthest-point order {len(WILD_FPS)}); sources: " + ", ".join(f"{k}:{len(v)}" for k, v in DIST.items()) + f" ({time.time()-t0:.0f}s)")


# %%
# ===== Cell 5: augmented real whitening, statistics, MMD, Fraudar =====
def make_bank_aug(Zref, h, seed):
    r = np.random.default_rng(seed); modes = r.standard_normal((M_MODES, Zref.shape[1])) / h / (2 * np.pi); m2 = 2 * M_MODES
    if USE_GPU:
        Mt = torch.as_tensor(modes, dtype=torch.float64, device=DEV); G = torch.zeros((m2, m2), dtype=torch.float64, device=DEV); mu = torch.zeros(m2, dtype=torch.float64, device=DEV)
        for s0 in range(0, len(Zref), 16384):
            th = (2 * math.pi) * (torch.as_tensor(Zref[s0:s0 + 16384], dtype=torch.float64, device=DEV) @ Mt.T)
            X = torch.cat([torch.cos(th), torch.sin(th)], 1); G += X.T @ X; mu += X.sum(0)
        G, mu = G.cpu().numpy(), mu.cpu().numpy()
    else:
        G = np.zeros((m2, m2)); mu = np.zeros(m2)
        for s0 in range(0, len(Zref), 4096):
            th = 2 * np.pi * (Zref[s0:s0 + 4096] @ modes.T); X = np.hstack([np.cos(th), np.sin(th)]); G += X.T @ X; mu += X.sum(0)
    G /= len(Zref); mu /= len(Zref); Sig = G - np.outer(mu, mu); ew, U = np.linalg.eigh((Sig + Sig.T) / 2); keep = ew > FLOOR * ew.max()
    B = dict(modes=modes, mu=mu, W=U[:, keep] / np.sqrt(ew[keep]), r=int(keep.sum()))
    if USE_GPU:
        B["MT"] = torch.as_tensor(modes, dtype=torch.float64, device=DEV); B["MU_T"] = torch.as_tensor(mu, dtype=torch.float64, device=DEV); B["W_T"] = torch.as_tensor(B["W"], dtype=torch.float64, device=DEV)
    return B
def whitened(pts, lab, cnt, B):
    if USE_GPU:
        Fc = torch.zeros((len(cnt), M_MODES), dtype=torch.float64, device=DEV); Fs = torch.zeros_like(Fc)
        for s0 in range(0, len(pts), CHUNK):
            P = torch.as_tensor(pts[s0:s0 + CHUNK], dtype=torch.float64, device=DEV); ix = torch.as_tensor(lab[s0:s0 + CHUNK], dtype=torch.long, device=DEV)
            th = (2.0 * math.pi) * (P @ B["MT"].T); Fc.index_add_(0, ix, torch.cos(th)); Fs.index_add_(0, ix, torch.sin(th))
        c = torch.as_tensor(cnt, dtype=torch.float64, device=DEV)
        return (((torch.cat([Fc, Fs], 1) - c[:, None] * B["MU_T"][None, :]) / torch.sqrt(c)[:, None]) @ B["W_T"]).cpu().numpy()
    import scipy.sparse as sp
    X = np.zeros((len(cnt), 2 * M_MODES))
    for s0 in range(0, len(pts), CHUNK):
        P, l_ = pts[s0:s0 + CHUNK], lab[s0:s0 + CHUNK]; th = 2 * np.pi * (P @ B["modes"].T)
        X += sp.csr_matrix((np.ones(len(P)), (l_, np.arange(len(P)))), shape=(len(cnt), len(P))) @ np.hstack([np.cos(th), np.sin(th)])
    return ((X - cnt[:, None] * B["mu"][None, :]) / np.sqrt(cnt)[:, None]) @ B["W"]
def union_S(Y, cnt, ix): return float(((Y[ix] * np.sqrt(cnt[ix])[:, None]).sum(0) ** 2).mean() / cnt[ix].sum())
def mmd2(Xw, refs):
    # Liu-Guo-Dong style: unbiased MMD^2 between window sample and each benign reference batch, averaged
    def k(A_, B_):
        if USE_GPU: return torch.exp(-torch.cdist(A_, B_) ** 2 / (2 * MMD_BW ** 2))
        d2 = (A_ ** 2).sum(1)[:, None] + (B_ ** 2).sum(1)[None, :] - 2 * A_ @ B_.T; return np.exp(-np.maximum(d2, 0) / (2 * MMD_BW ** 2))
    X = torch.as_tensor(Xw, dtype=torch.float32, device=DEV) if USE_GPU else Xw
    n = len(X); Kxx = k(X, X); sxx = (Kxx.sum() - Kxx.diagonal().sum()) / (n * (n - 1)); out = []
    for R in refs:
        m = len(R); Kyy = k(R, R); Kxy = k(X, R)
        out.append(float(sxx + (Kyy.sum() - Kyy.diagonal().sum()) / (m * (m - 1)) - 2 * Kxy.mean()))
    return float(np.mean(out))
def fraudar(lab, top, A_):
    C = np.zeros((A_, N_TOPICS)); np.add.at(C, (lab, top), 1.0); wt = 1.0 / np.log(C.sum(0) + 5.0); Wm = C * wt[None, :]
    da, dt = Wm.sum(1), Wm.sum(0); alive_a, alive_t = np.ones(A_, bool), np.ones(N_TOPICS, bool); tot = Wm.sum(); n = A_ + N_TOPICS; best = tot / n
    for _ in range(n - 1):
        ia = np.where(alive_a, da, np.inf).argmin(); it = np.where(alive_t, dt, np.inf).argmin()
        if da[ia] <= dt[it]:
            alive_a[ia] = False; tot -= da[ia]; dt -= Wm[ia] * alive_t; da[ia] = np.inf
        else:
            alive_t[it] = False; tot -= dt[it]; da -= Wm[:, it] * alive_a; dt[it] = np.inf
        n -= 1; best = max(best, tot / max(n, 1))
    return float(best)
def window_stats(idx_pts, lab, cnt, Bc, Bf, refs, extra=None, en_extra=None, top_extra=None):
    pts = Z[idx_pts] if extra is None else np.vstack([extra[0], Z[idx_pts]]); labs = lab if extra is None else np.concatenate([extra[1], lab])
    en = EN[idx_pts] if extra is None else np.vstack([en_extra, EN[idx_pts]]); top = TOPIC[idx_pts] if extra is None else np.concatenate([top_extra, TOPIC[idx_pts]])
    Yc = whitened(pts, labs, cnt, Bc); Yf = whitened(pts, labs, cnt, Bf); A_ = len(cnt)
    ev, V = np.linalg.eigh(Yc.T @ Yc / Bc["r"]); v = Yc @ V[:, -1]; v2 = v ** 2 / (v ** 2).sum()
    o = np.argsort(-v2); m = int(np.clip(np.searchsorted(np.cumsum(v2[o]), 0.5) + 1, 3, 100)); grp = o[:m]
    r2 = (Yc ** 2).mean(1); rr = np.random.default_rng(len(labs)); samp = rr.choice(len(en), min(MMD_WIN, len(en)), replace=False)
    return dict(spec=float(ev[-1]), r2=r2, acct=float(r2.max()), glob=union_S(Yc, cnt, np.arange(A_)),
                mmd=mmd2(en[samp].astype(np.float32), refs), fraudar=fraudar(labs, top, A_),
                vol=float(np.sort(cnt)[-10:].sum() / cnt.sum()), twoscale=float(np.log(union_S(Yc, cnt, grp)) - np.log(union_S(Yf, cnt, grp))), v2=v2, grp=grp)
def hc_stat(x, ns):
    M_ = len(ns); p = np.sort((1 + M_ - np.searchsorted(ns, x, side="left")) / (M_ + 1)); A_ = len(p); k = max(1, A_ // 2)
    i = np.arange(1, k + 1); pp = p[:k]; return float(np.max(np.sqrt(A_) * (i / A_ - pp) / np.sqrt(pp * (1 - pp))))


# %%
# ===== Cell 7: sources, query-cluster statistics, extended window statistics =====
FOLLOW = np.array([i for i in POOL_TXT if TURN[i] >= 2]) if len(POOL_TXT) else pool_idx
def _mm(key, ix): return DZ[key][ix], DIST[key][ix].astype(np.float32), km.predict(DZ[key][ix]).astype(np.int32)
def campaign_points(src, n, r):
    if src == "random": ix = r.choice(pool_idx, n, replace=False); return Z[ix], EN[ix].astype(np.float32), TOPIC[ix]
    if src == "wild_cluster": ix = r.choice(WILD, min(n, len(WILD)), replace=n > len(WILD)); return Z[ix], EN[ix].astype(np.float32), TOPIC[ix]
    if src in ("metamath_unique", "gsm8k_human", "metamath_template", "eval_gsm8k", "eval_mmlu"):
        D = DIST[src]; return _mm(src, r.choice(len(D), min(n, len(D)), replace=n > len(D)))
    if src == "metamath_camouflage": return _mm("metamath_unique", r.choice(CAMO, min(n, len(CAMO)), replace=n > len(CAMO)))
    if src == "metamath_rephrase":
        g = r.permutation(int(MM_GROUP.max()) + 1)[: int(math.ceil(n / 4))]; ix = np.flatnonzero(np.isin(MM_GROUP, g))[:n]; return _mm(src, ix)
    if src == "metamath_repeat":
        half = (n + 1) // 2; ix = r.choice(len(D_MMU), half, replace=False); ix = np.repeat(ix, 2)[:n]; return _mm("metamath_unique", ix)
    nm = (n + 1) // 2; ix = r.choice(len(D_MMU), nm, replace=False); other = FOLLOW if src == "metamath_followup" else pool_idx   # mixed / follow-up
    jx = r.choice(other, n - nm, replace=False); a, b, c = _mm("metamath_unique", ix)
    Zs, Es, Ts = np.empty((n, Z.shape[1]), np.float32), np.empty((n, EN.shape[1]), np.float32), np.empty(n, np.int32)
    Zs[0::2], Es[0::2], Ts[0::2] = a, b, c; Zs[1::2], Es[1::2], Ts[1::2] = Z[jx], EN[jx].astype(np.float32), TOPIC[jx]
    return Zs, Es, Ts                                                     # prompt and its padding/follow-up land in the same account

def cell_stats(pts, top, Bc, Bf):
    # two-scale score per topic cell, pooling queries across all accounts
    u, inv = np.unique(top, return_inverse=True); cnt = np.bincount(inv)
    yc = (whitened(pts, inv, cnt, Bc) ** 2).mean(1); yf = (whitened(pts, inv, cnt, Bf) ** 2).mean(1)
    ts = np.full(N_TOPICS, np.nan); n = np.zeros(N_TOPICS); n[u] = cnt; ok = cnt >= MINCELL
    ts[u[ok]] = np.log(yc[ok]) - np.log(yf[ok]); return ts, n / n.sum()
def window_stats15(idx_pts, lab, cnt, Bc, Bf, refs, extra=None, en_extra=None, top_extra=None):
    s = window_stats(idx_pts, lab, cnt, Bc, Bf, refs, extra=extra, en_extra=en_extra, top_extra=top_extra)
    pts = Z[idx_pts] if extra is None else np.vstack([extra[0], Z[idx_pts]]); labs = lab if extra is None else np.concatenate([extra[1], lab])
    en = EN[idx_pts] if extra is None else np.vstack([en_extra, EN[idx_pts]]); top = TOPIC[idx_pts] if extra is None else np.concatenate([top_extra, TOPIC[idx_pts]])
    s["cell_ts"], s["cell_share"] = cell_stats(pts, top, Bc, Bf)
    gm = np.isin(labs, s["grp"]); ge = en[gm]
    if len(ge) > MMD_WIN: ge = ge[np.random.default_rng(len(labs)).choice(len(ge), MMD_WIN, replace=False)]
    s["gmmd"] = mmd2(ge.astype(np.float32), refs) if len(ge) >= 20 else 0.0
    s["labs"] = labs; s["top"] = top; s["n_total"] = len(top); return s
def hc_stat(x, ns):
    M_ = len(ns); p = np.sort((1 + M_ - np.searchsorted(ns, x, side="left")) / (M_ + 1)); A_ = len(p); k = max(1, A_ // 2)
    i = np.arange(1, k + 1); pp = p[:k]; return float(np.max(np.sqrt(A_) * (i / A_ - pp) / np.sqrt(pp * (1 - pp))))
def calibrate15(recs):
    half = recs[0::2]; rest = recs[1::2] if len(recs) > 1 else recs
    TS = np.array([r["cell_ts"] for r in half]); SH = np.array([r["cell_share"] for r in half]); NT = np.array([r["n_total"] for r in half], float)
    LN = np.log(np.maximum(SH * NT[:, None], 1.0)); valid = np.mean(~np.isnan(TS), 0) >= 0.5
    A_ = np.zeros(N_TOPICS); B_ = np.zeros(N_TOPICS); SC = np.ones(N_TOPICS)
    for t in np.flatnonzero(valid):
        ok = ~np.isnan(TS[:, t]); x, y = LN[ok, t], TS[ok, t]
        b_, a_ = np.polyfit(x, y, 1) if np.ptp(x) > 0 else (0.0, float(np.median(y)))
        res = y - (a_ + b_ * x); A_[t], B_[t] = a_, b_; SC[t] = max(1.4826 * np.median(np.abs(res - np.median(res))), 0.05)
    med_sh = np.median(SH, 0); sc_sh = np.maximum(1.4826 * np.median(np.abs(SH - med_sh), 0), 0.002 + 0.1 * med_sh)
    T = dict(A=A_, B=B_, SC=SC, valid=valid, med_sh=med_sh, sc_sh=sc_sh)
    for r in recs: add_scores(r, T)
    q_ = lambda a: float(np.quantile(a, 1 - ALPHA))
    for d in DETS: T[d] = q_([r[d] for r in rest])
    return T
def add_scores(s, T):
    ln = np.log(np.maximum(s["cell_share"] * s["n_total"], 1.0))
    with np.errstate(all="ignore"):
        z = np.where(T["valid"] & ~np.isnan(s["cell_ts"]), (s["cell_ts"] - (T["A"] + T["B"] * ln)) / T["SC"], np.nan)
    s["qc"] = float(np.nanmax(z)) if np.any(~np.isnan(z)) else -1e9; s["qc_cell"] = int(np.nanargmax(z)) if np.any(~np.isnan(z)) else -1
    s["qcount"] = float(np.max((s["cell_share"] - T["med_sh"]) / T["sc_sh"])); return s
def fires15(s, T): return {d: s[d] > T[d] for d in DETS}
def boot_ci(x, B=1000, rng=np.random.default_rng(7)):
    x = np.array(x, float); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]; return float(x.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
def roc(null_vals, att_vals):
    null_vals, att_vals = np.array(null_vals), np.array(att_vals)
    auc = float(np.mean(att_vals[:, None] > null_vals[None, :]) + 0.5 * np.mean(att_vals[:, None] == null_vals[None, :]))
    return dict(auc=auc, tpr1=float(np.mean(att_vals > np.quantile(null_vals, 0.99))), tpr5=float(np.mean(att_vals > np.quantile(null_vals, 0.95))))


# %%
# ===== Cell 8: spectral machinery for R_c (Gram form), multi-outlier selection, diagnostics =====
from functools import lru_cache
@lru_cache(maxsize=64)
def mp_median(c):
    a, b = (1 - np.sqrt(c)) ** 2, (1 + np.sqrt(c)) ** 2; x = np.linspace(a + 1e-9, b - 1e-9, 20001)
    f = np.sqrt(np.maximum((b - x) * (x - a), 0)) / (2 * np.pi * c * x); F = np.cumsum(f) * (x[1] - x[0]); F /= F[-1]
    return float(x[np.searchsorted(F, 0.5)])
def top_group(v):
    v2 = v ** 2 / (v ** 2).sum(); o = np.argsort(-v2); m = int(np.clip(np.searchsorted(np.cumsum(v2[o]), 0.5) + 1, 3, 100)); return o[:m]
def ts_of(Yc, Yf, cnt, g): return float(np.log(union_S(Yc, cnt, g)) - np.log(union_S(Yf, cnt, g)))
def gram(Yn, kk=50):
    A_, r_ = Yn.shape
    if A_ >= r_:
        ev, U = np.linalg.eigh(Yn.T @ Yn); o = np.argsort(-ev); ev, U = ev[o], U[:, o]
        V = Yn @ U[:, :kk] / np.sqrt(np.maximum(ev[:kk], 1e-12)); return ev, V
    ev, V = np.linalg.eigh(Yn @ Yn.T); o = np.argsort(-ev); return ev[o], V[:, o][:, :kk]
def stats27(pts, labs, cnt, Bc, Bf, top=None, K_or=None, cells=False):
    Yc = whitened(pts, labs, cnt, Bc); Yf = whitened(pts, labs, cnt, Bf); A_, r_ = Yc.shape
    ev, Vc = np.linalg.eigh(Yc.T @ Yc / r_); gC = top_group(Yc @ Vc[:, -1])
    Ycc = Yc - Yc.mean(0, keepdims=True); Yn = Ycc / np.maximum(np.linalg.norm(Ycc, axis=1, keepdims=True), 1e-12)
    lam, V = gram(Yn); nz = lam[: min(A_, r_)]; g_ = A_ / r_
    e_hat = float(np.median(nz) / (max(g_, 1.0) * mp_median(min(g_, 1 / g_))) * (1 + np.sqrt(g_)) ** 2 - 1)
    J, kpre = [0], 0
    for j in range(min(20, V.shape[1])):
        if lam[j] - 1 > 1.05 * e_hat:
            kpre += 1
            if j > 0 and 1.0 / np.sum(V[:, j] ** 4) >= 5: J.append(j)
    gR = top_group(V[:, 0]); groups = [top_group(V[:, j]) for j in J]; sc = [ts_of(Yc, Yf, cnt, g) for g in groups]; jm = int(np.argmax(sc))
    s = dict(tsC=ts_of(Yc, Yf, cnt, gC), tsRc=sc[0], Tmax=float(max(sc)), k=len(J) - 1, kpre=kpre, lam1=float(lam[0] - 1), ehat=e_hat, gR=gR, gT=groups[jm])
    rr = np.random.default_rng(len(labs))
    def mcos(g): G_ = Yn[g] @ Yn[g].T; n_ = len(g); return float((G_.sum() - n_) / max(n_ * (n_ - 1), 1))
    s["cos_sel"] = mcos(gR); s["cos_rand"] = float(np.mean([mcos(rr.choice(A_, len(gR), replace=False)) for _ in range(5)]))
    if K_or:
        e = np.zeros(A_); e[:K_or] = 1 / np.sqrt(K_or); ov = (V.T @ e) ** 2; js = int(np.argmax(ov))
        s.update(jstar=js, ov=float(ov[js]), lam_js=float(lam[js] - 1), rho=mcos(np.arange(K_or)), orc=ts_of(Yc, Yf, cnt, np.arange(K_or)))
    if cells and top is not None:
        s["cell_ts"], s["cell_share"] = cell_stats(pts, top, Bc, Bf); s["n_total"] = len(top)
    return s
DETW = ["tsC", "tsRc", "Tmax"]
def acct_stats(pts, labs, cnt, Bc):
    Yc = whitened(pts, labs, cnt, Bc); return np.log(np.maximum((Yc ** 2).mean(1), 1e-9)), cnt
def stats32(pts, labs, cnt, Bc, Bf, top=None, K_or=None, weeks=None):
    # stats27 with optional per-source-week centring of the rows used for selection
    if weeks is None: s = stats27(pts, labs, cnt, Bc, Bf, top=top, K_or=K_or)
    else:
        global _WEEKS_CENTRE; _WEEKS_CENTRE = weeks; s = stats27_wc(pts, labs, cnt, Bc, Bf, K_or=K_or)
    return s
def stats27_wc(pts, labs, cnt, Bc, Bf, top=None, K_or=None, cells=False):
    Yc = whitened(pts, labs, cnt, Bc); Yf = whitened(pts, labs, cnt, Bf); A_, r_ = Yc.shape
    ev, Vc = np.linalg.eigh(Yc.T @ Yc / r_); gC = top_group(Yc @ Vc[:, -1])
    Ycc = Yc.copy()
    for w_ in np.unique(_WEEKS_CENTRE): m_ = _WEEKS_CENTRE == w_; Ycc[m_] -= Yc[m_].mean(0, keepdims=True); Yn = Ycc / np.maximum(np.linalg.norm(Ycc, axis=1, keepdims=True), 1e-12)
    lam, V = gram(Yn); nz = lam[: min(A_, r_)]; g_ = A_ / r_
    e_hat = float(np.median(nz) / (max(g_, 1.0) * mp_median(min(g_, 1 / g_))) * (1 + np.sqrt(g_)) ** 2 - 1)
    J, kpre = [0], 0
    for j in range(min(20, V.shape[1])):
        if lam[j] - 1 > 1.05 * e_hat:
            kpre += 1
            if j > 0 and 1.0 / np.sum(V[:, j] ** 4) >= 5: J.append(j)
    gR = top_group(V[:, 0]); groups = [top_group(V[:, j]) for j in J]; sc = [ts_of(Yc, Yf, cnt, g) for g in groups]; jm = int(np.argmax(sc))
    s = dict(tsC=ts_of(Yc, Yf, cnt, gC), tsRc=sc[0], Tmax=float(max(sc)), k=len(J) - 1, kpre=kpre, lam1=float(lam[0] - 1), ehat=e_hat, gR=gR, gT=groups[jm])
    rr = np.random.default_rng(len(labs))
    def mcos(g): G_ = Yn[g] @ Yn[g].T; n_ = len(g); return float((G_.sum() - n_) / max(n_ * (n_ - 1), 1))
    s["cos_sel"] = mcos(gR); s["cos_rand"] = float(np.mean([mcos(rr.choice(A_, len(gR), replace=False)) for _ in range(5)]))
    if K_or:
        e = np.zeros(A_); e[:K_or] = 1 / np.sqrt(K_or); ov = (V.T @ e) ** 2; js = int(np.argmax(ov))
        s.update(jstar=js, ov=float(ov[js]), lam_js=float(lam[js] - 1), rho=mcos(np.arange(K_or)), orc=ts_of(Yc, Yf, cnt, np.arange(K_or)))
    if cells and top is not None:
        s["cell_ts"], s["cell_share"] = cell_stats(pts, top, Bc, Bf); s["n_total"] = len(top)
    return s

def groups_of(pts, labs, cnt, Bc, weeks=None):
    Yc = whitened(pts, labs, cnt, Bc); A_, r_ = Yc.shape
    Ycc = Yc - Yc.mean(0, keepdims=True)
    if weeks is not None:
        Ycc = Yc.copy()
        for w_ in np.unique(weeks): m_ = weeks == w_; Ycc[m_] -= Yc[m_].mean(0, keepdims=True)
    Yn = Ycc / np.maximum(np.linalg.norm(Ycc, axis=1, keepdims=True), 1e-12); lam, V = gram(Yn); g_ = A_ / r_
    e_hat = float(np.median(lam[: min(A_, r_)]) / (max(g_, 1.0) * mp_median(min(g_, 1 / g_))) * (1 + np.sqrt(g_)) ** 2 - 1)
    return [top_group(V[:, j]) for j in range(min(20, V.shape[1])) if lam[j] - 1 > 1.05 * e_hat and (j == 0 or 1.0 / np.sum(V[:, j] ** 4) >= 5)]


# %%
# ===== Cell 9b: campaigns for this step =====
ONV_ROWS = np.intersect1d(pool_idx, ONV_KEYS) if len(np.intersect1d(pool_idx, ONV_KEYS)) >= 50 else pool_idx[np.argsort(-POSTV[pool_idx])[:max(50, len(pool_idx)//50)]]; OFFV_ROWS = np.setdiff1d(pool_idx, ONV_ROWS)
_cp_prev2 = campaign_points
def campaign_points(src, n, r):
    if src in ("voiced_c05", "voiced_c1", "tilted_v_c05", "matched_c05"):
        n_on = n if src == "voiced_c1" else (n + 1) // 2
        if src.startswith("voiced") or src == "matched_c05":
            key = "matched" if src == "matched_c05" else "voiced"
            k = r.choice(len(DZ[key]), n_on, replace=n_on > len(DZ[key])); Zon, Eon = DZ[key][k], DIST[key][k].astype(np.float32)
        else:
            k = r.choice(ONV_ROWS, n_on, replace=n_on > len(ONV_ROWS)); Zon, Eon = Z[k], EN[k].astype(np.float32)
        Ton = km.predict(Zon).astype(np.int32)
        if src == "voiced_c1": return Zon, Eon, Ton
        jx = r.choice(OFFV_ROWS, n - n_on, replace=False); Zs = np.empty((n, Z.shape[1]), np.float32); Es = np.empty((n, EN.shape[1]), np.float32); Ts = np.empty(n, np.int32)
        Zs[0::2], Es[0::2], Ts[0::2] = Zon[: len(Zs[0::2])], Eon[: len(Zs[0::2])], Ton[: len(Zs[0::2])]; Zs[1::2], Es[1::2], Ts[1::2] = Z[jx], EN[jx].astype(np.float32), TOPIC[jx]
        return Zs, Es, Ts
    return _cp_prev2(src, n, r)
# personas (Corollary 1): attacker-pool account-weeks with >= MIN_PER_ACCT queries and a majority on task; their on-task queries are replaced by written prompts
pk = np.lexsort((ACCT[pool_idx], WEEK[pool_idx])); prow = pool_idx[pk]; kk_ = WEEK[prow].astype(np.int64) * n_acct + ACCT[prow]
_groups = [g[:VOL_CAP] for g in np.split(prow, np.flatnonzero(np.diff(kk_)) + 1) if len(g) >= MIN_PER_ACCT]
PERSONAS = [g for g in _groups if SV[g].mean() >= 0.5]
if len(PERSONAS) < 20:
    PERSONAS = sorted(_groups, key=lambda g: -SV[g].mean())[: max(20, len(_groups) // 50)]; print(f"[persona] WARNING: few majority-on-task pool accounts; using the {len(PERSONAS)} most on-task")
print(f"[persona] {len(PERSONAS)} persona account-weeks; mean on-task share {np.mean([SV[g].mean() for g in PERSONAS]):.2f}")
def persona_points(K_, r):
    pick = r.choice(len(PERSONAS), K_, replace=len(PERSONAS) < K_); Zs, Es, lab, flags = [], [], [], []
    for a_, pi in enumerate(pick):
        g = PERSONAS[pi]; on = SV[g].copy(); z = Z[g].copy(); e = EN[g].astype(np.float32).copy()
        k = r.choice(len(DZ["voiced"]), int(on.sum()), replace=int(on.sum()) > len(DZ["voiced"])); z[on] = DZ["voiced"][k]; e[on] = DIST["voiced"][k].astype(np.float32)
        Zs.append(z); Es.append(e); lab.append(np.full(len(g), a_)); flags.append(on)
    Zc = np.vstack(Zs); return Zc, np.vstack(Es), km.predict(Zc).astype(np.int32), np.concatenate(lab), np.concatenate(flags)
def type_share(labs, flags):
    cnt_ = np.bincount(labs); sh = np.bincount(labs, weights=flags.astype(float)) / np.maximum(cnt_, 1); return float(np.mean(sh[cnt_ >= MIN_PER_ACCT] >= 0.5))


# %% [markdown]
# ## Weekly detection and per-account topicality

# %%
# ===== Cell W: weekly populations — conformal thresholds, campaigns, per-account topicality =====
t0 = time.time(); blk = WEEK.astype(np.int32); n_b = int(blk.max()) + 1
order = np.lexsort((ACCT, blk)); keys = blk[order].astype(np.int64) * n_acct + ACCT[order]; GA = {b: [] for b in range(n_b)}
for g in np.split(order, np.flatnonzero(np.diff(keys)) + 1):
    if not is_pool[int(ACCT[g[0]])] and len(g) >= MIN_PER_ACCT: GA[int(blk[g[0]])].append(g[:VOL_CAP])
ok = [b for b in range(WK["CAL_N"], n_b) if len(GA[b]) >= 40 and all(len(GA[v]) >= 20 for v in range(b - WK["CAL_N"], b))]
TB = [int(ok[i]) for i in np.linspace(0, len(ok) - 1, min(WK["N_TEST"], len(ok))).astype(int)]
def sub_window(b, r):
    k = max(20, int(SUB_FRAC * len(GA[b]))); pick = r.choice(len(GA[b]), k, replace=False); parts = [GA[b][i] for i in pick]
    vols = np.array([len(p) for p in parts]); return np.concatenate(parts), np.repeat(np.arange(len(parts)), vols), vols
WEEKS = [v for v in range(n_b) if len(GA[v]) >= 20]; BANK, NULLW = {}, {}
for b in WEEKS:
    prev = [np.concatenate(GA[v]) for v in range(max(0, b - WK["LAGB"]), b) if GA[v]]
    ref = np.concatenate(prev) if prev else np.concatenate(GA[b]); ref = np.random.default_rng(b).choice(ref, min(100000, len(ref)), replace=False)
    BANK[b] = (make_bank_aug(Z[ref], H_COARSE, SEED + 7), make_bank_aug(Z[ref], H_FINE, SEED + 8)); NULLW[b] = []
    for t in range(WK["NULL_PER"]):
        idx, lab, vols = sub_window(b, np.random.default_rng(SEED + 100 * b + t)); s = stats27(Z[idx], lab, np.bincount(lab), *BANK[b])
        s["accS"], s["accN"] = acct_stats(Z[idx], lab, np.bincount(lab), BANK[b][0]); s["typecount"] = type_share(lab, SV[idx]); NULLW[b].append(s)
print(f"[weekly] banks and null windows for {len(WEEKS)} weeks ({time.time()-t0:.0f}s)")
NBINS = [10, 20, 50, 10 ** 9]
def topic_hc(s, ref):
    ps = []
    for lo, hi in zip(NBINS[:-1], NBINS[1:]):
        m = (s["accN"] >= lo) & (s["accN"] < hi); R = ref[(lo, hi)]
        if m.sum() == 0 or len(R) < 20: continue
        u = (np.searchsorted(R, s["accS"][m]) + 0.5) / (len(R) + 1); ps.append(np.minimum(u, 1 - u) * 2)
    if not ps: return 0.0
    p = np.sort(np.concatenate(ps)); n = len(p); i = np.arange(1, n + 1); k = max(1, n // 2)
    return float(np.max(np.sqrt(n) * (i[:k] / n - p[:k]) / np.sqrt(np.maximum(p[:k] * (1 - p[:k]), 1e-12))))
REFT, TH = {}, {}
for b in TB:
    prior = [r for v in WEEKS if v < b for r in NULLW[v]]
    refw = [r for v in WEEKS if v < b - WK["CAL_N"] for r in NULLW[v]] or prior; calw = [r for v in WEEKS if b - WK["CAL_N"] <= v < b for r in NULLW[v]]
    REFT[b] = {(lo, hi): np.sort(np.concatenate([r["accS"][(r["accN"] >= lo) & (r["accN"] < hi)] for r in refw])) for lo, hi in zip(NBINS[:-1], NBINS[1:])}
    for r in NULLW[b]: r["topic"] = topic_hc(r, REFT[b])
    pr_t = [topic_hc(r, REFT[b]) for r in calw]                              # out of sample: reference from earlier weeks, threshold from the 8 weeks before b
    TH[b] = {d: float(np.quantile([r[d] for r in prior], 1 - ALPHA)) for d in DETW + ["typecount"]}; TH[b]["topic"] = float(np.quantile(pr_t, 1 - ALPHA))
DETS = DETW + ["topic", "typecount"]
# [PAPER] Cor. 1 / App. B / App. P (P32m): q = 0.25% (paper derives about 14 posing accounts at A = 10^5 from it)
Q_TYPE = float(np.mean([SV[g].mean() >= 0.5 for b in WEEKS for g in GA[b]]))
print(f"[types] benign share of majority-on-task accounts q = {Q_TYPE:.4f}")
# [PAPER] App. P (Step 32, P32f): per-account topicality HC at 25% false alarms (FPR["topic"])
FPR = {d: boot_ci([np.mean([r[d] > TH[b][d] for r in NULLW[b]]) for b in TB]) for d in DETS}
print("[weekly] held-out FPR: " + " ".join(f"{d}={FPR[d][0]:.4f}" for d in DETS))
RES = []
for src in ["metamath_unique", "metamath_mixed", "tilted_v_c05", "voiced_c05", "voiced_c1", "matched_c05", "persona", "random"]:
    pw = {d: [] for d in DETS}; nv, av = {d: [] for d in DETS}, {d: [] for d in DETS}
    for b in TB:
        acc = {d: [] for d in DETS}
        for rep in range(S27["REPS"]):
            r = np.random.default_rng(SEED + 9_000_000 + 1000 * b + 10 * rep); idx, lab, vols = sub_window(b, r); A_ = int(lab.max()) + 1
            if S27["K"] >= A_ // 2: continue
            rc = np.random.default_rng(SEED + 17 * b + rep)
            if src == "persona": Zc, Ec, Tc, labc, fc = persona_points(S27["K"], rc); n = len(Zc)
            else:
                cv = r.choice(np.array([len(g) for g in GA[b]]), S27["K"], replace=True)
                Zc, Ec, Tc = campaign_points(src, int(cv.sum()), rc); n = len(Zc); labc = np.repeat(np.arange(S27["K"]), cv)[:n]; fc = clf_v.predict_proba(Ec)[:, 1] >= 0.5
            keep = lab >= S27["K"]
            _, inv = np.unique(np.concatenate([labc, lab[keep]]), return_inverse=True); P_ = np.vstack([Zc, Z[idx[keep]]]); cn = np.bincount(inv)
            sa = stats27(P_, inv, cn, *BANK[b]); sa["accS"], sa["accN"] = acct_stats(P_, inv, cn, BANK[b][0]); sa["topic"] = topic_hc(sa, REFT[b])
            sa["typecount"] = type_share(inv, np.concatenate([fc, SV[idx[keep]]])); sb = NULLW[b][rep]
            for d in DETS: acc[d].append(float(sa[d] > TH[b][d] and not sb[d] > TH[b][d])); nv[d].append(sb[d]); av[d].append(sa[d])
        for d in DETS:
            if acc[d]: pw[d].append(np.mean(acc[d]))
    row = dict(source=src, **{d: boot_ci(pw[d]) for d in DETS}, **{d + "_roc": roc(nv[d], av[d]) for d in DETS}); RES.append(row)
    # [PAPER] §6 / App. G.2: matched_c05 evaded every detector; App. H / App. P (P32l): persona 0.00, type count 0.58; App. P (P32c, first run): voiced 0.00; 0.10; Fig. 1a x-axis = tsRc of voiced_c05 / matched_c05
    print(f"[W] {src:>14} | deployed " + " ".join(f"{d}={row[d][0]:.2f}" for d in DETS) + " | oracle " + " ".join(f"{d}={row[d+'_roc']['tpr1']:.2f}" for d in DETS) + f"  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Pooling artefact and deployable calibration at scale

# %%
# ===== Cell P: pooling artefact, per-source-week centring, deployable multi-outlier selection at scale =====
t0 = time.time()
order5 = np.lexsort((ACCT, WEEK)); keys5 = WEEK[order5].astype(np.int64) * n_acct + ACCT[order5]
POOLW5 = [(int(WEEK[g[0]]), g[:VOL_CAP]) for g in np.split(order5, np.flatnonzero(np.diff(keys5)) + 1) if not is_pool[int(ACCT[g[0]])] and len(g) >= 5]
refL = np.random.default_rng(12).choice(np.concatenate([g for _, g in POOLW5]), 100000 if not QUICK else 5000, replace=False)
BL = (make_bank_aug(Z[refL], H_COARSE, SEED + 7), make_bank_aug(Z[refL], H_FINE, SEED + 8))
def pwin(A_, r, pool):
    pick = r.choice(len(pool), min(A_, len(pool)), replace=False); parts = [pool[i][1] for i in pick]; wk = np.array([pool[i][0] for i in pick])
    vols = np.array([len(p) for p in parts]); return np.concatenate(parts), np.repeat(np.arange(len(parts)), vols), vols, wk
def pattack(A_, src, r, pool, centre_week):
    idx, lab, vols, wk = pwin(A_, r, pool); K_ = S27["KP"]; cv = r.choice(vols, K_, replace=True).astype(float); cv = np.maximum(1, np.round(cv * S27["NP"] / cv.sum())).astype(int)
    Zc, _, _ = campaign_points(src, int(cv.sum()), r); n = len(Zc); labc = np.repeat(np.arange(K_), cv)[:n]; keep = lab >= K_
    _, inv = np.unique(np.concatenate([labc, lab[keep]]), return_inverse=True); wk_all = np.r_[np.full(K_, int(np.median(wk))), wk[K_:]]
    return stats32(np.vstack([Zc, Z[idx[keep]]]), inv, np.bincount(inv), *BL, K_or=K_, weeks=wk_all if centre_week else None)
def seeded(pts, labs, cnt, seeds, colluders=None):
    Yc = whitened(pts, labs, cnt, BL[0]); Yf = whitened(pts, labs, cnt, BL[1]); Ycc = Yc - Yc.mean(0, keepdims=True)
    Yn = Ycc / np.maximum(np.linalg.norm(Ycc, axis=1, keepdims=True), 1e-12); cen = Yn[seeds].mean(0); corr = Yn @ cen; corr[seeds] = -np.inf
    G = np.r_[seeds, np.argsort(-corr)[: S32["EXPAND"]]]
    out = dict(seeded=ts_of(Yc, Yf, cnt, G), meta_only=ts_of(Yc, Yf, cnt, np.asarray(seeds)))
    if colluders is not None: out["recall"] = float(np.isin(colluders, G).mean())
    return out
def pattack_seeded(A_, src, r):
    idx, lab, vols, wk = pwin(A_, r, POOLW5); K_ = S27["KP"]; cv = r.choice(vols, K_, replace=True).astype(float); cv = np.maximum(1, np.round(cv * S27["NP"] / cv.sum())).astype(int)
    Zc, _, _ = campaign_points(src, int(cv.sum()), r); n = len(Zc); labc = np.repeat(np.arange(K_), cv)[:n]; keep = lab >= K_
    _, inv = np.unique(np.concatenate([labc, lab[keep]]), return_inverse=True)
    return seeded(np.vstack([Zc, Z[idx[keep]]]), inv, np.bincount(inv), r.choice(K_, S32["LINKED"], replace=False), colluders=np.arange(K_))
POOLRES = []
for A_ in S32["A_P"]:
    if A_ > 0.9 * len(POOLW5): continue
    same, expct = [], []
    for t in range(S32["NUL_P"] // 2):
        idx, lab, vols, wk = pwin(A_, np.random.default_rng(SEED + 51 * A_ + t), POOLW5)
        for g in groups_of(Z[idx], lab, vols, BL[0]):
            if len(g) < 2: continue
            # [PAPER] App. P (Step 32, P32g): same-week share 1.9–2.1× the random expectation
            w_ = wk[g]; same.append(np.mean(w_[:, None] == w_[None, :])); _, c_ = np.unique(wk, return_counts=True); expct.append(np.sum((c_ / c_.sum()) ** 2))
    for centre in [False, True]:
        nul = [stats32(Z[i], l, v, *BL, weeks=w if centre else None) for i, l, v, w in (pwin(A_, np.random.default_rng(SEED + 31 * A_ + t), POOLW5) for t in range(S32["NUL_P"]))]
        if not centre:
            nul_seed = [seeded(Z[i], l, v, np.random.default_rng(SEED + 5 * t).choice(len(v), S32["LINKED"], replace=False)) for i, l, v, w in (pwin(A_, np.random.default_rng(SEED + 31 * A_ + t), POOLW5) for t in range(S32["NUL_P"]))]
        for src in ["metamath_unique", "metamath_mixed"]:
            att = [pattack(A_, src, np.random.default_rng(SEED + 37 * A_ + t + len(src)), POOLW5, centre) for t in range(S32["ATT_P"])]
            row = dict(A=A_, centre_week=centre, source=src, same_week=float(np.mean(same)) if same else float("nan"), expected=float(np.mean(expct)) if expct else float("nan"))
            # [PAPER] App. P (P32h): per-week-centred Tmax 1.00 at 10^4 (centre=True); centre=False rows plausibly replicate App. Table 11 Tmax (primary source Step 27–31)
            for d in DETW: row[d] = float(np.mean([x[d] > np.quantile([y[d] for y in nul], 0.99) for x in att]))
            if not centre:
                sa_ = [pattack_seeded(A_, src, np.random.default_rng(SEED + 37 * A_ + t + len(src))) for t in range(S32["ATT_P"])]
                for key in ["seeded", "meta_only"]:
                    th_ = np.quantile([y[key] for y in nul_seed], 0.99); row["link_" + key] = float(np.mean([x[key] > th_ for x in sa_]))
                # [PAPER] §5 / App. D.5 / App. P (P32o): seeded expansion 1.00, recall 0.96 ("96–100%"), metadata alone 0.67
                row["link_recall"] = float(np.mean([x["recall"] for x in sa_]))
            POOLRES.append(row)
            print(f"[P] A={A_} centre_week={centre} {src:>16}: oracle TPR C {row['tsC']:.2f} Rc {row['tsRc']:.2f} Tmax {row['Tmax']:.2f} | same-week share {row['same_week']:.3f} vs random {row['expected']:.3f}" + (f" | {S32['LINKED']} linked colluders: metadata only {row['link_meta_only']:.2f}, seeded hollowed expansion {row['link_seeded']:.2f} (colluder recall {row['link_recall']:.2f})" if not centre else "") + f"  ({time.time()-t0:.0f}s)")
cut = 36 if n_b >= 50 else int(0.6 * n_b); calp = [x for x in POOLW5 if 8 <= x[0] < cut]; testp = [x for x in POOLW5 if x[0] >= cut]; DEPS = []
for A_req in S32["A_DEPS"]:
    A_ = min(A_req, int(0.8 * len(testp)), int(0.8 * len(calp)))
    if A_ < 3 * S27["KP"]: continue
    cal = [stats32(Z[i], l, v, *BL, weeks=w) for i, l, v, w in (pwin(A_, np.random.default_rng(SEED + 61 + t), calp) for t in range(S32["CAL_DEP"]))]
    thr = {d: float(np.quantile([x[d] for x in cal], 1 - ALPHA)) for d in DETW}
    tnul = [stats32(Z[i], l, v, *BL, weeks=w) for i, l, v, w in (pwin(A_, np.random.default_rng(SEED + 71 + t), testp) for t in range(S32["TEST_DEP"]))]
    fl = [x for x in tnul if x["Tmax"] > thr["Tmax"]]; gsz = float(np.mean([len(x["gT"]) for x in tnul]))
    # [PAPER] App. P (P32i, P32n): deployed Tmax false alarms 30% at 2,500, 5% at 7,552; §5 / App. D.5: "5% false alarms, not 1%"
    DEP = dict(A=A_, fpr={d: float(np.mean([x[d] > thr[d] for x in tnul])) for d in DETW}, null_group_size=gsz)
    # [PAPER] App. P (Step 32, P32p): 44–59 benign accounts flagged per week at 10^5 (A ≥ 5,000)
    DEP["benign_flagged_per_week_at_1e5"] = float(1e5 / A_ * DEP["fpr"]["Tmax"] * gsz)
    for src in ["metamath_unique", "metamath_mixed"]:
        att = [pattack(A_, src, np.random.default_rng(SEED + 81 + t + len(src)), testp, True) for t in range(S32["ATT_DEP"])]
        # [PAPER] App. D.5: deployed power unpadded 0.53–1.00, padded 0.03–0.10 at 5,000–7,552 account-weeks
        DEP[src] = {d: float(np.mean([x[d] > thr[d] for x in att])) for d in DETW}
        hit = [x for x in att if x["Tmax"] > thr["Tmax"]]
        # [PAPER] App. K: deployed precision 0.73 and 0.54 at 5,000–7,552 account-weeks
        DEP[src]["precision"] = float(np.mean([np.mean(x["gT"] < S27["KP"]) for x in hit])) if hit else float("nan")
        DEP[src]["recall"] = float(np.mean([np.sum(x["gT"] < S27["KP"]) / S27["KP"] for x in hit])) if hit else float("nan")
    DEPS.append(DEP)
    print(f"[P-dep] A={A_}: Tmax FPR {DEP['fpr']['Tmax']:.3f} | unique {DEP['metamath_unique']['Tmax']:.2f} (precision {DEP['metamath_unique']['precision']:.2f}, recall {DEP['metamath_unique']['recall']:.2f}) | mixed {DEP['metamath_mixed']['Tmax']:.2f} | benign accounts flagged per week at 1e5: {DEP['benign_flagged_per_week_at_1e5']:.1f}  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Distillation

# %%
# ===== Cell 5: LoRA fine-tuning and evaluation (merged fp16 weights for fast generation) =====
import sys, subprocess, importlib
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
for m in [m for m in list(sys.modules) if m in ("peft", "torchao") or m.startswith(("peft.", "torchao."))]: del sys.modules[m]
importlib.invalidate_caches()
try:
    import peft
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"]); importlib.invalidate_caches(); import peft
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from datasets import load_dataset
fmtq = lambda q: f"Question: {q.strip()}\nAnswer:"
def train_lora(student, pairs, seed, tok):
    torch.manual_seed(seed); model = AutoModelForCausalLM.from_pretrained(student, torch_dtype=torch.float32).to(DEV)
    model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4); steps = max(1, math.ceil(len(pairs) / BS))
    sch = get_linear_schedule_with_warmup(opt, max(1, steps // 20), steps); scaler = torch.cuda.amp.GradScaler()
    tok.padding_side = "right"; model.train(); order = np.random.default_rng(seed).permutation(len(pairs))
    for s0 in range(0, len(pairs), BS):
        batch = [pairs[k] for k in order[s0:s0 + BS]]
        enc = tok([fmtq(q) + " " + a.strip() + tok.eos_token for q, a in batch], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
        labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
        for k, (q, _) in enumerate(batch): labels[k, : len(tok(fmtq(q))["input_ids"])] = -100
        with torch.autocast("cuda", dtype=torch.float16): loss = model(**enc, labels=labels).loss
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
    model = model.merge_and_unload().half(); model.config.use_cache = True; return model
@torch.no_grad()
def generate(model, tok, questions, max_new):
    model.eval(); tok.padding_side = "left"; outs = []
    for i in range(0, len(questions), GEN_BS):
        enc = tok([fmtq(q) for q in questions[i:i + GEN_BS]], return_tensors="pt", padding=True).to(DEV)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).split("\nQuestion:")[0] for o in out]
    return outs
def last_number(t):
    m = re.search(r"answer is:?\s*\$?(-?[\d,]*\.?\d+)", t); s = m.group(1) if m else (re.findall(r"-?[\d,]*\.?\d+", t) or [None])[-1]
    try: return float(s.replace(",", "")) if s else None
    except ValueError: return None
def norm_math(a):
    if a is None: return None
    a = a.strip().strip("$").replace("\\left", "").replace("\\right", "").replace(" ", "").replace("\\!", "").replace("dfrac", "frac").replace("tfrac", "frac")
    a = re.sub(r"\\text\{(.*?)\}", r"\1", a); a = a.rstrip(".")
    try: return str(float(a)).rstrip("0").rstrip(".") if re.fullmatch(r"-?[\d.]+", a) else a
    except ValueError: return a
def extract_math(t):
    m = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", t)
    if m: return m[-1]
    m = re.search(r"answer is:?\s*(.+)", t); return m.group(1).split("\n")[0] if m else None

# ===== Cell 6: full fine-tuning for the small student =====
MAX_LEN, BS, GEN_BS = S32['MAX_LEN'], S32['BS'], S32['GEN_BS_EVAL']
def train_full(student, pairs, seed, tok, lr):
    torch.manual_seed(seed); model = AutoModelForCausalLM.from_pretrained(student, torch_dtype=torch.float32).to(DEV)
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0); steps = max(1, math.ceil(len(pairs) / BS))
    steps = max(1, math.ceil(len(pairs) / BS)); sch = get_linear_schedule_with_warmup(opt, max(1, steps // 30), steps); scaler = torch.cuda.amp.GradScaler()
    tok.padding_side = "right"; model.train(); order = np.random.default_rng(seed).permutation(len(pairs))
    MICRO, ACC = 4, max(1, BS // 4); starts = list(range(0, len(pairs), MICRO))
    for i, s0 in enumerate(starts):
        batch = [pairs[k] for k in order[s0:s0 + MICRO]]
        enc = tok([fmtq(q) + " " + a.strip() + tok.eos_token for q, a in batch], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
        labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
        for k, (q, _) in enumerate(batch): labels[k, : len(tok(fmtq(q))["input_ids"])] = -100
        with torch.autocast("cuda", dtype=torch.float16): loss = model(**enc, labels=labels).loss / ACC
        scaler.scale(loss).backward(); del enc, labels, loss
        if (i + 1) % ACC == 0 or i == len(starts) - 1:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step()
    model = model.half(); model.config.use_cache = True; return model

# ===== Cell D: distillation of the written and value-tilted campaigns =====
for _n in ["BANK", "NULLW", "BL"]:
    if _n in globals(): del globals()[_n]
gc.collect(); torch.cuda.empty_cache()
tokA = AutoTokenizer.from_pretrained(STUDENT); tokA.pad_token = tokA.pad_token or tokA.eos_token
gsm = load_dataset("openai/gsm8k", "main", split="test").select(range(S32["N_EVAL"])); GQ = list(gsm["question"]); GOLD = [float(a.split("####")[-1].strip().replace(",", "")) for a in gsm["answer"]]
def gsm_correct(model):
    preds = [last_number(t) for t in generate(model, tokA, GQ, S32["MAX_NEW"])]
    return np.array([p is not None and abs(p - g) < 1e-4 for p, g in zip(preds, GOLD)], float)
# [PAPER] App. G.2: the student solves 14 of 400 problems at baseline
bm = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.float16).to(DEV); BASE = gsm_correct(bm); del bm; torch.cuda.empty_cache()
OFF_PAIRS = [POOL_TXT[k] for k in OFFV_KEYS]; ONV_PAIRS = [POOL_TXT[k] for k in ONV_KEYS]
def pairs_for(var, n, rng):
    if var == "voiced_c1": k = min(n, len(VOICED)); return [VOICED[i] for i in rng.choice(len(VOICED), k, replace=False)]
    if var == "persona":
        out = []
        for pi in rng.permutation(len(PERSONAS)):
            for row in PERSONAS[pi]:
                if row in POOL_TXT and POOL_TXT[row][0] and POOL_TXT[row][1]: out.append(VOICED[rng.integers(len(VOICED))] if SV[row] else POOL_TXT[row])
            if len(out) >= n: return out[:n]
        return out
    on = {"voiced_c05": VOICED, "matched_c05": MATCHED}.get(var, ONV_PAIRS); k = min((n + 1) // 2, len(on))
    return [on[i] for i in rng.choice(len(on), k, replace=False)] + [OFF_PAIRS[i] for i in rng.choice(len(OFF_PAIRS), n - k, replace=False)]
DRES = {}; t0 = time.time()
print(f"[D] base GSM8K {BASE.mean():.3f}; written pairs {len(VOICED)}, value-region pool pairs {len(ONV_PAIRS)}")
for var in ["voiced_c05", "voiced_c1", "matched_c05", "persona", "tilted_v_c05"]:
    for sd in S32["SEEDS"]:
        pairs = pairs_for(var, S32["BUD"], np.random.default_rng(SEED + 1000 * sd)); model = train_full(STUDENT, pairs, sd, tokA, 5e-5)
        # [PAPER] §6 / App. G.2: matched_c05 +4.4 (42% of +10.4); App. H / App. P: tilted_v_c05 +1.1, persona +3.5 from 1,891 queries; App. P (P32d, first run): voiced_c05 +2.2
        DRES[(var, sd)] = gsm_correct(model).mean(); del model; gc.collect(); torch.cuda.empty_cache()
        print(f"[D] {var:>13} seed={sd} ({len(pairs)} pairs): GSM8K {DRES[(var, sd)]:.3f} (gain {100*(DRES[(var, sd)]-BASE.mean()):+.1f})  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Predictions and VERDICT REPORT — copy into the chat

# %%
# ===== Cell Z: predictions and verdict =====
g = lambda src, d: next(r[d][0] for r in RES if r["source"] == src)
gain = lambda v: float(np.mean([100 * (DRES[(v, s)] - BASE.mean()) for s in S32["SEEDS"] if (v, s) in DRES]))
# [PAPER] App. P (Step 32 rows): outcomes of P32a–p as printed below (P32a–i quote the first run; this rerun replicates them)
print(f"P32a p_v = {P_V:.4f} (pass if <= 0.05)")
print(f"P32b discriminator AUC {VOI_AUC:.3f} (<= 0.75), kept {VOI_KEEP:.3f} (>= 0.5), filter relaxed: {VOI_RELAXED}")
print(f"P32c voiced_c05 weekly: two-scale {g('voiced_c05','tsC'):.2f} (<= 0.10), hollowed {g('voiced_c05','tsRc'):.2f}, Tmax {g('voiced_c05','Tmax'):.2f} (<= 0.25); voiced_c1: hollowed {g('voiced_c1','tsRc'):.2f}")
print(f"P32d voiced_c05 gain at {S32['BUD']}: {gain('voiced_c05'):+.1f} (pass if >= 6; all-math 10.4, padding 7.6, tilted 2.2); voiced_c1 {gain('voiced_c1'):+.1f}")
print(f"P32e tilted_v_c05 gain {gain('tilted_v_c05'):+.1f} (pass if >= 4)")
print(f"P32f topicality HC: tilted_v_c05 {g('tilted_v_c05','topic'):.2f}, voiced_c05 {g('voiced_c05','topic'):.2f}, mixed {g('metamath_mixed','topic'):.2f}; FPR {FPR['topic'][0]:.4f} (pass if tilted >= 0.3 at FPR <= 0.02)")
for r in POOLRES:
    if not r["centre_week"] and r["source"] == "metamath_unique": print(f"P32g A={r['A']}: same-week share {r['same_week']:.3f} vs random {r['expected']:.3f} (pass if >= 2x)")
for r in POOLRES:
    if r["centre_week"]: print(f"P32h A={r['A']} {r['source']}: per-week-centred Tmax {r['Tmax']:.2f} (pass if unique >= 0.8 at the largest A)")
for D_ in DEPS: print(f"P32i/n deployable A={D_['A']}: Tmax FPR {D_['fpr']['Tmax']:.3f} (<= 0.02), unique {D_['metamath_unique']['Tmax']:.2f} (>= 0.5; precision {D_['metamath_unique']['precision']:.2f}), mixed {D_['metamath_mixed']['Tmax']:.2f}; P32p benign accounts flagged per week at 1e5: {D_['benign_flagged_per_week_at_1e5']:.1f} (<= 100)")
print(f"P32j discriminator AUC by round {[round(x, 3) for x in VOI_AUCS]} (pass if final <= 0.65)")
print(f"P32k matched (importance-resampled, ESS {ESS:.0f}): hollowed {g('matched_c05','tsRc'):.2f}, Tmax {g('matched_c05','Tmax'):.2f} (<= 0.10); gain {gain('matched_c05'):+.1f} (>= 6)")
print(f"P32l persona: two-scale {g('persona','tsC'):.2f}, hollowed {g('persona','tsRc'):.2f}, Tmax {g('persona','Tmax'):.2f} (<= 0.25); type count {g('persona','typecount'):.2f} (>= 0.9); persona gain {gain('persona'):+.1f}")
print(f"P32m benign share of majority-on-task accounts q = {Q_TYPE:.4f} (pass if <= 0.03)")
for r in POOLRES:
    if not r["centre_week"] and r["source"] == "metamath_mixed": print(f"P32o A={r['A']} padded, {S32['LINKED']} of {S27['KP']} colluders linked: metadata only {r['link_meta_only']:.2f} (<= 0.5), seeded expansion {r['link_seeded']:.2f} (>= 0.8), recall {r['link_recall']:.2f} (>= 0.5)")
print("=" * 90); print("STEP 32 VERDICT REPORT"); print("=" * 90)
print("[FPR]", json.dumps({d: [round(v, 4) for v in FPR[d]] for d in DETS}))
for r in RES: print(f"[W] {r['source']:>14} " + " ".join(f"{d}={r[d][0]:.2f} tpr1={r[d+'_roc']['tpr1']:.2f}" for d in DETS))
for r in POOLRES: print("[P]", json.dumps(r))
print("[DEP]", json.dumps(DEPS)); print(f"[Q_TYPE] {Q_TYPE:.4f} [ESS] {ESS:.0f} [AUCS] {VOI_AUCS}"); print("[D]", json.dumps({str(k): float(v) for k, v in DRES.items()}), f"base={BASE.mean():.4f}"); print("=" * 90)
