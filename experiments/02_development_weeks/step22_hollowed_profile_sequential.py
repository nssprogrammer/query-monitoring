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
# | Paper location | Quantity | Value in paper | Where computed here |
# |---|---|---|---|
# | App. Table 8 (col "Selection on C"; rows MetaMathQA, Half natural mixed, Real follow-up, Human GSM8K questions, Random natural prompts, GSM8K evaluation sweep 3 accounts) | oracle TPR at matched 1% (AUC) | 0.86 (0.92); 0.21 (0.66); 0.23 (0.66); 0.93 (0.97); 0.02 (0.52); 0.75 (0.87) | Cell 10, `row["tsC_roc"]` from `roc(...)`; verdict-cell "oracle tsC:auc=...,tpr1=..." lines |
# | App. Table 8 (col "Selection on R", same rows) | oracle TPR at matched 1% (AUC) | 1.00 (1.00); 0.94 (1.00); 0.93 (1.00); 1.00 (1.00); 0.00 (0.51); 0.34 (0.61) | Cell 10, `row["tsR_roc"]` |
# | App. Table 8 (col "R on topic cores", same rows) | oracle TPR at matched 1% (AUC) | 1.00 (1.00); 0.95 (1.00); 0.93 (1.00); 1.00 (1.00); 0.00 (0.48); 0.02 (0.54) | Cell 10, `row["coreR_roc"]` |
# | App. Table 8 caption; App. F 'Hollowing' | deployed false alarms for C / R / core variant | 0.8% / 8.4% / 7.6% | Cell 9, `FPR["tsC"]`, `FPR["tsR"]`, `FPR["coreR"]` ("[weekly] held-out FPR" print) |
# | App. P (Step 22, "Hollowing keeps calibration (≤ 1.5%) and power (≥ 0.75)") | hollowed (R) false alarms; MetaMathQA deployed power | Partial (8.4%; 0.90) | Cell 13 print "P22.4" (`FPR["tsR"]`, deployed `tsR` in `RES`) |
# | App. P (Step 22, "Hollowing: real/shuffled λmax (R) ≤ 2") | real / shuffled λmax(R) | Partial (0.35: common drift mode) | Cell 9, `DIAG["ratioR"]` ("[C1]" print); print "P22.1" |
# | App. P (Step 22, "Hollowed selection halves bulk-user groups") | templated bulk-user share of selected groups, C to R | Fail (0.08 to 0.06) | Cell 9, `DIAG["bulkC"]`, `DIAG["bulkR"]`; print "P22.2" |
# | App. P (Step 22, "Group MMD on hollowed groups AUC ≥ 0.6") | group-MMD AUC on MetaMathQA, hollowed groups | Fail (0.18) | Cell 10, `row["gmmdR_roc"]["auc"]`; print "P22.3" |
# | App. P (Step 22, "Profile score ≥ two-scale + 0.1 under padding"); App. F 'Statistics that failed' (profile "detected nothing under padding") | oracle TPR at 1% on mixed padding, profile vs two-scale | Fail (0.00 vs 0.21) | Cell 10, `profC_roc` / `tsC_roc` `tpr1` for metamath_mixed; print "P22.5" |
# | App. P (Step 22, "Profile score flags GSM8K sweeps ≥ 50%") | deployed profile score on GSM8K sweep (3 x 300) | Fail (43%) | Cell 13 print "P22.6" (`RES` deployed `profC`) |
# | App. P (Step 22, "Weeks to detection within 2× of ln(1/α)/D") | martingale median weeks vs ln(1/α)/D | Partial (15 vs 14.7, 16 vs 13.3; Proposition 3 holds unpadded at the floor) | Cell 11, `SEQ[src]["median_martingale"]` vs `SEQ[src]["predicted"]` ("[SEQ]" print; "P22.7/8") |
# | App. P (Step 22, "Martingale median ≤ 6 weeks under padding, all-time FA ≤ 1%") | martingale weeks under padding; null-run all-time false alarms | Fail (15 weeks; FA 0/33) | Cell 11, `SEQ["metamath_mixed"]["median_martingale"]`; `FA_M` over `len(starts)` null runs |
# | Table 2 (row "Two-scale / Half padding", col "Anytime-valid", development weeks) | week of first anytime-valid alarm (GSM8K gain) | week 15 (15.5) | Cell 11 martingale median (metamath_mixed); the gain in parentheses combines this with the Step 19 gain curves (not computed here) |
# | Table 2 (row "Two-scale / Dataset", col "Anytime-valid", development weeks); App. B 'Sequential detection in practice' ("about three weeks") | week of first anytime-valid alarm (GSM8K gain) | week 3 (12.7) | plausibly Cell 11 `SEQ["metamath_unique"]["median_martingale"]` (the conformal "floor" of App. P); gain from Step 19 |
# | App. P, corrections ('Sequential prediction') | slow-down factor predicted by Proposition 3 | about 18 | not isolated (plausibly the ratio of `SEQ[...]["predicted"]` for padded vs unpadded sources) |
# | App. P (Step 22, "Core-plus-halo calibrated power ≥ 0.4") | deployed power on mixed padding | Fail (halo 0.02; hollowed cores 0.90 at 7.6%) | Cell 13 print "P22.9" (`RES` `halo`, `coreR`; `FPR["coreR"]`) |
# | App. F 'Statistics that failed' | core-plus-halo statistic AUC | at chance (AUC 0.48) | Cell 10, `row["halo_roc"]["auc"]` ("[W] ... AUC ... halo=" print); campaign row not stated in the paper |
# | App. P (Step 22, "Prefilter keeps ≥ 95% of colluders at 10% pass"); App. F 'Statistics that failed' ("kept none of the colluders") | colluder retention of the behavioural prefilter | Fail (0%) | Cell 12, `RETAIN` ("[PF]" print); print "P22.10" |
# | App. P (Step 22, "Prefiltering restores power at 5% share") | filtered oracle TPR at 1% | Fail (0.00) | Cell 12, `PF["filtered"]["tpr"]`; print "P22.11" |
#
# Not reported in the paper: deployed power and FPR of `gmmdC`, `gmmdR`, `profR` and `glob`; the CUSUM detection times and false alarms; the KDE divergences `D`; the prefilter's test pass rate `q_test` and unfiltered TPR; the figure `step22_figure.png`.

# %% [markdown]
# # Step 22 — Hollowed group selection, structure-factor profiles, sequential detection, core-plus-halo, and a behavioural prefilter
# **Kaggle:** GPU + Internet + `HF_TOKEN` → **Run All** (≈80–110 min; reuses earlier caches). `QUICK=True` first.
#
# Implements the locked analyses listed under Step 22 in the paper's App. P (prediction log) (Changes 1, 2, 3, 5 and 6 of the revision proposal), on the weekly WildChat protocol with whole-window, lagged calibration.
# * **Hollowed selection (Change 1).** R = D^{-1/2}(YYᵀ − D)D^{-1/2} with D = diag(‖y_a‖²): the top eigenvector picks accounts that move together, not the most clumped single accounts. Real versus shuffled λmax(R); share of selected groups that are templated bulk users (near-duplicate rate > 0.2); group-level MMD on hollowed groups.
# * **Structure-factor profile (Change 2).** Φ(G) = (log S̄_h(G)) for h ∈ {0.25, 0.5, 1, 2, 4, 8}; profile score = robust distance from calibration-window groups of similar coarse concentration (|log S̄₄ − ·| ≤ 0.5).
# * **Sequential detection (Change 3, Proposition 3).** Weekly conformal p-values of the window two-scale score over consecutive weeks; a test martingale with the Vovk–Wang mixture calibrator e(p) = (1 − p + p ln p)/(p ln² p), alarm at 1/α = 100; a CUSUM on ln e calibrated on null runs; D(P₁‖P₀) estimated by KDE from campaign and null scores, and ln(1/α)/D compared with the observed weeks to detection.
# * **Core-plus-halo (Change 6).** Each account's densest topic cell is its core, the rest its halo; padding makes the halo look like independent draws from the population (S̄ ≈ 1). Window statistics: max over accounts of core share × exp(−|log S̄_halo|), and λmax of the hollowed matrix built from cores only.
# * **Behavioural prefilter (Change 5).** Lempel–Ziv entropy rate of each account-week's topic sequence; the filter keeps the 10% least predictable benign account-weeks (threshold from the calibration period). Attacker model: campaign queries in random order (pool sampling). Colluder retention and oracle power on filtered populations of A = 1,000 at 5% colluder share.
#
# Predictions are those locked in the paper's App. P (prediction log), Step 22 rows.

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
H6 = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]; DUP_COS, DUP_CUT = 0.95, 0.2
if QUICK: S22 = dict(K=20, N_FIX=300, REPS=2, SEQ_H=6, A_POOL=80, K_POOL=4, N_POOL=120, NULL_POOL=20, ATT_POOL=8)
else:     S22 = dict(K=50, N_FIX=1640, REPS=4, SEQ_H=16, A_POOL=1000, K_POOL=50, N_POOL=1640, NULL_POOL=100, ATT_POOL=40)

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
# ## Part 1 — projection, topics, statistics

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
# ===== Cell 8: hollowed selection, profiles, core-plus-halo statistics =====
def top_group(v):
    v2 = v ** 2 / (v ** 2).sum(); o = np.argsort(-v2); m = int(np.clip(np.searchsorted(np.cumsum(v2[o]), 0.5) + 1, 3, 100)); return o[:m]
def hollow(Y):
    G = Y @ Y.T / Y.shape[1]; d = np.maximum(np.diag(G).copy(), 1e-12); R = (G - np.diag(d)) / np.sqrt(np.outer(d, d)); ev, V = np.linalg.eigh(R); return float(ev[-1]), V[:, -1]
def dup_rate(E):
    if len(E) < 3: return 0.0
    E = E[np.random.default_rng(0).choice(len(E), min(300, len(E)), replace=False)].astype(np.float32); S = E @ E.T; iu = np.triu_indices(len(E), 1); return float(np.mean(S[iu] > DUP_COS))
def window_stats22(idx_pts, lab, cnt, B6, refs, extra=None, en_extra=None, top_extra=None):
    pts = Z[idx_pts] if extra is None else np.vstack([extra[0], Z[idx_pts]]); labs = lab if extra is None else np.concatenate([extra[1], lab])
    en = EN[idx_pts].astype(np.float32) if extra is None else np.vstack([en_extra, EN[idx_pts].astype(np.float32)]); top = TOPIC[idx_pts] if extra is None else np.concatenate([top_extra, TOPIC[idx_pts]])
    Y = {h: whitened(pts, labs, cnt, B6[h]) for h in H6}; Yc, Yf = Y[4.0], Y[0.5]; r_ = Yc.shape[1]
    evC, VC = np.linalg.eigh(Yc.T @ Yc / r_); vC = Yc @ VC[:, -1]; gC = top_group(vC)
    lR, vR = hollow(Yc); gR = top_group(vR)
    s = dict(lamC=float(evC[-1]), lamR=lR, glob=union_S(Yc, cnt, np.arange(len(cnt))), grpC=gC, grpR=gR, labs=labs, n_total=len(labs))
    for nm, g in [("C", gC), ("R", gR)]:
        s["ts" + nm] = float(np.log(union_S(Yc, cnt, g)) - np.log(union_S(Yf, cnt, g)))
        s["phi" + nm] = np.array([np.log(union_S(Y[h], cnt, g)) for h in H6])
        gm = np.isin(labs, g); ge = en[gm]
        if len(ge) > MMD_WIN: ge = ge[np.random.default_rng(len(labs)).choice(len(ge), MMD_WIN, replace=False)]
        s["gmmd" + nm] = mmd2(ge, refs) if len(ge) >= 20 else 0.0; s["dup" + nm] = dup_rate(en[gm])
    # core-plus-halo
    A_ = len(cnt); scores = np.zeros(A_); core_lab, core_idx, halo_lab, halo_idx = [], [], [], []
    order_ = np.argsort(labs, kind="stable"); bounds = np.searchsorted(labs[order_], np.arange(A_ + 1))
    for a in range(A_):
        ii = order_[bounds[a]:bounds[a + 1]]
        if len(ii) < 10: continue
        tc = np.bincount(top[ii]); c = int(np.argmax(tc)); cm = top[ii] == c
        if cm.sum() >= 5: core_idx.append(ii[cm]); core_lab.append(np.full(cm.sum(), a))
        if (~cm).sum() >= 5: halo_idx.append(ii[~cm]); halo_lab.append(np.full((~cm).sum(), a))
        scores[a] = cm.mean()
    halo_score = 0.0
    if halo_idx:
        hi, hl = np.concatenate(halo_idx), np.concatenate(halo_lab); u, inv = np.unique(hl, return_inverse=True); hc = np.bincount(inv)
        Sh = (whitened(pts[hi], inv, hc, B6[4.0]) ** 2).mean(1)
        sc = scores[u] * np.exp(-np.abs(np.log(np.maximum(Sh, 1e-6)))); sc[scores[u] < 0.3] = 0; halo_score = float(sc.max())
    s["halo"] = halo_score; s["coreR"] = 0.0
    if len(core_idx) >= 5:
        ci, cl = np.concatenate(core_idx), np.concatenate(core_lab); u, inv = np.unique(cl, return_inverse=True)
        s["coreR"] = hollow(whitened(pts[ci], inv, np.bincount(inv), B6[4.0]))[0]
    return s
def profile_score(phi, cal_phi, self_ix=None):
    d4 = np.abs(cal_phi[:, 4] - phi[4]); mask = d4 <= 0.5
    if self_ix is not None: mask[self_ix] = False
    if mask.sum() < 20:
        o = np.argsort(d4); o = o[o != self_ix] if self_ix is not None else o; mask = np.zeros(len(cal_phi), bool); mask[o[:30]] = True
    ref = cal_phi[mask]; med = np.median(ref, 0); mad = np.maximum(1.4826 * np.median(np.abs(ref - med), 0), 0.05)
    return float(np.sqrt(np.mean(((phi - med) / mad) ** 2)))


# %% [markdown]
# ## Part W — weekly calibration (six bandwidths) and Change-1 diagnostics

# %%
# ===== Cell 9: weekly populations with six bandwidths; calibration and held-out FPR =====
t0 = time.time(); blk = WEEK.astype(np.int32); n_b = int(blk.max()) + 1
order = np.lexsort((ACCT, blk)); keys = blk[order].astype(np.int64) * n_acct + ACCT[order]; GA = {b: [] for b in range(n_b)}
for g in np.split(order, np.flatnonzero(np.diff(keys)) + 1):
    if not is_pool[int(ACCT[g[0]])] and len(g) >= MIN_PER_ACCT: GA[int(blk[g[0]])].append(g[:VOL_CAP])
ok = [b for b in range(WK["CAL_N"], n_b) if len(GA[b]) >= 40 and all(len(GA[v]) >= 20 for v in range(b - WK["CAL_N"], b))]
TB = [int(ok[i]) for i in np.linspace(0, len(ok) - 1, min(WK["N_TEST"], len(ok))).astype(int)]
SEQW = ok if not QUICK else ok[:8]
def sub_window(b, r):
    k = max(20, int(SUB_FRAC * len(GA[b]))); pick = r.choice(len(GA[b]), k, replace=False); parts = [GA[b][i] for i in pick]
    vols = np.array([len(p) for p in parts]); return np.concatenate(parts), np.repeat(np.arange(len(parts)), vols), vols
NEED = sorted(set(v for b in set(TB) | set(SEQW) for v in range(b - WK["CAL_N"], b + 1) if v >= 0)); B6, REFS, NULLW = {}, {}, {}
for b in NEED:
    prev = [np.concatenate(GA[v]) for v in range(max(0, b - WK["LAGB"]), b) if GA[v]]
    ref = np.concatenate(prev) if prev else np.concatenate(GA[b]); ref = np.random.default_rng(b).choice(ref, min(100000, len(ref)), replace=False)
    B6[b] = {h: make_bank_aug(Z[ref], h, SEED + 7) for h in H6}
    rb = np.random.default_rng(b + 1).choice(ref, N_REF_BATCH * REF_BATCH, replace=len(ref) < N_REF_BATCH * REF_BATCH).reshape(N_REF_BATCH, REF_BATCH)
    REFS[b] = [torch.as_tensor(EN[x].astype(np.float32), device=DEV) if USE_GPU else EN[x].astype(np.float32) for x in rb]
    NULLW[b] = []
    for t in range(WK["NULL_PER"]):
        idx, lab, vols = sub_window(b, np.random.default_rng(SEED + 100 * b + t)); NULLW[b].append(window_stats22(idx, lab, np.bincount(lab), B6[b], REFS[b]))
print(f"[weekly] banks (6 bandwidths) and null windows for {len(NEED)} weeks ({time.time()-t0:.0f}s)")
DET22 = ["tsC", "tsR", "profC", "profR", "gmmdC", "gmmdR", "halo", "coreR", "glob"]
CAL = {}
for b in sorted(set(TB) | set(SEQW)):
    cal = [r for v in range(b - WK["CAL_N"], b) for r in NULLW[v]]; PC = np.array([r["phiC"] for r in cal]); PR = np.array([r["phiR"] for r in cal])
    for j, r in enumerate(cal): r["_profC"], r["_profR"] = profile_score(r["phiC"], PC, j), profile_score(r["phiR"], PR, j)
    th = {d: float(np.quantile([r[d] for r in cal], 1 - ALPHA)) for d in ["tsC", "tsR", "gmmdC", "gmmdR", "halo", "coreR", "glob"]}
    th["profC"] = float(np.quantile([r["_profC"] for r in cal], 1 - ALPHA)); th["profR"] = float(np.quantile([r["_profR"] for r in cal], 1 - ALPHA))
    CAL[b] = dict(th=th, PC=PC, PR=PR, ts=np.array([r["tsC"] for r in cal]))
def score22(s, b): s["profC"] = profile_score(s["phiC"], CAL[b]["PC"]); s["profR"] = profile_score(s["phiR"], CAL[b]["PR"]); return s
def fires22(s, b): return {d: s[d] > CAL[b]["th"][d] for d in DET22}
for b in TB:
    for r in NULLW[b]: score22(r, b)
# [PAPER] App. Table 8 caption, App. F 'Hollowing': deployed false alarms 0.8% (C = tsC), 8.4% (R = tsR), 7.6% (core variant = coreR)
FPR = {d: boot_ci([np.mean([fires22(r, b)[d] for r in NULLW[b]]) for b in TB]) for d in DET22}
print("[weekly] held-out FPR: " + " ".join(f"{d}={FPR[d][0]:.3f}" for d in DET22) + f"  ({time.time()-t0:.0f}s)")
# Change 1 diagnostics: real vs shuffled lambda, bulk-user share of selected groups
lamC_r, lamR_r, lamC_s, lamR_s = [], [], [], []
for b in TB:
    for t in range(3):
        r = np.random.default_rng(SEED + 300 * b + t); idx, lab, vols = sub_window(b, r); s = NULLW[b][t]
        lamC_r.append(s["lamC"]); lamR_r.append(s["lamR"]); idx2 = idx[r.permutation(len(idx))]
        ss = window_stats22(idx2, lab, np.bincount(lab), B6[b], REFS[b]); lamC_s.append(ss["lamC"]); lamR_s.append(ss["lamR"])
# [PAPER] App. P (Step 22): real/shuffled λmax(R) 0.35 (ratioR); bulk-user share of selected groups 0.08 to 0.06 (bulkC -> bulkR)
DIAG = dict(ratioC=float(np.median(lamC_r) / np.median(lamC_s)), ratioR=float(np.median(lamR_r) / np.median(lamR_s)),
            bulkC=float(np.mean([r["dupC"] > DUP_CUT for b in TB for r in NULLW[b]])), bulkR=float(np.mean([r["dupR"] > DUP_CUT for b in TB for r in NULLW[b]])))
print(f"[C1] lambda_max real/shuffled: C {DIAG['ratioC']:.2f}, hollowed R {DIAG['ratioR']:.2f} | selected groups that are templated bulk users (dup>{DUP_CUT}): C {DIAG['bulkC']:.2f}, R {DIAG['bulkR']:.2f}")

# %% [markdown]
# ## Campaigns: hollowed selection, profiles, core-plus-halo

# %%
# ===== Cell 10: campaigns (weekly, K=50, volume-matched), deployed and oracle comparisons =====
t0 = time.time(); RES = []; ATT_SCORES = {}
SRCS = ["metamath_unique", "metamath_mixed", "metamath_followup", "gsm8k_human", "random", ("eval_gsm8k", 3, 300)]
def attacked(b, src, r, K_=None, n_fixed=None, N=None):
    idx, lab, vols = sub_window(b, r); A_ = int(lab.max()) + 1; K_ = K_ or S22["K"]
    if K_ >= A_ // 2: return None
    cv = np.full(K_, n_fixed) if n_fixed else r.choice(np.array([len(g) for g in GA[b]]), K_, replace=True)
    if N: cv = np.maximum(1, np.round(cv * N / cv.sum())).astype(int)
    Zc, Ec, Tc = campaign_points(src, int(cv.sum()), r); n = len(Zc); labc = np.repeat(np.arange(K_), cv)[:n]; keep = lab >= K_
    _, inv = np.unique(np.concatenate([labc, lab[keep]]), return_inverse=True)
    return score22(window_stats22(idx[keep], inv[n:], np.bincount(inv), B6[b], REFS[b], extra=(Zc, inv[:n]), en_extra=Ec, top_extra=Tc), b), idx, lab
for src in SRCS:
    name = src if isinstance(src, str) else f"{src[0]}({src[1]}x{src[2]})"; s_, kk, nf = (src, None, None) if isinstance(src, str) else src
    pw = {d: [] for d in DET22}; nv, av = {d: [] for d in DET22}, {d: [] for d in DET22}
    for b in TB:
        acc = {d: [] for d in DET22}
        for rep in range(S22["REPS"]):
            r = np.random.default_rng(SEED + 9_000_000 + 1000 * b + 10 * rep); out = attacked(b, s_, r, K_=kk, n_fixed=nf)
            if out is None: continue
            sa, idx, lab = out; sb = NULLW[b][rep]; fb, fa = fires22(sb, b), fires22(sa, b)
            for d in DET22: acc[d].append(float(fa[d] and not fb[d])); nv[d].append(sb[d]); av[d].append(sa[d])
        for d in DET22:
            if acc[d]: pw[d].append(np.mean(acc[d]))
    # [PAPER] App. Table 8: oracle TPR at matched 1% (AUC) per source for selection on C (tsC_roc), R (tsR_roc), R on topic cores (coreR_roc)
    # [PAPER] App. P (Step 22): group-MMD AUC 0.18 (gmmdR_roc); profile 0.00 vs two-scale 0.21 on mixed padding (tpr1); App. F: core-plus-halo AUC 0.48 (halo_roc)
    row = dict(source=name, **{d: boot_ci(pw[d]) for d in DET22 if pw[d]}, **{d + "_roc": roc(nv[d], av[d]) for d in DET22}); RES.append(row)
    print(f"[W] {name:>22} | deployed " + " ".join(f"{d}={row[d][0]:.2f}" for d in DET22) + " | AUC " + " ".join(f"{d}={row[d+'_roc']['auc']:.2f}" for d in ["tsC", "tsR", "profC", "gmmdR", "halo", "coreR"]) + f"  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Sequential detection (Proposition 3)

# %%
# ===== Cell 11: sequential detection over consecutive weeks (Proposition 3) =====
t0 = time.time(); ALPHA_SEQ = 0.01
def calib_e(p): p = np.clip(p, 1e-12, 1.0); return np.where(p >= 1.0, 0.0, (1 - p + p * np.log(p)) / (p * np.log(p) ** 2))
def pval(score, b): cal = CAL[b]["ts"]; return float((1 + np.sum(cal >= score)) / (len(cal) + 1))
P0 = {w: pval(NULLW[w][0]["tsC"], w) for w in SEQW}; P1, SC1 = {}, {}
for src in ["metamath_unique", "metamath_mixed", "metamath_followup"]:
    P1[src], SC1[src] = {}, []
    for w in SEQW:
        out = attacked(w, src, np.random.default_rng(SEED + 12_000 + w))
        if out is None: continue
        P1[src][w] = pval(out[0]["tsC"], w); SC1[src].append(out[0]["tsC"])
weeks = [w for w in SEQW]
def run_time(pmap, start, H):
    M, C, tM, tC = 1.0, 0.0, None, None
    for k, w in enumerate([w for w in weeks if w >= start][:H]):
        if w not in pmap: continue
        e = float(calib_e(pmap[w])); M *= e; C = max(0.0, C + np.log(max(e, 1e-12)))
        if tM is None and M >= 1 / ALPHA_SEQ: tM = k + 1
        if tC is None and C >= CUSUM_H: tC = k + 1
    return tM, tC
starts = [w for i, w in enumerate(weeks) if len([x for x in weeks if x >= w]) >= S22["SEQ_H"]]
CUSUM_H = 1e9
nullmax = []
for s0 in starts:
    C = 0.0; mx = 0.0
    for w in [w for w in weeks if w >= s0][:S22["SEQ_H"]]: C = max(0.0, C + np.log(max(float(calib_e(P0[w])), 1e-12))); mx = max(mx, C)
    nullmax.append(mx)
CUSUM_H = float(np.quantile(nullmax, 0.99)) + 1e-9
# [PAPER] App. P (Step 22): all-time false alarms on null runs, FA 0/33
FA_M = float(np.mean([run_time(P0, s0, S22["SEQ_H"])[0] is not None for s0 in starts])); FA_C = float(np.mean([run_time(P0, s0, S22["SEQ_H"])[1] is not None for s0 in starts]))
from scipy.stats import gaussian_kde
null_sc = np.array([r["tsC"] for w in SEQW for r in NULLW[w]]); SEQ = {}
for src in P1:
    tm = [run_time(P1[src], s0, S22["SEQ_H"]) for s0 in starts]; m_ = [t[0] if t[0] else np.inf for t in tm]; c_ = [t[1] if t[1] else np.inf for t in tm]
    f1, f0 = gaussian_kde(SC1[src]), gaussian_kde(null_sc); x = np.array(SC1[src]); D = float(np.mean(np.log(np.maximum(f1(x), 1e-12) / np.maximum(f0(x), 1e-12))))
    # [PAPER] App. P (Step 22): martingale median vs ln(1/alpha)/D, 15 vs 14.7 and 16 vs 13.3; 15 weeks under padding
    # [PAPER] Table 2 (row "Two-scale / Half padding", col "Anytime-valid", dev weeks): week 15 (15.5); gain in parentheses from Step 19 (see header)
    SEQ[src] = dict(median_martingale=float(np.median(m_)), median_cusum=float(np.median(c_)), frac_detected=float(np.mean(np.isfinite(m_))), D=D, predicted=float(np.log(1 / ALPHA_SEQ) / max(D, 1e-9)))
    print(f"[SEQ] {src:>18}: martingale median {SEQ[src]['median_martingale']} weeks (detected {SEQ[src]['frac_detected']:.2f} within {S22['SEQ_H']}); CUSUM median {SEQ[src]['median_cusum']}; D={D:.3f} -> ln(1/alpha)/D = {SEQ[src]['predicted']:.1f} weeks")
print(f"[SEQ] null runs: martingale all-time false alarms {FA_M:.3f}, CUSUM {FA_C:.3f} over {len(starts)} runs of {S22['SEQ_H']} weeks  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Behavioural prefilter

# %%
# ===== Cell 12: behavioural prefilter (Lempel-Ziv entropy rate of topic sequences) at 5% colluder share =====
t0 = time.time()
def lz_entropy(x):
    n = len(x); x = list(x); L = 0.0
    for i in range(n):
        k = 1
        while i + k <= n:
            sub = x[i:i + k]; found = False
            for j in range(0, i):
                if x[j:j + k] == sub: found = True; break
            if not found: break
            k += 1
        L += k
    return float(n * np.log2(max(n, 2)) / L) if L > 0 else 0.0
order5 = np.lexsort((ACCT, WEEK)); keys5 = WEEK[order5].astype(np.int64) * n_acct + ACCT[order5]
POOLW = [(int(WEEK[g[0]]), g[:min(VOL_CAP, 200)]) for g in np.split(order5, np.flatnonzero(np.diff(keys5)) + 1) if not is_pool[int(ACCT[g[0]])] and len(g) >= MIN_PER_ACCT]
cut_w = 36 if n_b >= 50 else int(0.6 * n_b)
cal_pool = [g for w, g in POOLW if w < cut_w]; test_pool = [g for w, g in POOLW if w >= cut_w]
Hcal = np.array([lz_entropy(TOPIC[g]) for g in cal_pool[:: max(1, len(cal_pool) // 3000)]]); H_THR = float(np.quantile(Hcal, 0.9))
Htest = {i: lz_entropy(TOPIC[g]) for i, g in enumerate(test_pool)}; q_test = float(np.mean([h > H_THR for h in Htest.values()]))
ret = []
for t in range(200 if not QUICK else 20):
    r = np.random.default_rng(SEED + 13_000 + t); n = int(r.choice([len(g) for g in test_pool])); Zc, Ec, Tc = campaign_points("metamath_unique", n, r)
    ret.append(lz_entropy(Tc[r.permutation(len(Tc))]) > H_THR)
# [PAPER] App. P (Step 22): prefilter keeps 0% of colluders; App. F 'Statistics that failed': kept none of the colluders
RETAIN = float(np.mean(ret)); print(f"[PF] LZ threshold {H_THR:.3f}; benign pass rate q: calibration 0.10, test {q_test:.3f}; MetaMathQA colluder retention {RETAIN:.3f}  ({time.time()-t0:.0f}s)")
refP = np.random.default_rng(12).choice(np.concatenate(cal_pool), min(100000, sum(len(g) for g in cal_pool)), replace=False)
BP = {h: make_bank_aug(Z[refP], h, SEED + 7) for h in [0.5, 4.0]}
def lean_ts(pts, lab, cnt):
    Yc = whitened(pts, lab, cnt, BP[4.0]); Yf = whitened(pts, lab, cnt, BP[0.5]); ev, V = np.linalg.eigh(Yc.T @ Yc / Yc.shape[1]); g = top_group(Yc @ V[:, -1])
    return float(np.log(union_S(Yc, cnt, g)) - np.log(union_S(Yf, cnt, g)))
PF = {}
for mode in ["unfiltered", "filtered"]:
    nulls, atts = [], []
    for t in range(S22["NULL_POOL"] + S22["ATT_POOL"]):
        r = np.random.default_rng(SEED + 14_000 + t); pick = r.choice(len(test_pool), min(S22["A_POOL"], len(test_pool)), replace=False)
        if mode == "filtered": pick = np.array([i for i in pick if Htest[i] > H_THR])
        parts = [test_pool[i] for i in pick]; vols = np.array([len(p) for p in parts]); idx = np.concatenate(parts); lab = np.repeat(np.arange(len(parts)), vols)
        if t < S22["NULL_POOL"]: nulls.append(lean_ts(Z[idx], lab, vols)); continue
        cv = r.choice(np.array([len(g) for g in test_pool]), S22["K_POOL"], replace=True).astype(float); cv = np.maximum(1, np.round(cv * S22["N_POOL"] / cv.sum())).astype(int)
        Zc, Ec, Tc = campaign_points("metamath_unique", int(cv.sum()), r); n = len(Zc); labc = np.repeat(np.arange(S22["K_POOL"]), cv)[:n]
        if mode == "filtered":
            keepc = np.array([lz_entropy(Tc[labc == k]) > H_THR for k in range(S22["K_POOL"])]); m_ = keepc[labc]; Zc, labc = Zc[m_], labc[m_]
            if len(Zc) == 0: atts.append(-1e9); continue
            _, labc = np.unique(labc, return_inverse=True)
        k_eff = int(labc.max()) + 1; _, inv = np.unique(np.concatenate([labc, lab + k_eff]), return_inverse=True)
        atts.append(lean_ts(np.vstack([Zc, Z[idx]]), inv, np.bincount(inv)))
    # [PAPER] App. P (Step 22): prefiltering restores power at 5% share, Fail (0.00) (filtered TPR)
    thr = np.quantile(nulls, 0.99); PF[mode] = dict(tpr=float(np.mean(np.array(atts) > thr)), A_eff=float(np.mean([len(p) for p in [pick]])))
    print(f"[PF] {mode:>10}: oracle TPR at 1% = {PF[mode]['tpr']:.2f}  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Predictions, figure, VERDICT REPORT — copy into the chat

# %%
# ===== Cell 13: predictions, figure, verdict =====
g_ = lambda src, d, k=None: next(((r[d + "_roc"][k] if k else r[d][0]) for r in RES if r["source"] == src and (k or d in r)), None)
print(f"P22.1 lambda_max(R) real/shuffled {DIAG['ratioR']:.2f} (pass if <= 2; C: {DIAG['ratioC']:.2f})")
print(f"P22.2 bulk-user share of selected groups: C {DIAG['bulkC']:.2f} -> R {DIAG['bulkR']:.2f} (pass if falls by >= 50%)")
print(f"P22.3 group-MMD AUC on MetaMathQA: hollowed {g_('metamath_unique','gmmdR','auc'):.2f} vs C {g_('metamath_unique','gmmdC','auc'):.2f} (pass if hollowed >= 0.6)")
# [PAPER] App. P (Step 22): hollowing keeps calibration and power, Partial (8.4%; 0.90)
print(f"P22.4 hollowed two-scale: FPR {FPR['tsR'][0]:.3f} (<= 0.015), MetaMathQA power {g_('metamath_unique','tsR'):.2f} (>= 0.75)")
print(f"P22.5 mixed padding oracle TPR@1%: profile {g_('metamath_mixed','profC','tpr1'):.2f} vs two-scale {g_('metamath_mixed','tsC','tpr1'):.2f} (pass if >= +0.1)")
# [PAPER] App. P (Step 22): profile score flags GSM8K sweeps, Fail (43%)
print(f"P22.6 GSM8K evaluation sweep flagged by the profile score: {g_('eval_gsm8k(3x300)','profC'):.2f} (pass if >= 0.5)")
for src, v in SEQ.items(): print(f"P22.7/8 {src}: martingale median {v['median_martingale']} weeks vs ln(1/alpha)/D {v['predicted']:.1f}; detected within horizon {v['frac_detected']:.2f}")
print(f"P22.8 all-time false alarms on null runs: martingale {FA_M:.3f}, CUSUM {FA_C:.3f} (pass if <= 0.01 and median <= 6 on mixed)")
# [PAPER] App. P (Step 22): core-plus-halo, Fail (halo 0.02; hollowed cores 0.90 at 7.6%)
print(f"P22.9 core-plus-halo on mixed padding: deployed {g_('metamath_mixed','halo'):.2f} (FPR {FPR['halo'][0]:.3f}); core-hollowed {g_('metamath_mixed','coreR'):.2f} (FPR {FPR['coreR'][0]:.3f}) (pass if calibrated and >= 0.4)")
print(f"P22.10 prefilter: test pass rate {q_test:.3f}, colluder retention {RETAIN:.3f} (>= 0.95)")
print(f"P22.11 filtered oracle TPR at 5% share: {PF['filtered']['tpr']:.2f} vs unfiltered {PF['unfiltered']['tpr']:.2f} (pass if filtered >= 0.6)")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6)); srcs = [r["source"] for r in RES]; xs = np.arange(len(srcs)); w = 0.11
for j, d in enumerate(DET22): ax[0].bar(xs + (j - 4) * w, [g_(s_, d, "tpr1") or 0 for s_ in srcs], w, label=f"{d} (FPR {FPR[d][0]:.3f})")
ax[0].set_xticks(xs); ax[0].set_xticklabels(srcs, rotation=25, fontsize=7); ax[0].set_ylim(0, 1.02); ax[0].set_ylabel("oracle TPR at 1%"); ax[0].legend(fontsize=6)
for src, c_ in [("metamath_unique", "tab:red"), ("metamath_mixed", "tab:purple"), ("metamath_followup", "tab:orange")]:
    ax[1].bar(src, SEQ[src]["median_martingale"] if np.isfinite(SEQ[src]["median_martingale"]) else S22["SEQ_H"], color=c_, alpha=0.7); ax[1].plot([src], [SEQ[src]["predicted"]], "kD")
ax[1].set_ylabel("weeks to detection (bars: martingale median; diamonds: ln(1/α)/D)"); ax[1].set_title("sequential detection")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/step22_figure.png", dpi=130); plt.show()
print("=" * 90); print("STEP 22 VERDICT REPORT"); print("=" * 90)
print("[DIAG]", json.dumps(DIAG)); print("[FPR] " + " ".join(f"{d}={FPR[d][0]:.4f}[{FPR[d][1]:.4f},{FPR[d][2]:.4f}]" for d in DET22))
for r in RES:
    print(f"[W] {r['source']:>22} deployed " + " ".join(f"{d}={r[d][0]:.2f}" for d in DET22 if d in r))
    print("    oracle " + " ".join(f"{d}:auc={r[d+'_roc']['auc']:.2f},tpr1={r[d+'_roc']['tpr1']:.2f}" for d in DET22))
for k, v in SEQ.items(): print("[SEQ]", k, json.dumps(v))
print(f"[SEQ] FA martingale {FA_M:.3f}, CUSUM {FA_C:.3f}, runs {len(starts)}")
print(f"[PF] threshold {H_THR:.3f}, q_test {q_test:.3f}, retention {RETAIN:.3f}, " + json.dumps(PF)); print("=" * 90)
with open(f"{OUTDIR}/step22_results.json", "w") as fh: json.dump(dict(DIAG=DIAG, FPR=FPR, RES=RES, SEQ=SEQ, FA=(FA_M, FA_C), PF=PF, q=q_test, retain=RETAIN), fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
