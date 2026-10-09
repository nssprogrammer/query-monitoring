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
# Paper: "The Limits of Query Monitoring against Distributed Model Extraction".
# Run on Kaggle (see "Kaggle:" line below for hardware/time). Seeds are fixed in the configuration cell;
# re-runs vary by a few points (the paper reports run-to-run spread up to ~0.07-0.14 for weekly detection).
#
# | Paper location | Quantity | Value in paper | Where computed here |
# |---|---|---|---|
# | Prediction log (Step 18 rows) | locked predictions P18.1-P18.6 | Pass (conformal 0.0%; lagged 15%); Fail (0.20 vs 0.21; AUC 0.86); Fail (12%; 0.24); Partial (0.84; 0.35); Pass (0.05 vs 0.35); Pass (79–80%; MMLU 2–7%) | final cell "Predictions, figure, VERDICT REPORT", prints "P18.1" ... "P18.6" |
# | §5 'It detects topic shift, not extraction'; App. D.3 | benign groups sweeping the GSM8K test set from 3 or 10 accounts: two-scale alarm rate | 79–80% of weeks | Part Q/E weekly loop (Cell 8), rows `eval_gsm8k(3x300)` / `eval_gsm8k(10x100)`; print "P18.6 benign pipeline" |
# | App. D.3 | broad MMLU sweeps: two-scale alarm rate | 2–7% | Part Q/E weekly loop (Cell 8), rows `eval_mmlu(3x300)` / `eval_mmlu(10x100)`; print "P18.6 benign pipeline" |
# | App. F 'Query-level pooling' | null-week false alarms of per-cell lagged scoring with shrinkage (`qc_ewma`) and of gating behind the two-scale test (`qc_gate`) | 15%; 12% | Cell 8, `FPR[...]` print "[weekly] held-out FPR" |
# | App. F 'Query-level pooling' | full-history conformal threshold (`qc_conf`): reached calibration; power on padded campaigns | 0.20 of the time—no better than the two-scale test (Prediction log: 0.20 vs 0.21) | Cell 8 `FPR["qc_conf"]` and row `metamath_mixed`; print "P18.2/3 qc_conf" |
# | App. D.4 ('AUC up to 0.86'); App. F 'Query-level pooling' (AUC 0.82–0.86, range shared with Step 17) | QC repairs rank padded campaigns | AUC up to 0.86 | Cell 8, `qc_ewma_roc` AUC on `metamath_mixed` / `metamath_followup`; print "P18.2/3 ... AUC mixed" |
# | Table 6, col "QC" (plausibly from this run: the column is the repaired per-cell lagged QC and has no TPR at 5%) | AUC / TPR at 1%: MetaMathQA; human GSM8K; half natural mixed; real follow-up; random natural | 0.94 / 0.64; 1.00 / 0.93; 0.86 / 0.31; 0.85 / 0.27; 0.52 / 0.10 | Cell 8 `roc(...)` for `qc_ewma`; verdict "ROC:" lines `qc_ewma:auc=...,tpr1=...` |
# | App. D.5 'Smaller colluder shares'; Prediction log (Step 18, P18.4) | two-scale oracle detection, campaign fixed at 1,640 queries, A = 250 / 500 / 1,000 | 0.84, 0.60, 0.35 at 20%, 10%, 5% | Part S (Cell 10), `RESS` field `oracle`; print "P18.4/5" |
# | App. D.5 'Smaller colluder shares' | deployed false alarms in multi-week populations | 14–90% | Part S (Cell 10), `RESS` field `fpr`; "[S]" verdict lines |
# | Prediction log (Step 18, P18.5) | at A = 1,000: deployed vs oracle two-scale power | 0.05 vs 0.35 | Part S (Cell 10), `RESS` fields `deployed` / `oracle`; print "P18.4/5 A=1000" |
#
# Not reported in the paper: weekly paired detection of the main campaigns (MetaMathQA, mixed, follow-up, human GSM8K, random, wild_cluster) for two-scale/global/MMD/qc in this run; the Step 17 `qc` variant; global/MMD/QC and random/mixed sources in Part S; FPR confidence intervals; step18_figure.png. (Fig. 4b per-detector false alarms are not computed here.)

# %% [markdown]
# # Step 18 — Repairing query-cluster calibration, a clean scale test, and false alarms from benign evaluation pipelines
# **Kaggle:** GPU + Internet + `HF_TOKEN` → **Run All** (≈70–90 min; reuses the Step 11/13/14/15 caches). `QUICK=True` first.
#
# **Q — query-cluster (QC) calibration repairs** (weekly populations, K = 50, volume-matched; all variants share one per-cell count regression fitted on the 8 preceding weeks):
# * `qc` — Step 17 (pooled robust standardisation).
# * `qc_ewma` — **per-cell lagged scoring**: each cell's offset is an exponentially weighted mean (half-life 2 weeks) of its own residuals in the preceding weeks; per-cell scales are **shrunk hierarchically** toward the median scale across cells (prior weight 20 windows).
# * `qc_conf` — same score, **conformal threshold over the expanding history of all prior weeks** (each scored with its own lagged offsets).
# * `qc_gate` — `qc_ewma` **gated behind the two-scale test** (counted only when the two-scale score exceeds its 80th null percentile), with the joint threshold calibrated on the same windows.
#
# **S — scale with campaign size held fixed.** Rolling 4-week populations (calibration on the 8 preceding weeks, lagged reference), **N = 1,640 queries** over K = 50 colluders (volume shapes from benign users, rescaled to N), A = 250 / 500 / 1,000 (colluder share 20% / 10% / 5%). Reported both **as deployed** (calibrated thresholds) and as an **oracle** (thresholds from the pooled held-out null windows at the same A).
#
# **E — benign evaluation pipelines.** Benign groups that sweep a benchmark — GSM8K test questions or MMLU questions — from 3 accounts × 300 queries or 10 accounts × 100 queries, injected into null weeks. Their detection rate is a false-alarm rate on legitimate coordinated traffic.
#
# | ID | Locked prediction |
# |---|---|
# | **P18.1** | `qc_ewma` or `qc_conf` reaches held-out FPR ≤ 2% (weekly) |
# | **P18.2** | The best calibrated QC variant (FPR ≤ 2%) has paired power on mixed padding ≥ two-scale + 0.15, with AUC ≥ 0.75 |
# | **P18.3** | `qc_gate`: FPR ≤ 1.5% and power on mixed padding ≥ 0.3 |
# | **P18.4** | Fixed N: two-scale **oracle** TPR at 1% on MetaMathQA ≥ 0.6 at A = 250 and ≤ 0.3 at A = 1,000 |
# | **P18.5** | At A = 1,000, deployed two-scale power ≤ oracle − 0.2 (calibration is part of the scale failure) |
# | **P18.6** | A GSM8K evaluation sweep from 3 accounts × 300 queries triggers the two-scale test in ≥ 50% of weeks |

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


# %% [markdown]
# ## Part Q/E — weekly: QC calibration repairs, mimicry, benign evaluation pipelines

# %%
# ===== Cell 8: weekly populations — calibration, paired detection, ROC at matched realised FPR, cumulative detection =====
t0 = time.time(); blk = WEEK.astype(np.int32); n_b = int(blk.max()) + 1
order = np.lexsort((ACCT, blk)); keys = blk[order].astype(np.int64) * n_acct + ACCT[order]; GA = {b: [] for b in range(n_b)}
for g in np.split(order, np.flatnonzero(np.diff(keys)) + 1):
    if not is_pool[int(ACCT[g[0]])] and len(g) >= MIN_PER_ACCT: GA[int(blk[g[0]])].append(g[:VOL_CAP])
ok = [b for b in range(WK["CAL_N"], n_b) if len(GA[b]) >= 40 and all(len(GA[v]) >= 20 for v in range(b - WK["CAL_N"], b))]
TB = [int(ok[i]) for i in np.linspace(0, len(ok) - 1, min(WK["N_TEST"], len(ok))).astype(int)]
def sub_window(b, r):
    k = max(20, int(SUB_FRAC * len(GA[b]))); pick = r.choice(len(GA[b]), k, replace=False); parts = [GA[b][i] for i in pick]
    vols = np.array([len(p) for p in parts]); return np.concatenate(parts), np.repeat(np.arange(len(parts)), vols), vols
NEED = sorted(set(v for b in TB for v in range(b - WK["CAL_N"], b + 1))); BANK, REFS, NULLW = {}, {}, {}
for b in NEED:
    prev = [np.concatenate(GA[v]) for v in range(max(0, b - WK["LAGB"]), b) if GA[v]]
    ref = np.concatenate(prev) if prev else np.concatenate(GA[b]); ref = np.random.default_rng(b).choice(ref, min(100000, len(ref)), replace=False)
    BANK[b] = (make_bank_aug(Z[ref], H_COARSE, SEED + 7), make_bank_aug(Z[ref], H_FINE, SEED + 8))
    rb = np.random.default_rng(b + 1).choice(ref, N_REF_BATCH * REF_BATCH, replace=len(ref) < N_REF_BATCH * REF_BATCH).reshape(N_REF_BATCH, REF_BATCH)
    REFS[b] = [torch.as_tensor(EN[x].astype(np.float32), device=DEV) if USE_GPU else EN[x].astype(np.float32) for x in rb]
    NULLW[b] = [window_stats15(*sub_window(b, np.random.default_rng(SEED + 100 * b + t))[:2], np.bincount(sub_window(b, np.random.default_rng(SEED + 100 * b + t))[1]), *BANK[b], REFS[b]) for t in range(WK["NULL_PER"])]
print(f"[weekly] test weeks {TB}; banks and null windows for {len(NEED)} weeks ({time.time()-t0:.0f}s)")
THR = {b: calibrate15([r for v in range(b - WK["CAL_N"], b) for r in NULLW[v]]) for b in TB}
for b in TB:
    for r in NULLW[b]: add_scores(r, THR[b])
# ---- QC calibration repairs: per-cell lagged (EWMA) offsets, hierarchical shrinkage, conformal over weeks, gating
def fit_cells(recs):
    TS = np.array([r["cell_ts"] for r in recs]); SH = np.array([r["cell_share"] for r in recs]); NT = np.array([r["n_total"] for r in recs], float)
    LN = np.log(np.maximum(SH * NT[:, None], 1.0)); valid = np.mean(~np.isnan(TS), 0) >= 0.5
    A_, B_, SC, KN = np.zeros(N_TOPICS), np.zeros(N_TOPICS), np.ones(N_TOPICS), np.zeros(N_TOPICS)
    for t in np.flatnonzero(valid):
        ok = ~np.isnan(TS[:, t]); x, y = LN[ok, t], TS[ok, t]
        b_, a_ = np.polyfit(x, y, 1) if np.ptp(x) > 0 else (0.0, float(np.median(y)))
        res = y - (a_ + b_ * x); A_[t], B_[t] = a_, b_; SC[t] = 1.4826 * np.median(np.abs(res - np.median(res))); KN[t] = ok.sum()
    med = float(np.median(SC[valid])) if valid.any() else 1.0
    SCs = np.maximum((KN * SC + 20.0 * med) / (KN + 20.0), 0.05)
    return dict(A=A_, B=B_, SCs=SCs, valid=valid)
def cell_res(r, F):
    ln = np.log(np.maximum(r["cell_share"] * r["n_total"], 1.0))
    return np.where(F["valid"] & ~np.isnan(r["cell_ts"]), r["cell_ts"] - (F["A"] + F["B"] * ln), np.nan)
def ewma(weeks, M, half=2.0):
    if not weeks: return np.zeros(N_TOPICS)
    W = np.array([0.5 ** ((weeks[-1] - w) / half) for w in weeks]); X = np.array([M[w] for w in weeks])
    num = np.nansum(X * W[:, None], 0); den = np.sum(np.where(np.isnan(X), 0.0, 1.0) * W[:, None], 0)
    return np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
def qc_score(r, F, off):
    with np.errstate(all="ignore"): z = (cell_res(r, F) - off) / F["SCs"]
    return float(np.nanmax(z)) if np.any(~np.isnan(z)) else -1e9
QCV = {}
for b in TB:
    Cw = list(range(b - WK["CAL_N"], b)); F = fit_cells([r for v in Cw for r in NULLW[v][0::2]])
    hist = sorted(v for v in NULLW if v < b)
    with np.errstate(all="ignore"): M = {v: np.nanmean(np.array([cell_res(r, F) for r in NULLW[v]]), 0) for v in hist}
    off = {v: ewma([w for w in hist if w < v][-8:], M) for v in hist + [b]}
    cal = {v: [qc_score(r, F, off[v]) for r in NULLW[v][1::2]] for v in hist}
    g80 = float(np.quantile([r["twoscale"] for v in Cw for r in NULLW[v][1::2]], 0.8))
    gated = [cal[v][j] if r["twoscale"] > g80 else -1e9 for v in Cw for j, r in enumerate(NULLW[v][1::2])]
    QCV[b] = dict(F=F, off=off[b], g80=g80, ewma=float(np.quantile(np.concatenate([cal[v] for v in Cw]), 1 - ALPHA)),
                  conf=float(np.quantile(np.concatenate([cal[v] for v in hist]), 1 - ALPHA)), gate=float(np.quantile(gated, 1 - ALPHA)))
DETS18 = ["twoscale", "qc", "qc_ewma", "qc_conf", "qc_gate", "glob", "mmd"]
def score18(s, b):
    v = qc_score(s, QCV[b]["F"], QCV[b]["off"]); s["qc_ewma"] = s["qc_conf"] = v; s["qc_gate"] = v if s["twoscale"] > QCV[b]["g80"] else -1e9; return s
def fires18(s, b):
    f = fires15(s, THR[b])
    f.update(qc_ewma=s["qc_ewma"] > QCV[b]["ewma"], qc_conf=s["qc_conf"] > QCV[b]["conf"], qc_gate=s["qc_gate"] > QCV[b]["gate"]); return f
for b in TB:
    for r in NULLW[b]: score18(r, b)
# [PAPER] App. F 'Query-level pooling' / Prediction log (Step 18, P18.1, P18.3): held-out FPR qc_ewma 15%, qc_gate 12%, qc_conf 0.0%
FPR = {d: boot_ci([np.mean([fires18(r, b)[d] for r in NULLW[b]]) for b in TB]) for d in DETS18}
print("[weekly] held-out FPR: " + " ".join(f"{d}={FPR[d][0]:.3f}" for d in DETS18) + f"  ({time.time()-t0:.0f}s)")
RES = []; WEEKP = {}
CONFIGS = [(s_, 50 if not QUICK else 20, None) for s_ in SRC_MAIN + ["wild_cluster"]] + [(s_, k_, n_) for s_, k_, n_ in EVALS]
base_cache = {}
for src, K_, n_fixed in CONFIGS:
    pw = {d: [] for d in DETS18}; nv, av = {d: [] for d in DETS18}, {d: [] for d in DETS18}
    for b in TB:
        acc = {d: [] for d in DETS18}
        for rep in range(WK["REPS"]):
            r = np.random.default_rng(SEED + 9_000_000 + 1000 * b + 10 * rep + K_); idx, lab, vols = sub_window(b, r); A_ = int(lab.max()) + 1
            if K_ >= A_ // 2: continue
            key = (b, rep, K_)
            if key not in base_cache:
                sb_ = add_scores(window_stats15(idx, lab, np.bincount(lab, minlength=A_), *BANK[b], REFS[b]), THR[b]); base_cache[key] = score18(sb_, b)
            sb = base_cache[key]; fb = fires18(sb, b)
            cv = np.full(K_, n_fixed) if n_fixed else r.choice(np.array([len(g) for g in GA[b]]), K_, replace=True); N = int(cv.sum())
            Zc, Ec, Tc = campaign_points(src, N, np.random.default_rng(SEED + 17 * b + rep + K_ + len(src))); N = len(Zc)
            labc = np.repeat(np.arange(K_), cv)[:N]; keep = lab >= K_
            _, inv = np.unique(np.concatenate([labc, lab[keep]]), return_inverse=True)
            s_ = window_stats15(idx[keep], inv[N:], np.bincount(inv), *BANK[b], REFS[b], extra=(Zc, inv[:N]), en_extra=Ec, top_extra=Tc)
            s_ = score18(add_scores(s_, THR[b]), b); f = fires18(s_, b)
            for d in DETS18: acc[d].append(float(f[d] and not fb[d])); nv[d].append(sb[d]); av[d].append(s_[d])
        for d in DETS18:
            if acc[d]: pw[d].append(np.mean(acc[d]))
    lab_ = src if n_fixed is None else f"{src}({K_}x{n_fixed})"; WEEKP[lab_] = pw
    # [PAPER] Table 6 col "QC" (plausibly this run, qc_ewma AUC / TPR at 1%): 0.94/0.64, 1.00/0.93, 0.86/0.31, 0.85/0.27, 0.52/0.10; App. D.4 "AUC up to 0.86"
    row = dict(source=lab_, K=K_, **{d: boot_ci(pw[d]) for d in DETS18 if pw[d]}, **{d + "_roc": roc(nv[d], av[d]) for d in DETS18})
    RES.append(row)
    print(f"[weekly] {lab_:>24} | " + " ".join(f"{d}={row[d][0]:.2f}" for d in DETS18) + " | AUC " + " ".join(f"{d}={row[d+'_roc']['auc']:.2f}" for d in ["twoscale", "qc_ewma", "glob", "mmd"]) + f"  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Part S — scale with campaign size fixed

# %%
# ===== Cell 10: scale with campaign size fixed (N fixed, A varied), deployed and oracle power =====
t0 = time.time(); positions = [t for t in range(POP["FIRST"], n_b, POP["STEP"]) if t - 11 >= 0]
def aw_pool(w0, w1): return [g for w in range(max(0, w0), w1 + 1) for g in GA.get(w, [])]
def pop_window(pool, A_, r):
    pick = r.choice(len(pool), min(A_, len(pool)), replace=False); parts = [pool[i] for i in pick]
    vols = np.array([len(p) for p in parts]); return np.concatenate(parts), np.repeat(np.arange(len(parts)), vols), vols
DS = ["twoscale", "qc", "glob", "mmd"]; SC_ = {}
for t in positions:
    pool_t = aw_pool(t - 3, t)
    refw = np.concatenate(aw_pool(t - 7, t - 4)); refw = np.random.default_rng(t).choice(refw, min(100000, len(refw)), replace=False)
    BPt = (make_bank_aug(Z[refw], H_COARSE, SEED + 7), make_bank_aug(Z[refw], H_FINE, SEED + 8))
    rb = np.random.default_rng(t + 1).choice(refw, N_REF_BATCH * REF_BATCH, replace=len(refw) < N_REF_BATCH * REF_BATCH).reshape(N_REF_BATCH, REF_BATCH)
    REFt = [torch.as_tensor(EN[x].astype(np.float32), device=DEV) if USE_GPU else EN[x].astype(np.float32) for x in rb]
    for A_ in POP["AS"]:
        if A_ > 0.8 * len(pool_t) or A_ < 3 * POP["K"]: continue
        cal = []
        for j, (w0, w1) in enumerate([(t - 11, t - 8), (t - 7, t - 4)]):
            pc = aw_pool(w0, w1)
            for k in range(POP["N_CAL"] // 2):
                idx, lab, vols = pop_window(pc, min(A_, int(0.8 * len(pc))), np.random.default_rng(SEED + 1000 * t + 100 * j + k + A_)); cal.append(window_stats15(idx, lab, np.bincount(lab), *BPt, REFt))
        TP = calibrate15(cal)
        for k in range(POP["N_TEST"]):
            idx, lab, vols = pop_window(pool_t, A_, np.random.default_rng(SEED + 7000 * t + k + A_)); sn = add_scores(window_stats15(idx, lab, np.bincount(lab), *BPt, REFt), TP)
            f = fires15(sn, TP)
            for d in DS: SC_.setdefault((A_, "null", d), []).append((t, float(f[d]), sn[d]))
        for src in ["metamath_unique", "metamath_mixed", "random"]:
            for rep in range(POP["REPS"]):
                r = np.random.default_rng(SEED + 9000 * t + rep + A_); idx, lab, vols = pop_window(pool_t, A_, r)
                sb = add_scores(window_stats15(idx, lab, np.bincount(lab), *BPt, REFt), TP); fb = fires15(sb, TP)
                cv = r.choice(vols, POP["K"], replace=True).astype(float); cv = np.maximum(1, np.round(cv * POP["N_FIX"] / cv.sum())).astype(int)
                N = int(cv.sum()); Zc, Ec, Tc = campaign_points(src, N, r); N = len(Zc); labc = np.repeat(np.arange(POP["K"]), cv)[:N]; keep = lab >= POP["K"]
                _, inv = np.unique(np.concatenate([labc, lab[keep]]), return_inverse=True)
                s_ = add_scores(window_stats15(idx[keep], inv[N:], np.bincount(inv), *BPt, REFt, extra=(Zc, inv[:N]), en_extra=Ec, top_extra=Tc), TP); f = fires15(s_, TP)
                for d in DS: SC_.setdefault((A_, src, d), []).append((t, float(f[d] and not fb[d]), s_[d]))
    print(f"[scale] position week {t}: population {len(pool_t)} account-weeks ({time.time()-t0:.0f}s)")
RESS = []
# [PAPER] App. D.5 'Smaller colluder shares': two-scale oracle 0.84, 0.60, 0.35 at 20%, 10%, 5%; deployed false alarms 14–90%
for A_ in POP["AS"]:
    if (A_, "null", "twoscale") not in SC_: continue
    for d in DS:
        nullv = np.array([v for _, _, v in SC_[(A_, "null", d)]]); fpr = float(np.mean([x for _, x, _ in SC_[(A_, "null", d)]])); thr = np.quantile(nullv, 0.99)
        for src in ["metamath_unique", "metamath_mixed", "random"]:
            L = SC_[(A_, src, d)]; RESS.append(dict(A=A_, share=POP["K"] / A_, detector=d, source=src, fpr=fpr, deployed=float(np.mean([x for _, x, _ in L])), oracle=float(np.mean([v > thr for _, _, v in L]))))
for r in RESS:
    if r["source"] != "random" or r["detector"] == "twoscale":
        print(f"[scale] A={r['A']:5d} ({100*r['share']:.0f}%) {r['detector']:>8} {r['source']:>16}: deployed {r['deployed']:.2f} (FPR {r['fpr']:.3f}) | oracle TPR@1% {r['oracle']:.2f}")

# %% [markdown]
# ## Predictions, figure, VERDICT REPORT — copy into the chat

# %%
# ===== Cell 12: predictions, figure, verdict =====
fmt = lambda x: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
G = lambda src, d: next((r[d][0] for r in RES if r["source"] == src and d in r), None)
GA_ = lambda src, d, k="auc": next((r[d + "_roc"][k] for r in RES if r["source"] == src), None)
# [PAPER] Prediction log (Step 18, P18.1): Pass (conformal 0.0%; lagged 15%)
print(f"P18.1 FPR: qc {FPR['qc'][0]:.3f}, qc_ewma {FPR['qc_ewma'][0]:.3f} [{FPR['qc_ewma'][1]:.3f},{FPR['qc_ewma'][2]:.3f}], qc_conf {FPR['qc_conf'][0]:.3f} [{FPR['qc_conf'][1]:.3f},{FPR['qc_conf'][2]:.3f}] (pass if either <= 0.02)")
cal_ok = [d for d in ["qc_ewma", "qc_conf", "qc_gate"] if FPR[d][0] <= 0.02]
# [PAPER] Prediction log (Step 18, P18.2/P18.3): qc_conf mixed 0.20 vs two-scale 0.21, AUC 0.86; qc_gate FPR 12%, mixed 0.24
for d in ["qc_ewma", "qc_conf", "qc_gate"]:
    print(f"P18.2/3 {d}: FPR {FPR[d][0]:.3f}; mixed {fmt(G('metamath_mixed', d))}, follow-up {fmt(G('metamath_followup', d))}, unique {fmt(G('metamath_unique', d))} | AUC mixed {fmt(GA_('metamath_mixed', d))} | two-scale mixed {fmt(G('metamath_mixed', 'twoscale'))}")
print("P18.2 calibrated QC variants (FPR <= 2%):", cal_ok)
# [PAPER] Prediction log (Step 18, P18.4/P18.5) and App. D.5: oracle 0.84 (A=250), 0.35 (A=1,000); at A=1,000 deployed 0.05 vs oracle 0.35
for A_ in POP["AS"]:
    r_ = next((r for r in RESS if r["A"] == A_ and r["detector"] == "twoscale" and r["source"] == "metamath_unique"), None)
    if r_: print(f"P18.4/5 A={A_} ({100*r_['share']:.0f}%): two-scale oracle {r_['oracle']:.2f}, deployed {r_['deployed']:.2f} (FPR {r_['fpr']:.3f})")
# [PAPER] §5 'It detects topic shift, not extraction' / App. D.3 / Prediction log (P18.6): GSM8K sweeps 79–80% of weeks; MMLU sweeps 2–7%
for lab_ in [k for k in WEEKP if k.startswith("eval")]:
    print(f"P18.6 benign pipeline {lab_}: two-scale alarm rate {fmt(G(lab_, 'twoscale'))}, qc_ewma {fmt(G(lab_, 'qc_ewma'))}, global {fmt(G(lab_, 'glob'))}, MMD {fmt(G(lab_, 'mmd'))}")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(19, 4.6))
srcs = [r["source"] for r in RES]; xs = np.arange(len(srcs)); w = 0.16
for j, d in enumerate(["twoscale", "qc", "qc_ewma", "qc_conf", "qc_gate"]): ax[0].bar(xs + (j - 2) * w, [G(s_, d) or 0 for s_ in srcs], w, label=f"{d} (FPR {FPR[d][0]:.3f})")
ax[0].set_xticks(xs); ax[0].set_xticklabels(srcs, rotation=30, fontsize=7); ax[0].set_ylim(0, 1.02); ax[0].legend(fontsize=7); ax[0].set_title("weekly paired detection (deployed)")
for j, d in enumerate(["twoscale", "qc_ewma", "glob", "mmd"]): ax[1].bar(xs + (j - 1.5) * w, [GA_(s_, d) or 0 for s_ in srcs], w, label=d)
ax[1].set_xticks(xs); ax[1].set_xticklabels(srcs, rotation=30, fontsize=7); ax[1].set_ylim(0, 1.02); ax[1].axhline(0.5, color="k", lw=0.5); ax[1].legend(fontsize=7); ax[1].set_title("AUC")
for d, c_ in [("twoscale", "tab:blue"), ("qc", "tab:orange"), ("glob", "tab:green"), ("mmd", "tab:red")]:
    rr = sorted([r for r in RESS if r["detector"] == d and r["source"] == "metamath_unique"], key=lambda r: r["A"])
    if rr:
        ax[2].plot([r["A"] for r in rr], [r["oracle"] for r in rr], "o-", color=c_, label=f"{d} oracle"); ax[2].plot([r["A"] for r in rr], [r["deployed"] for r in rr], "s--", color=c_, mfc="white", label=f"{d} deployed")
ax[2].set_xlabel("population A (campaign fixed at N queries over K=50)"); ax[2].set_ylim(-0.02, 1.02); ax[2].legend(fontsize=6.5); ax[2].set_title("scale with N fixed")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/step18_figure.png", dpi=130); plt.show()
print("=" * 90); print("STEP 18 VERDICT REPORT"); print("=" * 90)
print("[FPR weekly] " + " ".join(f"{d}={FPR[d][0]:.4f}[{FPR[d][1]:.4f},{FPR[d][2]:.4f}]" for d in DETS18))
for r in RES:
    print(f"[W] {r['source']:>24} | " + " ".join(f"{d}={r[d][0]:.2f}[{r[d][1]:.2f},{r[d][2]:.2f}]" for d in DETS18 if d in r))
    print("    ROC: " + " ".join(f"{d}:auc={r[d+'_roc']['auc']:.2f},tpr1={r[d+'_roc']['tpr1']:.2f}" for d in DETS18))
for r in RESS: print(f"[S] A={r['A']} share={r['share']:.2f} {r['detector']:>8} {r['source']:>16} deployed={r['deployed']:.2f} fpr={r['fpr']:.3f} oracle={r['oracle']:.2f}")
print("=" * 90)
with open(f"{OUTDIR}/step18_results.json", "w") as fh: json.dump(dict(FPR=FPR, RES=RES, RESS=RESS), fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
