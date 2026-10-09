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
# | Prediction log (Step 20 rows) | locked predictions P20.1-P20.9 | Fail (0.00 vs 0.81, 0.95); Pass (0.8%; others 2.5–12%); Fail (0.00 vs 0.24); Fail (0.31; 0.00); Fail (slope −0.17 vs +0.62); Fail (502 vs 2,081); Fail (ξ = −0.16); Fail (1 of 4 inside band); Pass (all of it) | final cell "Predictions, figure, VERDICT REPORT", prints "P20.1" ... "P20.9" |
# | §5 'Attacks on the monitor'; App. D.6 'Optimising against the statistic' | weekly two-scale detection, prompts ordered greedily against the statistic, no padding | from 0.80 to 0.31 | Part B/D weekly loop (Cell 8): 0.31 = deployed two-scale on `metamath_adaptive` (print "P20.4"); 0.80 plausibly the `metamath_unique` row of the same run |
# | §5 'It detects topic shift, not extraction'; App. D.3 | dataset-versus-chat classifier and topic-concentration statistic on the selected group | ranked campaigns moderately (AUC 0.69–0.84) but caught none at a 1% false-alarm rate | Part B/D weekly loop (Cell 8), `style_grp` / `topic_conc` `_roc` (auc, tpr1); verdict "oracle" lines; print "P20.1" |
# | Table 10, cols "Null median", "Null 99%", "Detector TPR at 1%" (A = 250 / 1,000 / 2,500 / 5,000 / 10,000) | two-scale null median; null 99%; detector oracle TPR at 1% | 1.27, 1.21, 1.05, 0.94, 0.61; 3.28, 3.16, 4.35, 3.18, 2.95; 1.00, 0.80, 0.42, 0.15, 0.00 | Part L (Cell 10), print "[L] A=...: null median ..., 99% ... oracle TPR@1%" (`LAW`) |
# | Table 10, col "True colluder set above threshold"; Fig. 5 series "true colluders scored (oracle)" | true colluder set scored against the same null threshold | 1.00 at every A | Part L (Cell 10), `EVT` field `tpr_oracle_group`; print "[EVT] ... true colluder set" |
# | Fig. 5 series "top eigenvector only"; §5 'Scale: selection fails before the barrier does'; App. D.5 'Up to 10^4 accounts' | two-scale power as colluder share goes from 20% to 0.5% | fell from 1.00 to 0.00 | Part L (Cell 10), `LAW` field `tpr` / `EVT` field `tpr_selected` |
# | App. D.5 'Up to 10^4 accounts' | null threshold of the two-scale score across A | stayed flat between 2.9 and 4.3 | Part L (Cell 10), `LAW` field `q99`; print "[L] ... 99%" |
#
# Not reported in the paper: style_acct (per-account classifier) deployed power except via P20.3/P20.4; global-stream and MMD rows of the weekly table; mixed/follow-up/human GSM8K/random/wild_cluster weekly rows beyond the Prediction log summaries; Gumbel-predicted medians and 99% quantiles (`med_pred`, `q99_pred`); campaign medians; GEV extrapolation to A = 100,000; step20_figure.png. (Fig. 5 series "every outlier eigenvector" comes from a later step, not this script.)

# %% [markdown]
# # Step 20 — Simple source-shift baselines, an adaptive attacker, and Proposition 2's scaling law up to 10⁴ accounts
# **Kaggle:** GPU + Internet + `HF_TOKEN` → **Run All** (≈60–80 min; reuses the Step 11/13/14/15/18 caches). `QUICK=True` first.
#
# **B — the simple baselines the mechanism analysis implies.** A **dataset-versus-chat classifier** (logistic regression on MiniLM embeddings; positives from public datasets *disjoint from every campaign source* — Dolly-15k, SQuAD v2, CommonsenseQA, ARC; negatives from WildChat benign turns of weeks 0–7, before any test week). Window statistics: `style_grp` (mean classifier logit over the queries of the top-eigenvector group, i.e. the two-scale test's own group selection), `style_acct` (max over accounts of the mean logit), `topic_conc` (1 − normalised topic entropy of the selected group). Same rolling whole-window calibration; deployed and oracle (matched realised 1% FPR) power.
#
# **D — an adaptive attacker.** The attacker knows the two-scale statistic, MiniLM, and has its own benign reference (its pool of natural prompts). It greedily orders MetaMathQA prompts so that the running sum of their whitened coarse features (under *its own* bank) stays near zero — the campaign stays all-math but minimises the coarse concentration the two-scale test reads. Detection is then measured with the defender's lagged banks.
#
# **L — Proposition 2's scaling law and the production barrier (Corollary 2).** Null windows of A = 250 … 10,000 benign account-weeks (≥ 5 queries) from the whole year, one reference bank. Under independent benign groups the window maximum is Gumbel with location growing like β ln A; a Gumbel fit at A₀ = 250 predicts the null median and 99% quantile at every A, and — with the campaign's score at A₀ (N = 1,640 over K = 50) — the population size A* at which oracle detection breaks.
#
# | ID | Locked prediction |
# |---|---|
# | **P20.1** | The best simple baseline matches the two-scale test: oracle TPR at 1% on MetaMathQA and human GSM8K ≥ two-scale − 0.05 |
# | **P20.2** | Calibration is the two-scale test's edge: it is the only detector with deployed FPR ≤ 2% |
# | **P20.3** | Under mixed padding, `style_acct` oracle TPR at 1% ≥ two-scale + 0.1 |
# | **P20.4** | The adaptive attacker cuts two-scale paired power to ≤ 0.3, while `style_acct` oracle TPR at 1% stays ≥ 0.5 |
# | **P20.5** | The null median of the two-scale score grows linearly in ln A with a slope within ±50% of the Gumbel scale β fitted at A₀ = 250 |
# | **P20.6** | The predicted breaking point A* is within a factor of 2 of the A at which oracle TPR at 1% crosses 0.5 |
# | **P20.7** | The benign window-maximum tail is heavy: fitted GEV shape ξ > 0 with a 95% bootstrap interval excluding 0 |
# | **P20.8** | The barrier extrapolates out of sample: a max-stable GEV fit at A₀ = 250 predicts the measured 99% threshold at every larger A within its 90% bootstrap band |
# | **P20.9** | Selection, not the barrier, drives the decay: at the largest A, scoring the true colluder set recovers ≥ half of the oracle power lost relative to A₀ |

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
SRC_MAIN = ["metamath_unique", "metamath_mixed", "metamath_followup", "metamath_adaptive", "gsm8k_human", "random"]
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
CACHE20 = f"{OUTDIR}/step20_style_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
CACHE18 = f"{OUTDIR}/step18_evals_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
if QUICK: SCALE = dict(AS=[30, 60, 120], NULL={30: 40, 60: 30, 120: 20}, ATT=8, K=6, N_FIX=200, MINQ=5)
else:     SCALE = dict(AS=[250, 1000, 2500, 5000, 10000], NULL={250: 400, 1000: 300, 2500: 200, 5000: 120, 10000: 80}, ATT=40, K=50, N_FIX=1640, MINQ=5)

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
# ## Part 1 — projection, topics, statistics, classifier, adaptive attacker

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
# ===== Cell 5b: dataset-versus-chat classifier (positives disjoint from every campaign source) =====
from sklearn.linear_model import LogisticRegression
if os.path.exists(CACHE20):
    _c = np.load(CACHE20); POS = _c["POS"]
else:
    if SYNTHETIC_TEST:
        POS = (np.random.default_rng(31).standard_normal((2000, 384)) * 0.6 + D_GSM[:1].astype(np.float32)).astype(np.float16)
    else:
        from datasets import load_dataset
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            os.system("pip install -q sentence-transformers"); from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda"); enc.max_seq_length = 256
        texts = []; rr = np.random.default_rng(SEED)
        for repo, cfg_, split, field in [("databricks/databricks-dolly-15k", None, "train", "instruction"), ("rajpurkar/squad_v2", None, "train", "question"),
                                         ("tau/commonsense_qa", None, "train", "question"), ("allenai/ai2_arc", "ARC-Challenge", "train", "question")]:
            try:
                ds = load_dataset(repo, cfg_, split=split) if cfg_ else load_dataset(repo, split=split); col = ds[field]
                pick = rr.choice(len(col), min(5000, len(col)), replace=False); texts += [str(col[int(i)])[:1000] for i in pick]
                print(f"classifier positives from {repo}: {len(pick)}")
            except Exception as e: print(f"{repo}: skipped ({e})")
        POS = enc.encode(texts, batch_size=512, convert_to_numpy=True, normalize_embeddings=True).astype(np.float16)
    np.savez(CACHE20, POS=POS)
neg_rows = np.flatnonzero((~is_pool[ACCT]) & (WEEK < 8)); neg_rows = np.random.default_rng(4).choice(neg_rows, min(4 * len(POS), len(neg_rows)), replace=False)
Xc = np.vstack([POS.astype(np.float32), EN[neg_rows].astype(np.float32)]); yc = np.r_[np.ones(len(POS)), np.zeros(len(neg_rows))]
clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(Xc, yc); SW, SB = clf.coef_[0].astype(np.float32), float(clf.intercept_[0])
STY = np.empty(len(EN), np.float32)
for s0 in range(0, len(EN), 200000): STY[s0:s0 + 200000] = EN[s0:s0 + 200000].astype(np.float32) @ SW + SB
print(f"style classifier: {len(POS)} dataset positives vs {len(neg_rows)} WildChat negatives (weeks 0-7); mean logit benign {STY[~is_pool[ACCT]].mean():.2f}, "
      f"MetaMathQA {float((DIST['metamath_unique'].astype(np.float32) @ SW + SB).mean()):.2f}, human GSM8K {float((DIST['gsm8k_human'].astype(np.float32) @ SW + SB).mean()):.2f}")

# %%
# ===== Cell 7b: adaptive attacker — greedy ordering of MetaMathQA prompts to cancel coarse concentration under the attacker's own bank =====
t0 = time.time(); ap = np.random.default_rng(8).choice(pool_idx, min(100000, len(pool_idx)), replace=False)
B_att = make_bank_aug(Z[ap], H_COARSE, SEED + 99)
Ymm = whitened(DZ["metamath_unique"], np.arange(len(DZ["metamath_unique"])), np.ones(len(DZ["metamath_unique"])), B_att)   # one row per prompt
NMAX = min(4000 if not QUICK else 400, len(Ymm)); order, Ssum, avail = [], np.zeros(Ymm.shape[1]), np.ones(len(Ymm), bool)
Yt = torch.as_tensor(Ymm, device=DEV) if USE_GPU else Ymm
for step in range(NMAX):
    if USE_GPU:
        sc = ((torch.as_tensor(Ssum, device=DEV)[None, :] + Yt) ** 2).sum(1); sc[~torch.as_tensor(avail, device=DEV)] = float("inf"); i = int(torch.argmin(sc))
    else:
        sc = ((Ssum[None, :] + Ymm) ** 2).sum(1); sc[~avail] = np.inf; i = int(np.argmin(sc))
    order.append(i); avail[i] = False; Ssum = Ssum + Ymm[i]
ADAPT = np.array(order)
rand_ = np.random.default_rng(9).choice(len(Ymm), NMAX, replace=False)
s_ad = float((Ymm[ADAPT[:NMAX]].sum(0) ** 2).mean() / NMAX); s_rd = float((Ymm[rand_].sum(0) ** 2).mean() / NMAX)
print(f"adaptive ordering of {NMAX} prompts: attacker-side coarse S of the union {s_ad:.2f} vs {s_rd:.2f} for a random MetaMathQA subset ({time.time()-t0:.0f}s)")
_cp_old = campaign_points
def campaign_points(src, n, r):
    if src == "metamath_adaptive":
        ix = ADAPT[:min(n, len(ADAPT))]; return DZ["metamath_unique"][ix], DIST["metamath_unique"][ix].astype(np.float32), km.predict(DZ["metamath_unique"][ix]).astype(np.int32)
    return _cp_old(src, n, r)
def window_stats20(idx_pts, lab, cnt, Bc, Bf, refs, extra=None, en_extra=None, top_extra=None):
    s = window_stats15(idx_pts, lab, cnt, Bc, Bf, refs, extra=extra, en_extra=en_extra, top_extra=top_extra)
    sty = STY[idx_pts] if extra is None else np.concatenate([en_extra @ SW + SB, STY[idx_pts]])
    labs, grp = s["labs"], s["grp"]; gm = np.isin(labs, grp)
    s["style_grp"] = float(sty[gm].mean()) if gm.any() else -1e9
    cnt_ = np.bincount(labs); mean_ = np.bincount(labs, weights=sty) / np.maximum(cnt_, 1); s["style_acct"] = float(np.max(np.where(cnt_ >= 10, mean_, -1e9)))
    tg = s["top"][gm]; p = np.bincount(tg, minlength=N_TOPICS) / max(len(tg), 1); H = -np.sum(p[p > 0] * np.log(p[p > 0]))
    s["topic_conc"] = float(1 - H / np.log(N_TOPICS)); return s


# %% [markdown]
# ## Part B/D — weekly: simple baselines and the adaptive attacker

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
    NULLW[b] = [window_stats20(*sub_window(b, np.random.default_rng(SEED + 100 * b + t))[:2], np.bincount(sub_window(b, np.random.default_rng(SEED + 100 * b + t))[1]), *BANK[b], REFS[b]) for t in range(WK["NULL_PER"])]
print(f"[weekly] test weeks {TB}; banks and null windows for {len(NEED)} weeks ({time.time()-t0:.0f}s)")
THR = {b: calibrate15([r for v in range(b - WK["CAL_N"], b) for r in NULLW[v]]) for b in TB}
for b in TB:
    for r in NULLW[b]: add_scores(r, THR[b])
# ---- thresholds for every detector from the 8 preceding weeks (deployed), and oracle comparisons
DETS20 = ["twoscale", "style_grp", "style_acct", "topic_conc", "glob", "mmd"]
TH20 = {b: {d: float(np.quantile([r[d] for v in range(b - WK["CAL_N"], b) for r in NULLW[v]], 1 - ALPHA)) for d in DETS20} for b in TB}
fires20 = lambda s, b: {d: s[d] > TH20[b][d] for d in DETS20}
# [PAPER] Prediction log (Step 20, P20.2): deployed FPR two-scale 0.8%; others 2.5–12%
FPR = {d: boot_ci([np.mean([fires20(r, b)[d] for r in NULLW[b]]) for b in TB]) for d in DETS20}
print("[weekly] held-out FPR: " + " ".join(f"{d}={FPR[d][0]:.3f}" for d in DETS20) + f"  ({time.time()-t0:.0f}s)")
RES = []; base_cache = {}; K1 = 50 if not QUICK else 20
for src in SRC_MAIN + ["wild_cluster"]:
    pw = {d: [] for d in DETS20}; nv, av = {d: [] for d in DETS20}, {d: [] for d in DETS20}
    for b in TB:
        acc = {d: [] for d in DETS20}
        for rep in range(WK["REPS"]):
            r = np.random.default_rng(SEED + 9_000_000 + 1000 * b + 10 * rep + K1); idx, lab, vols = sub_window(b, r); A_ = int(lab.max()) + 1
            if K1 >= A_ // 2: continue
            key = (b, rep)
            if key not in base_cache: base_cache[key] = window_stats20(idx, lab, np.bincount(lab, minlength=A_), *BANK[b], REFS[b])
            sb = base_cache[key]; fb = fires20(sb, b)
            cv = r.choice(np.array([len(g) for g in GA[b]]), K1, replace=True); N = int(cv.sum())
            Zc, Ec, Tc = campaign_points(src, N, np.random.default_rng(SEED + 17 * b + rep + len(src))); N = len(Zc)
            labc = np.repeat(np.arange(K1), cv)[:N]; keep = lab >= K1; _, inv = np.unique(np.concatenate([labc, lab[keep]]), return_inverse=True)
            s_ = window_stats20(idx[keep], inv[N:], np.bincount(inv), *BANK[b], REFS[b], extra=(Zc, inv[:N]), en_extra=Ec, top_extra=Tc); f = fires20(s_, b)
            for d in DETS20: acc[d].append(float(f[d] and not fb[d])); nv[d].append(sb[d]); av[d].append(s_[d])
        for d in DETS20:
            if acc[d]: pw[d].append(np.mean(acc[d]))
    # [PAPER] §5 'Attacks on the monitor' / App. D.6: two-scale on metamath_adaptive 0.31 (from 0.80, plausibly this run's metamath_unique row); App. D.3: style_grp / topic_conc AUC 0.69–0.84, none caught at 1%
    row = dict(source=src, **{d: boot_ci(pw[d]) for d in DETS20 if pw[d]}, **{d + "_roc": roc(nv[d], av[d]) for d in DETS20})
    RES.append(row)
    print(f"[weekly] {src:>19} | deployed " + " ".join(f"{d}={row[d][0]:.2f}" for d in DETS20) + " | oracle TPR@1% " + " ".join(f"{d}={row[d+'_roc']['tpr1']:.2f}" for d in DETS20) + f"  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Part L — Proposition 2's scaling law

# %%
# ===== Cell 10: Proposition 2's scaling law — null maxima and breaking point up to 10^4 accounts =====
t0 = time.time()
order5 = np.lexsort((ACCT, WEEK)); keys5 = WEEK[order5].astype(np.int64) * n_acct + ACCT[order5]
POOL5 = [g[:VOL_CAP] for g in np.split(order5, np.flatnonzero(np.diff(keys5)) + 1) if not is_pool[int(ACCT[g[0]])] and len(g) >= SCALE["MINQ"]]
refL = np.random.default_rng(12).choice(np.concatenate(POOL5), min(100000, sum(len(g) for g in POOL5)), replace=False)
BL = (make_bank_aug(Z[refL], H_COARSE, SEED + 7), make_bank_aug(Z[refL], H_FINE, SEED + 8))
print(f"[L] account-weeks with >= {SCALE['MINQ']} queries: {len(POOL5)}")
def lean_twoscale(pts, lab, cnt, oracle=None):
    Yc = whitened(pts, lab, cnt, BL[0]); Yf = whitened(pts, lab, cnt, BL[1])
    ev, V = np.linalg.eigh(Yc.T @ Yc / BL[0]["r"]); v = Yc @ V[:, -1]; v2 = v ** 2 / (v ** 2).sum()
    o = np.argsort(-v2); m = int(np.clip(np.searchsorted(np.cumsum(v2[o]), 0.5) + 1, 3, 100)); g = o[:m]
    sel = float(np.log(union_S(Yc, cnt, g)) - np.log(union_S(Yf, cnt, g)))
    if oracle is None: return sel, g
    return sel, float(np.log(union_S(Yc, cnt, oracle)) - np.log(union_S(Yf, cnt, oracle)))
def big_window(A_, r):
    pick = r.choice(len(POOL5), min(A_, len(POOL5)), replace=False); parts = [POOL5[i] for i in pick]
    vols = np.array([len(p) for p in parts]); return np.concatenate(parts), np.repeat(np.arange(len(parts)), vols), vols
NUL, ATT, ORC = {}, {}, {}
for A_ in SCALE["AS"]:
    if A_ > 0.9 * len(POOL5): print(f"[L] A={A_} exceeds the pool; skipped"); continue
    NUL[A_] = []
    for t in range(SCALE["NULL"][A_]):
        idx, lab, vols = big_window(A_, np.random.default_rng(SEED + 31 * A_ + t)); NUL[A_].append(lean_twoscale(Z[idx], lab, vols)[0])
    ATT[A_] = []
    for t in range(SCALE["ATT"]):
        r = np.random.default_rng(SEED + 37 * A_ + t); idx, lab, vols = big_window(A_, r); K_ = SCALE["K"]
        cv = r.choice(vols, K_, replace=True).astype(float); cv = np.maximum(1, np.round(cv * SCALE["N_FIX"] / cv.sum())).astype(int)
        Zc, _, _ = campaign_points("metamath_unique", int(cv.sum()), r); N = len(Zc); labc = np.repeat(np.arange(K_), cv)[:N]; keep = lab >= K_
        _, inv = np.unique(np.concatenate([labc, lab[keep]]), return_inverse=True)
        sel_, orc_ = lean_twoscale(np.vstack([Zc, Z[idx[keep]]]), inv, np.bincount(inv), oracle=np.arange(K_))
        ATT[A_].append(sel_); ORC.setdefault(A_, []).append(orc_)
    # [PAPER] Table 10: null median 1.27/1.21/1.05/0.94/0.61, null 99% 3.28/3.16/4.35/3.18/2.95 ("flat between 2.9 and 4.3", App. D.5),
    # detector TPR at 1% 1.00, 0.80, 0.42, 0.15, 0.00 (= Fig. 5 "top eigenvector only")
    print(f"[L] A={A_:6d}: null median {np.median(NUL[A_]):.3f}, 99% {np.quantile(NUL[A_], .99):.3f} | campaign median {np.median(ATT[A_]):.3f} | oracle TPR@1% {np.mean(np.array(ATT[A_]) > np.quantile(NUL[A_], .99)):.2f}  ({time.time()-t0:.0f}s)")
AS_ = sorted(NUL); A0 = AS_[0]; x0 = np.array(NUL[A0]); beta = float(x0.std() * np.sqrt(6) / np.pi); mu = float(x0.mean() - 0.5772 * beta)
LAW = []
for A_ in AS_:
    pm = mu + beta * (np.log(A_ / A0) + 0.3665); p99 = mu + beta * (np.log(A_ / A0) + 4.600)
    LAW.append(dict(A=A_, med=float(np.median(NUL[A_])), med_pred=float(pm), q99=float(np.quantile(NUL[A_], .99)), q99_pred=float(p99),
                    tpr=float(np.mean(np.array(ATT[A_]) > np.quantile(NUL[A_], .99))), camp_med=float(np.median(ATT[A_]))))
slope = float(np.polyfit(np.log([l["A"] for l in LAW]), [l["med"] for l in LAW], 1)[0])
dc = LAW[0]["camp_med"]; A_star = float(A0 * np.exp((dc - mu) / beta - 4.600))
tp = [l["tpr"] for l in LAW]; As = [l["A"] for l in LAW]; A_emp = float("nan")
for (a1, t1), (a2, t2) in zip(zip(As, tp), zip(As[1:], tp[1:])):
    if t1 >= 0.5 > t2: A_emp = float(np.exp(np.log(a1) + (t1 - 0.5) / (t1 - t2) * (np.log(a2) - np.log(a1)))); break
# [PAPER] Prediction log (Step 20, P20.5): Fail (slope −0.17 vs +0.62)
print(f"[L] Gumbel fit at A0={A0}: mu={mu:.3f}, beta={beta:.3f}; empirical slope of null median vs ln A = {slope:.3f}")
# [PAPER] Prediction log (Step 20, P20.6): Fail (502 vs 2,081)
print(f"[L] predicted breaking point A* = {A_star:.0f}; empirical A where oracle TPR@1% crosses 0.5 = {A_emp:.0f}")
# ---- Change 4: GEV tail shape, max-stable extrapolation, selection-vs-barrier decomposition
from scipy.stats import genextreme
def gev_fit(x):
    c, loc, sc = genextreme.fit(x); return -c, loc, sc                     # scipy's c = -xi
def gev_q(q, xi, loc, sc, A_, A0_):
    k = A_ / A0_; locA = loc + sc * ((k ** xi - 1) / xi if abs(xi) > 1e-6 else np.log(k)); scA = sc * k ** xi
    return float(genextreme.ppf(q, -xi, loc=locA, scale=scA))
xi0, loc0, sc0 = gev_fit(np.array(NUL[A0])); rb = np.random.default_rng(3); boots = []
for _ in range(200):
    xb = np.array(NUL[A0])[rb.integers(0, len(NUL[A0]), len(NUL[A0]))]
    try: boots.append(gev_fit(xb))
    except Exception: pass
xis = np.array([b[0] for b in boots]); XI_CI = (float(np.quantile(xis, .025)), float(np.quantile(xis, .975)))
EVT = []
for A_ in AS_ + [100000]:
    pred = gev_q(0.99, xi0, loc0, sc0, A_, A0); band = [gev_q(0.99, *b, A_, A0) for b in boots]
    row = dict(A=A_, q99_pred=pred, band=(float(np.quantile(band, .05)), float(np.quantile(band, .95))))
    if A_ in NUL:
        thr = float(np.quantile(NUL[A_], .99)); row.update(q99=thr, inside=bool(row["band"][0] <= thr <= row["band"][1]),
                                                          tpr_selected=float(np.mean(np.array(ATT[A_]) > thr)), tpr_oracle_group=float(np.mean(np.array(ORC[A_]) > thr)))
    EVT.append(row)
    # [PAPER] Table 10 col "True colluder set above threshold" / Fig. 5 "true colluders scored (oracle)": 1.00 at every A
    print(f"[EVT] A={A_:6d}: predicted 99% threshold {pred:.3f} [{row['band'][0]:.3f}, {row['band'][1]:.3f}]" + (f" | measured {row['q99']:.3f} inside={row['inside']} | TPR selected group {row['tpr_selected']:.2f}, true colluder set {row['tpr_oracle_group']:.2f}" if 'q99' in row else " | extrapolation"))
# [PAPER] Prediction log (Step 20, P20.7): Fail (ξ = −0.16)
print(f"[EVT] GEV shape xi = {xi0:.3f}, 95% bootstrap interval [{XI_CI[0]:.3f}, {XI_CI[1]:.3f}]")

# %% [markdown]
# ## Predictions, figure, VERDICT REPORT — copy into the chat

# %%
# ===== Cell 12: predictions, figure, verdict =====
fmt = lambda x: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
O = lambda src, d: next((r[d + "_roc"]["tpr1"] for r in RES if r["source"] == src), None)
D_ = lambda src, d: next((r[d][0] for r in RES if r["source"] == src and d in r), None)
# [PAPER] Prediction log (Step 20, P20.1): Fail (0.00 vs 0.81, 0.95)
for src in ["metamath_unique", "gsm8k_human"]:
    best = max(["style_grp", "style_acct", "topic_conc"], key=lambda d: O(src, d) or 0)
    print(f"P20.1 {src}: oracle TPR@1% two-scale {fmt(O(src,'twoscale'))}; best simple baseline {best} {fmt(O(src,best))} (pass if >= two-scale - 0.05)")
print("P20.2 deployed FPR: " + " ".join(f"{d}={FPR[d][0]:.3f}" for d in DETS20) + " (pass if only two-scale <= 0.02)")
# [PAPER] Prediction log (Step 20, P20.3): Fail (0.00 vs 0.24)
print(f"P20.3 mixed padding: oracle TPR@1% style_acct {fmt(O('metamath_mixed','style_acct'))} vs two-scale {fmt(O('metamath_mixed','twoscale'))} (pass if >= +0.1)")
# [PAPER] Prediction log (Step 20, P20.4) and §5 / App. D.6: two-scale 0.31; style_acct oracle 0.00
print(f"P20.4 adaptive attacker: two-scale paired {fmt(D_('metamath_adaptive','twoscale'))} (<= 0.3) | style_acct oracle {fmt(O('metamath_adaptive','style_acct'))} (>= 0.5) | MMD oracle {fmt(O('metamath_adaptive','mmd'))}, global {fmt(O('metamath_adaptive','glob'))}")
print(f"P20.5 null-median slope vs ln A {slope:.3f} vs Gumbel beta {beta:.3f} (pass if within +/-50%)")
print(f"P20.6 breaking point: predicted {A_star:.0f}, empirical {A_emp:.0f} (pass if within a factor of 2)")
print(f"P20.7 GEV shape xi {xi0:.3f} [{XI_CI[0]:.3f}, {XI_CI[1]:.3f}] (pass if interval excludes 0 on the positive side)")
# [PAPER] Prediction log (Step 20, P20.8): Fail (1 of 4 inside band)
print("P20.8 out-of-sample: " + ", ".join(f"A={r['A']}: {'inside' if r['inside'] else 'OUTSIDE'}" for r in EVT if 'inside' in r and r['A'] != A0))
big = [r for r in EVT if 'tpr_selected' in r]; r0, r1 = big[0], big[-1]
lost = r0["tpr_selected"] - r1["tpr_selected"]; rec = r1["tpr_oracle_group"] - r1["tpr_selected"]
# [PAPER] Prediction log (Step 20, P20.9): Pass (all of it) — the true colluder set recovers all the lost power
print(f"P20.9 decay A={r0['A']}->{r1['A']}: selected TPR {r0['tpr_selected']:.2f}->{r1['tpr_selected']:.2f}; true colluder set at A={r1['A']}: {r1['tpr_oracle_group']:.2f} (recovers {rec/lost if lost > 0 else float('nan'):.2f} of the loss; pass if >= 0.5)")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6)); srcs = [r["source"] for r in RES]; xs = np.arange(len(srcs)); w = 0.14
for j, d in enumerate(DETS20): ax[0].bar(xs + (j - 2.5) * w, [O(s_, d) or 0 for s_ in srcs], w, label=f"{d} (deployed FPR {FPR[d][0]:.3f})")
ax[0].set_xticks(xs); ax[0].set_xticklabels(srcs, rotation=25, fontsize=8); ax[0].set_ylim(0, 1.02); ax[0].set_ylabel("oracle TPR at matched 1% FPR"); ax[0].legend(fontsize=6.5)
ax[1].plot([l["A"] for l in LAW], [l["med"] for l in LAW], "o-", label="null median"); ax[1].plot([l["A"] for l in LAW], [l["med_pred"] for l in LAW], "o--", label="Gumbel-predicted median")
ax[1].plot([l["A"] for l in LAW], [l["q99"] for l in LAW], "s-", label="null 99%"); ax[1].plot([l["A"] for l in LAW], [l["q99_pred"] for l in LAW], "s--", label="predicted 99%")
ax[1].plot([l["A"] for l in LAW], [l["camp_med"] for l in LAW], "^-", label="campaign median (N fixed)"); ax[1].set_xscale("log"); ax[1].set_xlabel("accounts A"); ax[1].set_ylabel("two-scale score"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/step20_figure.png", dpi=130); plt.show()
print("=" * 90); print("STEP 20 VERDICT REPORT"); print("=" * 90)
print("[FPR] " + " ".join(f"{d}={FPR[d][0]:.4f}[{FPR[d][1]:.4f},{FPR[d][2]:.4f}]" for d in DETS20))
for r in RES:
    print(f"[W] {r['source']:>19} deployed " + " ".join(f"{d}={r[d][0]:.2f}[{r[d][1]:.2f},{r[d][2]:.2f}]" for d in DETS20 if d in r))
    print("    oracle " + " ".join(f"{d}:auc={r[d+'_roc']['auc']:.2f},tpr1={r[d+'_roc']['tpr1']:.2f}" for d in DETS20))
for l in LAW: print("[L]", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in l.items()}))
print(f"[L] mu={mu:.3f} beta={beta:.3f} slope={slope:.3f} A*_pred={A_star:.0f} A*_emp={A_emp:.0f}"); print("=" * 90)
with open(f"{OUTDIR}/step20_results.json", "w") as fh: json.dump(dict(FPR=FPR, RES=RES, LAW=LAW, mu=mu, beta=beta, slope=slope, A_star=A_star, A_emp=A_emp, EVT=EVT, XI=xi0, XI_CI=XI_CI), fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
