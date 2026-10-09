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
# | Prediction log (Step 17, P17.1); App. F 'Query-level pooling' ("robust per-cell calibration fired on 28% of null weeks"); App. D.4 and §5 'Padding, and a fix that looks at directions' (first of the "four repairs that pool topic cells across accounts") | corrected query-cluster (QC) held-out false alarms, weekly | Fail (28%) | Cell 8 `FPR["qc"]` ("held-out FPR" print); Cell 12 print "P17.1" |
# | Prediction log (Step 17, P17.2) | QC paired power under mimicry (mixed padding, follow-up); AUC | Fail (0.38–0.40; AUC 0.80–0.82) | Cell 8 rows `metamath_mixed` / `metamath_followup` (`qc`, `qc_roc`); Cell 12 print "P17.2" |
# | Prediction log (Step 17, P17.3) | QC on random natural campaigns; on human GSM8K questions | Partial (0.11; 0.58) | Cell 8 rows `random` / `gsm8k_human` (`qc`); Cell 12 print "P17.3" |
# | Prediction log (Step 17, P17.4) | rolling 4-week populations (A = 1,000): held-out false alarms, two-scale; QC | Fail (24%, 69%) | Cell 10 `FPRP_CI` ("held-out FPR (position-clustered)" print); Cell 12 print "P17.4/5" (`fpr` values) |
# | Prediction log (Step 17, P17.5) | rolling populations, K = 10 (1%): power of every detector | Pass (≤ 0.08) | Cell 10 `RESP` rows K=10 ("[pop] K=10" prints); Cell 12 print "P17.4/5" |
# | §5 'Scale: selection fails before the barrier does' | colluder shares at which no detector was both calibrated and powerful | 1–2.5% | Cell 10 rolling populations, K = 10 and 25 of A = 1,000 (plausibly combined with Step 15 Part P; summary statement, not isolated) |
#
# Not reported in the paper: corrected QC power on unpadded MetaMathQA and `wild_cluster`; two-scale/MMD/global
# weekly rows of this step (Table 6 uses Step 15's run); `qcount`, `gmmd`, `spec`, `fraudar` values; QC purity;
# rolling-population power at K = 25 and for `metamath_mixed` / `random`; the cumulative-detection cell (Part L);
# the figure. Corrections (docs/corrections.md), 'Corrections' ('Query-cluster calibration and the population protocol') describes this
# script as the corrected version of Step 15's QC calibration and population protocol.

# %% [markdown]
# # Step 17 — A correctly calibrated query-cluster detector against mimicry, and realistic colluder share with rolling calibration
# **Kaggle:** GPU + Internet + `HF_TOKEN` → **Run All** (≈60–80 min; reuses the Step 11/13/14/15 caches). `QUICK=True` first.
#
# Step 15's query-cluster (QC) test standardised each topic cell with statistics from as few as 3 calibration windows (and a 1e-6 floor), which produced 64% false alarms; its realistic-share populations used a single calibration period without lagged adaptation. Both are fixed here.
#
# **Corrected QC.** A topic cell is used only if it has ≥ MINCELL queries in at least half the calibration windows. Within each cell, the two-scale score is regressed on log(cell count) across calibration windows, and residuals are standardised by median and 1.4826·MAD (floored at 0.05). The cell-volume variant uses median/MAD of cell shares with a floor of 0.002 + 0.1·median. Window statistic: max over cells; threshold from the held-out half of calibration windows. Queries are pooled across accounts, so within-account padding cannot dilute a cell.
#
# **Rolling realistic-share populations.** For each test position t (every 4th week from week 20), the population is all eligible account-weeks of weeks t−3…t (≈ 700–1,600); windows sample A = 1,000 of them. Calibration windows come from the 8 preceding 4-week blocks' account-weeks (weeks t−11…t−4), each test position whitened against a lagged reference (weeks t−7…t−4). K = 10 and 25 colluders (1% and 2.5%).
#
# | ID | Locked prediction |
# |---|---|
# | **P17.1** | Corrected QC held-out FPR ≤ 2% (weekly) |
# | **P17.2** | Corrected QC paired power on mixed padding and follow-up mimicry ≥ 0.5 and within 0.25 of its power on unpadded MetaMathQA (weekly, K = 50) |
# | **P17.3** | Corrected QC on random natural campaigns ≤ 0.05; on human GSM8K questions ≥ 0.5 (it also detects source shift) |
# | **P17.4** | Rolling populations (A = 1,000): two-scale and QC each calibrated (≤ 2%); at K = 25 (2.5%), QC power ≥ two-scale power + 0.2 |
# | **P17.5** | Rolling populations: at K = 10 (1%), every calibrated detector ≤ 0.3 |

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
SRC_EXTRA = ["wild_cluster"]
CAMO_FRAC = 0.2
if QUICK:
    WK = dict(KS=[20], CAL_N=4, LAGB=2, N_TEST=3, NULL_PER=12, REPS=2)
    POP = dict(A=60, KS=[3, 6], STEP=4, FIRST=12, N_CAL=16, N_TEST=6, REPS=3)
    MMD_WIN, N_REF_BATCH, REF_BATCH = 400, 5, 200
else:
    WK = dict(KS=[50], CAL_N=8, LAGB=4, N_TEST=24, NULL_PER=40, REPS=4)
    POP = dict(A=1000, KS=[10, 25], STEP=4, FIRST=20, N_CAL=60, N_TEST=20, REPS=12)
CACHE11 = f"{OUTDIR}/step11_wildchat_E384_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
CACHE13 = f"{OUTDIR}/step13_distill_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
CACHE14 = f"{OUTDIR}/step14_meta_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
CACHE15 = f"{OUTDIR}/step15_extra_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
N_CONV = 30000 if QUICK else None; DIMS = [384]
print(f"backend {'GPU ' + torch.cuda.get_device_name(0) if USE_GPU else 'CPU'} | QUICK={QUICK}")

# %% [markdown]
# ## Part 0 — caches: WildChat embeddings, conversation metadata, MetaMathQA sets

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
    if src in ("metamath_unique", "gsm8k_human", "metamath_template"):
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
# ## Part W — weekly populations: query-cluster detector, fair baselines, mimicry, source shift

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
# [PAPER] Prediction log (Step 17, P17.1): corrected QC weekly false alarms 28% (FPR["qc"]); App. F 'Query-level pooling': robust per-cell calibration fired on 28% of null weeks
FPR = {d: boot_ci([np.mean([fires15(r, THR[b])[d] for r in NULLW[b]]) for b in TB]) for d in DETS}
print("[weekly] held-out FPR: " + " ".join(f"{d}={FPR[d][0]:.3f}" for d in DETS))
RES, WEEKP, RAW = [], {}, {}
for K_ in WK["KS"]:
    srcs = SRC_MAIN + (SRC_EXTRA if K_ == max(WK["KS"]) else [])
    base_cache = {}
    for src in srcs:
        pw = {d: [] for d in DETS}; prec, purity, null_v, att_v = [], [], {d: [] for d in DETS}, {d: [] for d in DETS}
        for b in TB:
            acc = {d: [] for d in DETS}
            for rep in range(WK["REPS"]):
                r = np.random.default_rng(SEED + 9_000_000 + 1000 * b + 10 * rep + K_); idx, lab, vols = sub_window(b, r); A_ = int(lab.max()) + 1
                if K_ >= A_ // 2: continue
                key = (b, rep, K_)
                if key not in base_cache: base_cache[key] = add_scores(window_stats15(idx, lab, np.bincount(lab, minlength=A_), *BANK[b], REFS[b]), THR[b])
                sb = base_cache[key]; fb = fires15(sb, THR[b])
                cv = r.choice(np.array([len(g) for g in GA[b]]), K_, replace=True); N = int(cv.sum())
                Zc, Ec, Tc = campaign_points(src, N, np.random.default_rng(SEED + 17 * b + rep + K_ + len(src))); N = len(Zc)
                labc = np.repeat(np.arange(K_), cv)[:N]; keep = lab >= K_
                _, inv = np.unique(np.concatenate([labc, lab[keep]]), return_inverse=True); lc, lk = inv[:N], inv[N:]
                s = add_scores(window_stats15(idx[keep], lk, np.bincount(inv), *BANK[b], REFS[b], extra=(Zc, lc), en_extra=Ec, top_extra=Tc), THR[b]); f = fires15(s, THR[b])
                for d in DETS:
                    acc[d].append(float(f[d] and not fb[d])); null_v[d].append(sb[d]); att_v[d].append(s[d])
                if f["twoscale"]: g = s["grp"]; prec.append(float(np.mean(g < K_)))
                if f["qc"] and s["qc_cell"] >= 0:
                    m_ = s["top"] == s["qc_cell"]; purity.append(float(np.mean(s["labs"][m_] < K_)))
            for d in DETS:
                if acc[d]: pw[d].append(np.mean(acc[d]))
        WEEKP[(src, K_)] = {d: pw[d] for d in DETS}
        row = dict(source=src, K=K_, ts_precision=float(np.mean(prec)) if prec else float("nan"), qc_purity=float(np.mean(purity)) if purity else float("nan"))
        for d in DETS:
            if pw[d]: m, lo, hi = boot_ci(pw[d]); row[d] = m; row[d + "_ci"] = (lo, hi)
            # [PAPER] Prediction log (Step 17, P17.2): QC AUC under mimicry 0.80–0.82 (row["qc_roc"]["auc"])
            row[d + "_roc"] = roc(null_v[d], att_v[d])
        RES.append(row)
        print(f"[weekly] {src:>19} K={K_:3d} | " + " ".join(f"{d}={row.get(d, float('nan')):.2f}" for d in DETS) + f" | prec2s={row['ts_precision']:.2f} QCpurity={row['qc_purity']:.2f} | TPR@1% 2s/qc/mmd/gmmd = "
              + "/".join(f"{row[d+'_roc']['tpr1']:.2f}" for d in ["twoscale", "qc", "mmd", "gmmd"]) + f"  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Part P — realistic colluder share (rolling)

# %%
# ===== Cell 10: realistic colluder share — rolling 4-week populations of account-weeks, rolling calibration, lagged reference =====
t0 = time.time(); RESP = []; FPRP = {d: [] for d in DETS}; HITP = {}
positions = [t for t in range(POP["FIRST"], n_b, POP["STEP"]) if t - 11 >= 0]
def aw_pool(w0, w1): return [g for w in range(max(0, w0), w1 + 1) for g in GA.get(w, [])]
def pop_window(pool, A_, r):
    pick = r.choice(len(pool), min(A_, len(pool)), replace=False); parts = [pool[i] for i in pick]
    vols = np.array([len(p) for p in parts]); return np.concatenate(parts), np.repeat(np.arange(len(parts)), vols), vols
for t in positions:
    pool_t = aw_pool(t - 3, t); A_ = min(POP["A"], int(0.7 * len(pool_t)))
    if A_ < 4 * max(POP["KS"]): continue
    refw = np.concatenate([np.concatenate(g) if False else g for g in aw_pool(t - 7, t - 4)]); refw = np.random.default_rng(t).choice(refw, min(100000, len(refw)), replace=False)
    BPt = (make_bank_aug(Z[refw], H_COARSE, SEED + 7), make_bank_aug(Z[refw], H_FINE, SEED + 8))
    rb = np.random.default_rng(t + 1).choice(refw, N_REF_BATCH * REF_BATCH, replace=len(refw) < N_REF_BATCH * REF_BATCH).reshape(N_REF_BATCH, REF_BATCH)
    REFt = [torch.as_tensor(EN[x].astype(np.float32), device=DEV) if USE_GPU else EN[x].astype(np.float32) for x in rb]
    cal = []
    for j, (w0, w1) in enumerate([(t - 11, t - 8), (t - 7, t - 4)]):
        pc = aw_pool(w0, w1)
        for k in range(POP["N_CAL"] // 2):
            idx, lab, vols = pop_window(pc, min(A_, int(0.7 * len(pc))), np.random.default_rng(SEED + 1000 * t + 100 * j + k)); cal.append(window_stats15(idx, lab, np.bincount(lab), *BPt, REFt))
    TP = calibrate15(cal)
    for k in range(POP["N_TEST"]):
        idx, lab, vols = pop_window(pool_t, A_, np.random.default_rng(SEED + 7000 * t + k)); sn = add_scores(window_stats15(idx, lab, np.bincount(lab), *BPt, REFt), TP)
        f = fires15(sn, TP)
        for d in DETS: FPRP[d].append((t, float(f[d])))
    for K_ in POP["KS"]:
        for src in ["metamath_unique", "metamath_mixed", "random"]:
            for rep in range(POP["REPS"]):
                r = np.random.default_rng(SEED + 9000 * t + 31 * K_ + rep); idx, lab, vols = pop_window(pool_t, A_, r)
                sb = add_scores(window_stats15(idx, lab, np.bincount(lab), *BPt, REFt), TP); fb = fires15(sb, TP)
                cv = r.choice(vols, K_, replace=True); N = int(cv.sum()); Zc, Ec, Tc = campaign_points(src, N, r); N = len(Zc)
                labc = np.repeat(np.arange(K_), cv)[:N]; keep = lab >= K_; _, inv = np.unique(np.concatenate([labc, lab[keep]]), return_inverse=True)
                s_ = add_scores(window_stats15(idx[keep], inv[N:], np.bincount(inv), *BPt, REFt, extra=(Zc, inv[:N]), en_extra=Ec, top_extra=Tc), TP); f = fires15(s_, TP)
                for d in DETS: HITP.setdefault((K_, src, d), []).append((t, float(f[d] and not fb[d])))
    print(f"[pop] position week {t}: population {len(pool_t)} account-weeks, A={A_} ({time.time()-t0:.0f}s)")
def by_position(lst):
    pos = sorted(set(p for p, _ in lst)); return [np.mean([v for p, v in lst if p == q]) for q in pos]
# [PAPER] Prediction log (Step 17, P17.4): rolling populations, two-scale and QC false alarms 24%, 69%
FPRP_CI = {d: boot_ci(by_position(FPRP[d])) for d in DETS}
print("[pop] held-out FPR (position-clustered): " + " ".join(f"{d}={FPRP_CI[d][0]:.3f}" for d in DETS))
for K_ in POP["KS"]:
    for src in ["metamath_unique", "metamath_mixed", "random"]:
        row = dict(K=K_, A=POP["A"], source=src, **{d: boot_ci(by_position(HITP[(K_, src, d)])) for d in DETS}, **{f"{d}_fpr": FPRP_CI[d][0] for d in DETS})
        # [PAPER] Prediction log (Step 17, P17.5): rolling populations at K = 10 (1%), every detector ≤ 0.08; §5 'Scale': no detector both calibrated and powerful at 1–2.5% (summary, see header)
        RESP.append(row)
        print(f"[pop] K={K_} ({100*K_/POP['A']:.1f}%) {src:>16}: " + " ".join(f"{d}={row[d][0]:.2f}" for d in DETS))

# %% [markdown]
# ## Part L — cumulative detection

# %%
# ===== Cell 11: cumulative detection over a campaign's lifetime =====
LIFE = {}
def cumulative(pweeks, w):
    p = np.array(pweeks); starts = range(0, len(p) - w + 1)
    return float(np.mean([1 - np.prod(1 - p[s0:s0 + w]) for s0 in starts])) if len(p) >= w else float("nan")
kk = max(WK["KS"])
for src in ["metamath_unique", "metamath_mixed", "metamath_followup"]:
    for d in ["twoscale", "qc"]:
        pw = WEEKP.get((src, kk), {}).get(d, [])
        LIFE[(src, d)] = {w: cumulative(pw, w) for w in [1, 4, 8, 12]}
    either = [1 - (1 - a) * (1 - b) for a, b in zip(WEEKP[(src, kk)]["twoscale"], WEEKP[(src, kk)]["qc"])]
    LIFE[(src, "either")] = {w: cumulative(either, w) for w in [1, 4, 8, 12]}
    print(f"[L] {src:>19}: P(caught within w weeks) two-scale " + " ".join(f"w{w}={LIFE[(src,'twoscale')][w]:.2f}" for w in [1, 4, 12])
          + " | QC " + " ".join(f"w{w}={LIFE[(src,'qc')][w]:.2f}" for w in [1, 4, 12]) + " | either " + " ".join(f"w{w}={LIFE[(src,'either')][w]:.2f}" for w in [1, 4, 12]))
for d in ["twoscale", "qc"]:
    f1 = FPR[d][0]; print(f"[L] {d}: P(at least one false alarm in 12 weeks) = {1 - (1 - f1) ** 12:.3f} (weekly FPR {f1:.4f})")
f_either = 1 - (1 - FPR["twoscale"][0]) * (1 - FPR["qc"][0]); print(f"[L] either: P(>=1 false alarm in 12 weeks) = {1 - (1 - f_either) ** 12:.3f}")

# %% [markdown]
# ## Predictions, figure, VERDICT REPORT — copy into the chat

# %%
# ===== Cell 12: predictions, figure, verdict =====
fmt = lambda x: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
g = lambda src, K_, d: next((r.get(d) for r in RES if r["source"] == src and r["K"] == K_), None)
gr = lambda src, K_, d, k="tpr1": next((r[d + "_roc"][k] for r in RES if r["source"] == src and r["K"] == K_), None)
K1 = max(WK["KS"])
# [PAPER] Prediction log (Step 17, P17.1): Fail (28%)
print(f"P17.1 corrected QC weekly FPR {FPR['qc'][0]:.4f} [{FPR['qc'][1]:.4f}, {FPR['qc'][2]:.4f}] (pass if <= 0.02); qcount {FPR['qcount'][0]:.4f}; two-scale {FPR['twoscale'][0]:.4f}")
u, m_, f_ = g("metamath_unique", K1, "qc"), g("metamath_mixed", K1, "qc"), g("metamath_followup", K1, "qc")
# [PAPER] Prediction log (Step 17, P17.2): Fail (0.38–0.40; AUC 0.80–0.82)
print(f"P17.2 QC power: unique {fmt(u)}, mixed {fmt(m_)}, follow-up {fmt(f_)} (pass if both >= 0.5 and within 0.25 of unique); AUC unique/mixed/follow-up "
      f"{fmt(gr('metamath_unique', K1, 'qc', 'auc'))}/{fmt(gr('metamath_mixed', K1, 'qc', 'auc'))}/{fmt(gr('metamath_followup', K1, 'qc', 'auc'))}")
# [PAPER] Prediction log (Step 17, P17.3): Partial (0.11; 0.58)
print(f"P17.3 QC: random {fmt(g('random', K1, 'qc'))} (<= 0.05), human GSM8K {fmt(g('gsm8k_human', K1, 'qc'))} (>= 0.5)")
# [PAPER] Prediction log (Step 17, P17.4): Fail (24%, 69%); P17.5: Pass (≤ 0.08)
for r in RESP:
    if r["source"] == "metamath_unique": print(f"P17.4/5 rolling A={r['A']} K={r['K']}: two-scale {r['twoscale'][0]:.2f} (fpr {r['twoscale_fpr']:.3f}), QC {r['qc'][0]:.2f} (fpr {r['qc_fpr']:.3f}), qcount {r['qcount'][0]:.2f} (fpr {r['qcount_fpr']:.3f}), MMD {r['mmd'][0]:.2f} (fpr {r['mmd_fpr']:.3f}), global {r['glob'][0]:.2f} (fpr {r['glob_fpr']:.3f})")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6)); srcs = [s_ for s_ in SRC_MAIN + SRC_EXTRA if g(s_, K1, "twoscale") is not None]; xs = np.arange(len(srcs)); w = 0.2
for j, d in enumerate(["twoscale", "qc", "qcount", "mmd"]):
    ax[0].bar(xs + (j - 1.5) * w, [g(s_, K1, d) or 0 for s_ in srcs], w, label=f"{d} (FPR {FPR[d][0]:.3f})")
ax[0].set_xticks(xs); ax[0].set_xticklabels(srcs, rotation=25, fontsize=8); ax[0].set_ylim(0, 1.02); ax[0].set_ylabel("paired detection (calibrated)"); ax[0].legend(fontsize=7); ax[0].set_title(f"weekly, K={K1}")
lab_ = [f"K={r['K']}\\n{r['source']}" for r in RESP]; xs = np.arange(len(RESP))
for j, d in enumerate(["twoscale", "qc", "qcount", "mmd"]): ax[1].bar(xs + (j - 1.5) * w, [r[d][0] for r in RESP], w, label=d)
ax[1].set_xticks(xs); ax[1].set_xticklabels(lab_, fontsize=7); ax[1].set_ylim(0, 1.02); ax[1].set_title(f"rolling 4-week populations, A={POP['A']}"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/step17_figure.png", dpi=130); plt.show()
print("=" * 90); print("STEP 17 VERDICT REPORT"); print("=" * 90)
print("[FPR weekly] " + " ".join(f"{d}={FPR[d][0]:.4f}[{FPR[d][1]:.4f},{FPR[d][2]:.4f}]" for d in DETS))
for r in RES:
    print(f"[W] {r['source']:>19} K={r['K']:3d} prec2s={fmt(r['ts_precision'])} QCpur={fmt(r['qc_purity'])} | " + " ".join(f"{d}={fmt(r.get(d))}[{fmt(r.get(d+'_ci',(None,None))[0])},{fmt(r.get(d+'_ci',(None,None))[1])}]" for d in DETS))
    print("    ROC: " + " ".join(f"{d}:auc={r[d+'_roc']['auc']:.2f},tpr1={r[d+'_roc']['tpr1']:.2f}" for d in DETS))
print("[pop FPR] " + " ".join(f"{d}={FPRP_CI[d][0]:.3f}[{FPRP_CI[d][1]:.3f},{FPRP_CI[d][2]:.3f}]" for d in DETS))
# [PAPER] Prediction log (Step 17, P17.5): every detector ≤ 0.08 at K = 10 (1%)
for r in RESP: print(f"[P] K={r['K']} {r['source']:>16} | " + " ".join(f"{d}={r[d][0]:.2f}[{r[d][1]:.2f},{r[d][2]:.2f}]" for d in DETS))
for k, v in LIFE.items(): print(f"[L] {k}: " + " ".join(f"w{w}={x:.2f}" for w, x in v.items()))
print("=" * 90)
with open(f"{OUTDIR}/step17_results.json", "w") as fh: json.dump(dict(FPR=FPR, RES=RES, RESP=RESP, LIFE={str(k): v for k, v in LIFE.items()}), fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
