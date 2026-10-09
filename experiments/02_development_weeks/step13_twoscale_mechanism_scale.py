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
# | §5 'Coordinated, dataset-like campaigns are catchable'; App. D.2 | two-scale flagged-group precision (share of the group that are colluders) | 0.94–1.00 (§5); 1.00 at K = 50, 0.94–0.96 at K = 10 (App. D.2) | Cell 6 `run_block`, `prec` list -> `ts_precision`; printed "two-scale group precision" (weekly, `metamath_unique`/`metamath_rephrase`, K = 10/50) and verdict `ts_prec=` |
# | §5 'Coordinated, dataset-like campaigns are catchable'; App. D.2 | two-scale flagged-group recall (share of colluders in the group) | 12–31% | Cell 6 `run_block`, `rec` list -> `ts_recall`; printed "recall" and verdict `ts_rec=` (range plausibly over weekly sources and K) |
# | Prediction log (Step 13, P13.4) | group precision; recall (metamath_unique, weekly K = 50) | Pass (1.00; recall 0.17) | Cell 8, print "P13.4" |
# | Table 7 rows "MetaMathQA, one prompt per source problem" / "four variants per source problem"; App. D.3; Prediction log (Step 13, P13.1) | weekly paired detection, K = 50: two-scale / global / MMD | 0.88 / 0.68 / 0.57 (one prompt); 0.80 / 0.71 / 0.60 (four variants); App. D.3 "from 0.88 to 0.80"; Prediction log "Fail (0.88 vs 0.80)" | Cell 7 Part A (weekly), rows `metamath_unique` / `metamath_rephrase` K=50, columns `twoscale`, `glob`, `mmd`; Cell 8 print "P13.1" |
# | Table 7 rows "WildChat broad cluster, random prompts" / "farthest-point deduplicated"; Prediction log (Step 13, P13.2) | weekly paired detection, K = 50: two-scale / global / MMD | 0.01 / 0.01 / 0.07 (random); 0.04 / 0.02 / 0.07 (deduplicated); Prediction log "Fail (0.04 vs 0.01)" | Cell 7 Part A, rows `wild_cluster` / `wild_cluster_dedup` K=50; Cell 8 print "P13.2" |
# | Prediction log (Step 13, P13.3) | two-scale held-out false alarms, weekly; 4-week | Partial (0.6%; 9.8%) | Cell 6 `FPR[...]` ("held-out FPR" print); Cell 8 print "P13.3" |
# | Prediction log (Step 13, P13.5) | 4-week populations, K = 200: two-scale on metamath_unique; on random | Pass (0.86; 0.00) | Cell 7 Part B rows; Cell 8 print "P13.5" |
# | App. D.5 'Smaller colluder shares'; Prediction log (Step 13, P13.6); App. L item 4; Corrections (docs/corrections.md), 'Corrections' ('Sharding') | paired attacker-aware sharding, one colluder per 100-account shard, per-shard power of every detector | "defeated every detector in all 352 shard tests" (App. D.5); "Pass (0.00 for all)" (Prediction log); "Sharding defeats every detector" (App. L) | Cell 6 `run_block(..., shard=True)` shard branch, printed "sharded (one colluder per 100-account shard)" with `n=`; Cell 8 print "P13.6" (352 plausibly the total of `n` over the two sources) |
#
# Not reported in the paper: the `magicoder` and `random` weekly rows; the K = 10 detection rows; the `spec`, `hc`,
# `acct`, `fraudar` and `vol` detector columns; 4-week K = 50 rows and the 4-week `metamath_rephrase` /
# `wild_cluster_dedup` rows; held-out FPRs of detectors other than two-scale; bootstrap intervals; the figure.
# The paper reports this step's mechanism result as "Not deduplication" (Prediction log); the deduplication claim in the
# header below was withdrawn (Corrections (docs/corrections.md), 'Corrections', 'Mechanism claims').

# %% [markdown]
# # Step 13 — Why does the two-scale test work, does it identify colluders, and does it scale?
# **Kaggle:** GPU + Internet + `HF_TOKEN` → **Run All** (≈50–70 min; reuses the Step 11 WildChat cache). `QUICK=True` first.
#
# Step 12: under volume-matched colluders and whole weekly populations, the **two-scale test** (log S̄ at coarse h = 4 minus log S̄ at fine h = 0.5, on the group carried by the top eigenvector) was the only calibrated detector (0.7%) and caught MetaMathQA campaigns 93% of the time (MMD: 60%). Three questions remain.
#
# **1. Mechanism (deduplication vs generation style), on real data.**
# * `metamath_unique`: one query per MetaMathQA source problem (deduplicated generated set).
# * `metamath_rephrase`: four distinct variants (rephrasings, self-verification, backward questions) per source problem — same generator and style, but clumpy at fine scale.
# * `wild_cluster` vs `wild_cluster_dedup`: natural WildChat prompts from one broad cluster, sampled at random vs by farthest-point coverage (natural text, deduplicated).
# * `magicoder`, `random` as references.
#
# **2. Identification.** Precision and recall of the flagged two-scale group (fraction of the group that are colluders; fraction of colluders in the group).
#
# **3. Scale.** Part B repeats the test on **4-week populations** (accounts = IP × 4-week block, ≥ 10 queries), where K = 200 is feasible, and runs a **paired attacker-aware sharding** test: 100-account shards, one colluder per shard, each shard scored with and without its colluder at a per-shard α = 1%.
#
# | ID | Locked prediction |
# |---|---|
# | **P13.1** | Two-scale power: `metamath_unique` ≥ `metamath_rephrase` + 0.3 (weekly, K = 50) — deduplication drives the signal |
# | **P13.2** | Two-scale power: `wild_cluster_dedup` ≥ `wild_cluster` + 0.2 (weekly, K = 50) — deduplication works on natural text too |
# | **P13.3** | Two-scale held-out FPR ≤ 2% in both weekly and 4-week populations |
# | **P13.4** | Two-scale group precision ≥ 0.5 when it fires on `metamath_unique` at K = 50 |
# | **P13.5** | 4-week populations, K = 200: two-scale power ≥ 0.8 on `metamath_unique`; ≤ 0.05 on `random` |
# | **P13.6** | Paired sharding (single colluder per shard): per-shard two-scale power ≤ 0.2 (exploratory) |

# %%
# ===== Cell 1: configuration =====
import math, json, time, os, gc
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
N_TOPICS, N_REF_BATCH, REF_BATCH, MMD_WIN = 100, 20, 500, 2000
DETS = ["spec", "hc", "acct", "glob", "mmd", "fraudar", "vol", "twoscale"]
SRC_A = ["metamath_unique", "metamath_rephrase", "magicoder", "wild_cluster", "wild_cluster_dedup", "random"]
SRC_B = ["metamath_unique", "metamath_rephrase", "wild_cluster_dedup", "random"]
if QUICK:
    PART_A = dict(BLOCK=1, KS=[10, 50], CAL_N=4, LAGB=2, N_TEST=3, NULL_PER=12, REPS=2)
    PART_B = dict(BLOCK=4, KS=[20, 50], CAL_N=2, LAGB=1, N_TEST=2, NULL_PER=12, REPS=2)
    MMD_WIN, N_REF_BATCH, REF_BATCH = 400, 5, 200
else:
    PART_A = dict(BLOCK=1, KS=[10, 50], CAL_N=8, LAGB=4, N_TEST=24, NULL_PER=40, REPS=4)
    PART_B = dict(BLOCK=4, KS=[50, 200], CAL_N=4, LAGB=1, N_TEST=10, NULL_PER=60, REPS=4)
SHARD = 100
CACHE11 = f"{OUTDIR}/step11_wildchat_E384_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
CACHE13 = f"{OUTDIR}/step13_distill_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
N_CONV = 30000 if QUICK else None; DIMS = [384]
print(f"backend {'GPU ' + torch.cuda.get_device_name(0) if USE_GPU else 'CPU'} | QUICK={QUICK}")

# %% [markdown]
# ## Part 0 — WildChat 384-d embeddings (Step 11 cache)

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
# ===== Cell 3: MetaMathQA grouped by source problem (unique vs rephrased) + Magicoder, MiniLM, cached =====
if os.path.exists(CACHE13):
    _c = np.load(CACHE13); DIST = {k[2:]: _c[k] for k in _c.files if k.startswith("D_")}; MM_GROUP = _c["MM_GROUP"]; print({k: v.shape for k, v in DIST.items()})
else:
    t0 = time.time(); DIST = {}
    if SYNTHETIC_TEST:
        r = np.random.default_rng(3); c = r.standard_normal(384) * 2
        base = c + r.standard_normal((1500, 384)) * 0.7
        DIST["metamath_unique"] = base.astype(np.float16)
        reph = np.repeat(base[:600], 4, axis=0) + r.standard_normal((2400, 384)) * 0.05; DIST["metamath_rephrase"] = reph.astype(np.float16)
        MM_GROUP = np.repeat(np.arange(600), 4)
        DIST["magicoder"] = (r.standard_normal(384) * 2 + r.standard_normal((2000, 384)) * 0.7).astype(np.float16)
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
        DIST["metamath_unique"] = emb(uniq); DIST["metamath_rephrase"] = emb(reph); MM_GROUP = np.array(gid)
        print(f"MetaMathQA: {len(uniq)} source problems (unique set), {len(multi)} problems x 4 variants (rephrase set) ({time.time()-t0:.0f}s)")
        mg = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", token=tok)
        idx = np.random.default_rng(SEED).choice(len(mg), min(25000, len(mg)), replace=False)
        DIST["magicoder"] = emb(list(dict.fromkeys(str(mg[int(i)]["problem"])[:1000] for i in idx)))
        print(f"Magicoder: {len(DIST['magicoder'])} prompts ({time.time()-t0:.0f}s)")
    np.savez(CACHE13, MM_GROUP=MM_GROUP, **{f"D_{k}": v for k, v in DIST.items()})

# %% [markdown]
# ## Part 1 — projection, topics, natural source cluster

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


# %% [markdown]
# ## Part 2 — pipeline

# %%
# ===== Cell 6: the full pipeline for one block length (weekly or 4-week), as a function =====
def campaign_points(src, n, r):
    if src == "random": ix = r.choice(pool_idx, n, replace=False); return Z[ix], EN[ix].astype(np.float32), TOPIC[ix]
    if src == "wild_cluster": ix = r.choice(WILD, min(n, len(WILD)), replace=n > len(WILD)); return Z[ix], EN[ix].astype(np.float32), TOPIC[ix]
    if src == "wild_cluster_dedup":
        start = int(r.integers(0, max(1, len(WILD_FPS) - n))); ix = WILD_FPS[start:start + n] if n <= len(WILD_FPS) else WILD_FPS
        return Z[ix], EN[ix].astype(np.float32), TOPIC[ix]
    D = DIST[src]
    if src == "metamath_rephrase":                                     # whole groups of 4 variants of the same source problem
        g = r.permutation(int(MM_GROUP.max()) + 1)[: int(math.ceil(n / 4))]; ix = np.flatnonzero(np.isin(MM_GROUP, g))[:n]
    else:
        ix = r.choice(len(D), min(n, len(D)), replace=n > len(D))
    return DZ[src][ix], D[ix].astype(np.float32), km.predict(DZ[src][ix]).astype(np.int32)
def hc_stat(x, ns):
    M_ = len(ns); p = np.sort((1 + M_ - np.searchsorted(ns, x, side="left")) / (M_ + 1)); A_ = len(p); k = max(1, A_ // 2)
    i = np.arange(1, k + 1); pp = p[:k]; return float(np.max(np.sqrt(A_) * (i / A_ - pp) / np.sqrt(pp * (1 - pp))))
def calibrate(recs, alpha):
    ns = np.sort(np.concatenate([r["r2"] for r in recs])); T = {d: float(np.quantile([r[d] for r in recs], 1 - alpha)) for d in DETS if d != "hc"}
    T["hc"] = float(np.quantile([hc_stat(r["r2"], ns) for r in recs], 1 - alpha)); T["ns"] = ns; return T
def fires(s, T):
    f = {d: s[d] > T[d] for d in DETS if d != "hc"}; f["hc"] = hc_stat(s["r2"], T["ns"]) > T["hc"]; return f
def boot_ci(x, B=1000, rng=np.random.default_rng(7)):
    x = np.array(x); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]; return float(x.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))

def run_block(P, sources, label, shard=False):
    t0 = time.time(); BLK = P["BLOCK"]; blk = (WEEK // BLK).astype(np.int32); n_b = int(blk.max()) + 1
    order = np.lexsort((ACCT, blk)); keys = blk[order].astype(np.int64) * n_acct + ACCT[order]
    GA = {b: [] for b in range(n_b)}; rg = np.random.default_rng(SEED + BLK)
    for g in np.split(order, np.flatnonzero(np.diff(keys)) + 1):
        a = int(ACCT[g[0]]); b = int(blk[g[0]])
        if is_pool[a] or len(g) < MIN_PER_ACCT: continue
        if len(g) > VOL_CAP * BLK: g = rg.choice(g, VOL_CAP * BLK, replace=False)
        GA[b].append(g)
    ok = [b for b in range(P["CAL_N"], n_b) if len(GA[b]) >= 40 and all(len(GA[v]) >= 20 for v in range(b - P["CAL_N"], b))]
    TB = [int(ok[i]) for i in np.linspace(0, len(ok) - 1, min(P["N_TEST"], len(ok))).astype(int)] if ok else []
    print(f"[{label}] accounts per block: " + ", ".join(f"{b}:{len(GA[b])}" for b in range(n_b) if GA[b]) + f"\n[{label}] test blocks {TB}")
    def sub_window(b, r, frac=SUB_FRAC, k=None):
        k = k or max(20, int(frac * len(GA[b]))); pick = r.choice(len(GA[b]), min(k, len(GA[b])), replace=False); parts = [GA[b][i] for i in pick]
        vols = np.array([len(p) for p in parts]); return np.concatenate(parts), np.repeat(np.arange(len(parts)), vols), vols
    NEED = sorted(set(v for b in TB for v in range(b - P["CAL_N"], b + 1))); BANK, REFS, NULLW = {}, {}, {}
    for b in NEED:
        prev = [np.concatenate(GA[v]) for v in range(max(0, b - P["LAGB"]), b) if GA[v]]
        ref = np.concatenate(prev) if prev else np.concatenate(GA[b]); ref = np.random.default_rng(b).choice(ref, min(100000, len(ref)), replace=False)
        BANK[b] = (make_bank_aug(Z[ref], H_COARSE, SEED + 7), make_bank_aug(Z[ref], H_FINE, SEED + 8))
        rb = np.random.default_rng(b + 1).choice(ref, N_REF_BATCH * REF_BATCH, replace=len(ref) < N_REF_BATCH * REF_BATCH).reshape(N_REF_BATCH, REF_BATCH)
        REFS[b] = [torch.as_tensor(EN[x].astype(np.float32), device=DEV) if USE_GPU else EN[x].astype(np.float32) for x in rb]
        NULLW[b] = []
        for t in range(P["NULL_PER"]):
            idx, lab, vols = sub_window(b, np.random.default_rng(SEED + 100 * b + t)); NULLW[b].append(window_stats(idx, lab, np.bincount(lab), *BANK[b], REFS[b]))
    print(f"[{label}] banks and null windows for {len(NEED)} blocks ({time.time()-t0:.0f}s)")
    THR = {b: calibrate([r for v in range(b - P["CAL_N"], b) for r in NULLW[v]], ALPHA) for b in TB}
    # [PAPER] Prediction log (Step 13, P13.3): two-scale held-out false alarms, weekly 0.6%; 4-week 9.8%
    FPR = {d: boot_ci([np.mean([fires(r, THR[b])[d] for r in NULLW[b]]) for b in TB]) for d in DETS}
    print(f"[{label}] held-out FPR: " + " ".join(f"{d}={FPR[d][0]:.3f}" for d in DETS))
    rows = []
    for src in sources:
        for K_ in P["KS"]:
            pw = {d: [] for d in DETS}; prec, rec, Ns = [], [], []
            for b in TB:
                acc = {d: [] for d in DETS}
                for rep in range(P["REPS"]):
                    r = np.random.default_rng(SEED + 9_000_000 + 1000 * b + 10 * rep + K_); idx, lab, vols = sub_window(b, r); A_ = int(lab.max()) + 1
                    if K_ >= A_ // 2: continue
                    base = fires(window_stats(idx, lab, np.bincount(lab, minlength=A_), *BANK[b], REFS[b]), THR[b])
                    cv = r.choice(np.array([len(g) for g in GA[b]]), K_, replace=True); N = int(cv.sum()); Ns.append(N)
                    Zc, Ec, Tc = campaign_points(src, N, np.random.default_rng(SEED + 17 * b + rep + K_ + sources.index(src))); N = len(Zc)
                    labc = np.repeat(np.arange(K_), cv)[:N]; keep = lab >= K_
                    _, inv = np.unique(np.concatenate([labc, lab[keep]]), return_inverse=True); lc, lk = inv[:N], inv[N:]
                    s = window_stats(idx[keep], lk, np.bincount(inv), *BANK[b], REFS[b], extra=(Zc, lc), en_extra=Ec, top_extra=Tc); f = fires(s, THR[b])
                    for d in DETS: acc[d].append(float(f[d] and not base[d]))
                    # [PAPER] §5 / App. D.2: two-scale group precision 0.94–1.00 (1.00 at K = 50, 0.94–0.96 at K = 10), recall 12–31%
                    if f["twoscale"]:
                        g = s["grp"]; nc = int(lc.max()) + 1; prec.append(float(np.mean(g < nc))); rec.append(float(np.sum(g < nc) / nc))
                for d in DETS:
                    if acc[d]: pw[d].append(np.mean(acc[d]))
            row = dict(part=label, source=src, K=K_, N_mean=float(np.mean(Ns)) if Ns else 0.0,
                       ts_precision=float(np.mean(prec)) if prec else float("nan"), ts_recall=float(np.mean(rec)) if rec else float("nan"))
            for d in DETS:
                if pw[d]: m, lo, hi = boot_ci(pw[d]); row[d] = m; row[d + "_ci"] = (lo, hi)
            rows.append(row)
            # [PAPER] Table 7 (weekly, K = 50; cols Two-scale / Global / MMD): one prompt 0.88 / 0.68 / 0.57; four variants 0.80 / 0.71 / 0.60; WildChat random 0.01 / 0.01 / 0.07; deduplicated 0.04 / 0.02 / 0.07
            print(f"[{label}] {src:>19} K={K_:4d} N~{row['N_mean']:7.0f} | " + " ".join(f"{d}={row.get(d, float('nan')):.2f}" for d in DETS)
                  + f" | two-scale group precision {row['ts_precision']:.2f} recall {row['ts_recall']:.2f}  ({time.time()-t0:.0f}s)")
    shard_rows = []
    if shard:
        for src in ["metamath_unique", "random"]:
            per_shard = {d: [] for d in DETS}
            for b in TB:
                nulls = []
                for v in range(b - P["CAL_N"], b):
                    for t in range(P["NULL_PER"]):
                        idx, lab, vols = sub_window(v, np.random.default_rng(SEED + 55 * v + t), k=SHARD); nulls.append(window_stats(idx, lab, np.bincount(lab), *BANK[b], REFS[b]))
                Ts = calibrate(nulls, ALPHA); n_sh = max(1, len(GA[b]) // SHARD)
                for rep in range(P["REPS"]):
                    for sh in range(n_sh):
                        r = np.random.default_rng(SEED + 77_000 * b + 100 * rep + sh); idx, lab, vols = sub_window(b, r, k=SHARD)
                        base = fires(window_stats(idx, lab, np.bincount(lab), *BANK[b], REFS[b]), Ts)
                        cv = int(r.choice(vols)); Zc, Ec, Tc = campaign_points(src, cv, r); cv = len(Zc); keep = lab >= 1
                        _, inv = np.unique(np.concatenate([np.zeros(cv, int), lab[keep]]), return_inverse=True)
                        s = window_stats(idx[keep], inv[cv:], np.bincount(inv), *BANK[b], REFS[b], extra=(Zc, inv[:cv]), en_extra=Ec, top_extra=Tc); f = fires(s, Ts)
                        for d in DETS: per_shard[d].append(float(f[d] and not base[d]))
            row = dict(part=label + "-shard", source=src, **{d: float(np.mean(v)) for d, v in per_shard.items()}, n=len(per_shard["spec"]))
            shard_rows.append(row)
            # [PAPER] App. D.5: sharding "defeated every detector in all 352 shard tests"; Prediction log (Step 13, P13.6): Pass (0.00 for all)
            print(f"[{label}] sharded (one colluder per {SHARD}-account shard), per-shard paired power at alpha={ALPHA}, {src}: " + " ".join(f"{d}={row[d]:.2f}" for d in DETS) + f" (n={row['n']})")
    return dict(FPR=FPR, rows=rows, shard=shard_rows, TB=TB)


# %% [markdown]
# ## Part 3 — run weekly and 4-week

# %%
# ===== Cell 7: Part A (weekly) and Part B (4-week populations + sharding) =====
RES_A = run_block(PART_A, SRC_A, "weekly")
RES_B = run_block(PART_B, SRC_B, "4-week", shard=True)

# %% [markdown]
# ## Predictions, figure, VERDICT REPORT — copy into the chat

# %%
# ===== Cell 8: predictions, figure, verdict =====
fmt = lambda x: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
def get(res, src, K_, key="twoscale"):
    return next((r.get(key) for r in res["rows"] if r["source"] == src and r["K"] == K_), None)
kA, kB = max(PART_A["KS"]), max(PART_B["KS"])
u, rp = get(RES_A, "metamath_unique", kA), get(RES_A, "metamath_rephrase", kA)
# [PAPER] Prediction log (Step 13, P13.1): Fail (0.88 vs 0.80); also Table 7 and App. D.3 ("from 0.88 to 0.80")
print(f"P13.1 weekly K={kA}: two-scale unique {fmt(u)} vs rephrase {fmt(rp)} (pass if diff >= 0.3)")
wc, wd = get(RES_A, "wild_cluster", kA), get(RES_A, "wild_cluster_dedup", kA)
# [PAPER] Prediction log (Step 13, P13.2): Fail (0.04 vs 0.01); also Table 7 (two-scale col)
print(f"P13.2 weekly K={kA}: two-scale natural dedup {fmt(wd)} vs natural raw {fmt(wc)} (pass if diff >= 0.2)")
# [PAPER] Prediction log (Step 13, P13.3): Partial (0.6%; 9.8%)
print(f"P13.3 two-scale FPR: weekly {fmt(RES_A['FPR']['twoscale'][0])} [{fmt(RES_A['FPR']['twoscale'][1])}, {fmt(RES_A['FPR']['twoscale'][2])}], 4-week {fmt(RES_B['FPR']['twoscale'][0])} [{fmt(RES_B['FPR']['twoscale'][1])}, {fmt(RES_B['FPR']['twoscale'][2])}]")
# [PAPER] Prediction log (Step 13, P13.4): Pass (1.00; recall 0.17)
print(f"P13.4 two-scale group precision/recall (metamath_unique, weekly K={kA}): {fmt(get(RES_A, 'metamath_unique', kA, 'ts_precision'))} / {fmt(get(RES_A, 'metamath_unique', kA, 'ts_recall'))}")
# [PAPER] Prediction log (Step 13, P13.5): Pass (0.86; 0.00)
print(f"P13.5 4-week K={kB}: metamath_unique {fmt(get(RES_B, 'metamath_unique', kB))}, random {fmt(get(RES_B, 'random', kB))}")
# [PAPER] Prediction log (Step 13, P13.6): Pass (0.00 for all); App. D.5 (all 352 shard tests)
for r in RES_B["shard"]: print(f"P13.6 sharded {r['source']}: per-shard two-scale {fmt(r['twoscale'])}, per-account {fmt(r['acct'])}, spectral {fmt(r['spec'])}")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
for a_, res, K_, srcs in [(ax[0], RES_A, kA, SRC_A), (ax[1], RES_B, kB, SRC_B)]:
    xs = np.arange(len(srcs)); w_ = 0.8 / 4
    for j, d in enumerate(["twoscale", "mmd", "glob", "spec"]):
        a_.bar(xs + (j - 1.5) * w_, [get(res, s, K_, d) or 0 for s in srcs], w_, label=d)
    a_.set_xticks(xs); a_.set_xticklabels(srcs, rotation=25, fontsize=8); a_.set_ylim(0, 1.02); a_.set_ylabel("paired detection power")
    a_.set_title(f"{res['rows'][0]['part']} populations, K={K_}"); a_.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/step13_figure.png", dpi=130); plt.show()
print("=" * 90); print("STEP 13 VERDICT REPORT"); print("=" * 90)
for nm, res in [("weekly", RES_A), ("4-week", RES_B)]:
    print(f"[{nm}] test blocks {res['TB']}"); print(f"[{nm}] FPR " + " ".join(f"{d}={res['FPR'][d][0]:.3f}[{res['FPR'][d][1]:.3f},{res['FPR'][d][2]:.3f}]" for d in DETS))
    for r in res["rows"]:
        print(f"[{nm}] {r['source']:>19} K={r['K']:4d} N~{r['N_mean']:.0f} ts_prec={fmt(r['ts_precision'])} ts_rec={fmt(r['ts_recall'])} | " + " ".join(f"{d}={fmt(r.get(d))}[{fmt(r.get(d+'_ci', (None, None))[0])},{fmt(r.get(d+'_ci', (None, None))[1])}]" for d in DETS))
    for r in res["shard"]: print(f"[{nm}-shard] {r['source']}: " + " ".join(f"{d}={fmt(r[d])}" for d in DETS) + f" n={r['n']}")
print("=" * 90)
with open(f"{OUTDIR}/step13_results.json", "w") as fh: json.dump(dict(A=RES_A, B=RES_B), fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
