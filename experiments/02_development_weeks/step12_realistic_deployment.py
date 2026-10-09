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
# Table 1 columns map to detector keys: Two-scale = `twoscale`, Global stream = `glob`, MMD = `mmd`,
# Spectral = `spec`, Per-account / HC / Fraudar = `acct` / `hc` / `fraudar`; all at K = 50, ρ = 0 (`{d}|0.0`).
#
# | Paper location | Quantity | Value in paper | Where computed here |
# |---|---|---|---|
# | Table 1, row "MetaMathQA (generated math)"; App. P (Step 12, "Two-scale beats spectral on a generated set") | weekly paired detection: Two-scale / Global / MMD / Spectral / Per-account, HC, Fraudar | 0.93 / 0.71 / 0.60 / 0.19 / ≤0.01; App. P: Pass (0.93 vs 0.19) | Cell 8 per-row print (`metamath`, K = 50); verdict "[B]" lines; Cell 10 print "P12.3/4" |
# | Table 1, row "Magicoder (generated code)" | same columns | 0.62 / 0.46 / 0.32 / 0.00 / 0.00 | Cell 8 print (`magicoder`, K = 50); verdict "[B]" |
# | Table 1, row "Alpaca (generated, broad)" | same columns | 0.01 / 0.00 / 0.01 / 0.00 / 0.00 | Cell 8 print (`alpaca`, K = 50); verdict "[B]" |
# | Table 1, row "Narrow cluster of real prompts"; App. P (Step 12, "Natural campaigns ≤ 0.2 on every detector") | same columns | 0.12 / 0.04 / 0.27 / 0.00 / 0.00; App. P: Fail (MMD 0.27) | Cell 8 print (`wild_narrow`, K = 50); Cell 10 print "P12.2" |
# | Table 1, row "Random real prompts" | same columns | 0.01 / 0.00 / 0.01 / 0.00 / 0.00 | Cell 8 print (`random`, K = 50); verdict "[B]" |
# | Table 1, last row "Realised false alarms"; Fig. 4b; §5 "Calibration breaks first"; App. D.1; App. P (Step 12, "Every detector's false alarms ≤ 2%") | held-out false alarms at nominal 1%, rolling calibration, week-clustered intervals | Two-scale 0.7%, Global 4.1%, MMD 11.6%, Spectral 7.8%, Per-account / HC / Fraudar 7.5 / 5.3 / 2.9%; volume test 14.2% (§5: "2.9–14.2%"; D.1: "4.1–7.8%"); App. P: Fail (only two-scale) | Cell 7 print "held-out FPR" (`FPR[d]` with `boot_ci`); verdict "[FPR]" lines; Cell 10 print "P12.7" |
# | App. P (Step 12, "Volume-share test ≤ 0.1") | max paired detection of the volume test | Pass (≤ 0.02) | Cell 10 print "P12.1" (`vol` column of Cell 8) |
# | App. P (Step 12, "Generated sets at K = 200: MMD or global ≥ 0.5") | — | Partial (K = 200 infeasible) | Cell 10 print "cells skipped" and "P12.3/4" (largest evaluated K) |
# | App. P (Step 12, "Partial linkage raises per-account or HC by ≥ 0.1") | — | Untestable (base ≈ 0) | Cell 8 "linked(rho=0.3)" part of the print; Cell 10 print "P12.5" |
# | App. P (Step 12, "Sharding: spectral ≤ 0.1"); App. P "Corrections made during the study" (Sharding) | — | Untestable (1–5 shards; unpaired) | Part 5 (Cell 9) print "sharded ..."; Cell 10 print "P12.6" |
# | Table 1 caption; §5; App. D | K = 50 volume-matched colluders, 24 development weeks; "about 1,640 queries a week" | as quoted | Cell 1 (`KS`, `N_TEST_W`); 1,640 plausibly the `N~` printed for K = 50 in Cell 8 (protocol value reused by later Steps); not isolated |
# | §2 "Data" | monitored accounts have at least 10 queries in the week; windows hold 64–364 accounts | as quoted | Cell 1 `MIN_PER_ACCT`; window sizes plausibly the "accounts per test-week window" print in Cell 10 (not isolated) |
# | App. Table 16 (Fourier modes, Reference, Calibration, MMD baseline, Fraudar baseline) | m = 300 modes with augmented real whitening; h = 4 coarse with h = 0.5 fine; up to 100,000 benign turns from the four preceding weeks; 40 null windows per week, thresholds from the 8 preceding weeks; MMD window sample of 2,000 against 20 reference batches of 500; 100 k-means topics, weights 1/ log(degree + 5) | as quoted | Cell 1 configuration; Cell 5 (`make_bank_aug`, `mmd2`, `fraudar`); Cell 6 (lagged reference) |
# | App. P "Corrections made during the study" (Padding construction) | in Steps 12–13 the padded campaign assigned natural prompts to separate colluder accounts | (segregated construction; two-scale robustness 0.62 holds only for it) | Cell 8 `campaign_points` + `labc`: `metamath_pad50` here is the segregated construction; the 0.62 value is not isolated (may come from Step 13) |
#
# Not reported in the paper: K = 10 and K = 200 cells, `metamath_pad50` cells, `any_geo` (union), precision at K
# of the top-r² accounts (the paper's group precision 0.94–1.00 refers to the two-scale flagged group, not computed
# here), the ρ = 0.3 linkage values, the sharding numbers, and the figure.
# Note: the description below motivates the two-scale test as detecting "topic-concentrated but deduplicated"
# groups; that mechanism was later rejected (App. D.3; App. P "Corrections made during the study", Mechanism claims).
# "Innovation 2/3" are internal labels from an earlier draft.

# %% [markdown]
# # Step 12 — Realistic deployment test on WildChat: volume-matched colluders, whole weekly populations, all baselines, real distillation prompt sets
# **Kaggle:** GPU + Internet + `HF_TOKEN` → **Run All** (≈60–90 min; reuses `step11_wildchat_E384_full.npz` if present). `QUICK=True` first.
#
# **What changes versus Steps 9–11 (reviewer requests).**
# * **Volume-matched colluders:** each colluder's weekly volume is drawn from the benign volume distribution of that week, so N = sum of K matched volumes (K = 10 / 50 / 200).
# * **Whole populations:** the monitored window is the whole week's population of benign accounts with ≥ 10 queries (70% random subsamples give repeated test windows). Rolling calibration on the 8 preceding weeks, lagged references, 24 test weeks, **week-clustered bootstrap** intervals.
# * **Augmented real whitening:** real and imaginary Fourier parts whitened jointly (fixes the pseudo-covariance issue).
# * **Detectors, each reported separately at 1%:** spectral upper edge, Higher Criticism, per-account max, global stream, **Liu–Guo–Dong MMD** (unbiased MMD² vs 20 benign reference batches, Gaussian kernel, median bandwidth; MiniLM instead of their bge-small), **Fraudar** (greedy densest block on the account × topic matrix with log column weights), **volume share** (top-10 accounts' share of traffic), and the new **two-scale test** (Innovation 2): the group carried by the top spectral eigenvector is scored by log S̄_coarse − log S̄_fine, which is high for groups that are topic-concentrated but deduplicated.
# * **Campaign sources:** real *generated* distillation sets — `metamath` (MetaMathQA), `magicoder` (Magicoder-OSS-Instruct), `alpaca` — plus natural WildChat prompts `wild_narrow` (1 of 30 clusters) and `random`, and `metamath_pad50` (half natural padding).
# * **Partial metadata linkage (Innovation 3):** ρ = 30% of colluders are known to be linked and merged into one account.
# * **Attacker-aware sharding:** 100-account shards, one colluder per shard.
# * **Operational metrics:** precision at K of the top-ranked accounts, and expected false alarms per week.
#
# | ID | Locked prediction |
# |---|---|
# | **P12.1** | Volume-share test: paired power ≤ 0.1 for every volume-matched campaign |
# | **P12.2** | Natural-prompt campaigns (`wild_narrow`, `random`): every geometry detector ≤ 0.2 at every K |
# | **P12.3** | Generated sets (`metamath`, `magicoder`) at K = 200: MMD or global-stream power ≥ 0.5 |
# | **P12.4** | Two-scale test beats the spectral test on at least one generated source at K = 200 |
# | **P12.5** | Partial linkage (ρ = 0.3) raises per-account or HC power by ≥ 0.1 wherever base power is in [0.1, 0.9] |
# | **P12.6** | Attacker-aware sharding: spectral power ≤ 0.1 |
# | **P12.7** | Each detector's held-out FPR ≤ 2% (nominal 1%), week-clustered 95% CI reported |

# %%
# ===== Cell 1: configuration =====
import math, json, time, os, gc
import numpy as np
from scipy.stats import rankdata
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
# [PAPER] App. Table 16: m = 300 modes (augmented real whitening); h = 4 coarse with h = 0.5 fine; Table 1: nominal 1%
H_COARSE, H_FINE, M_MODES, ALPHA, FLOOR = 4.0, 0.5, 300, 0.01, 1e-4
# [PAPER] §2 "Data": at least 10 queries in the week; App. Table 16: thresholds from the 8 preceding weeks, lagged reference from the four preceding weeks
VOL_CAP, MIN_PER_ACCT, POOL_FRAC, SUB_FRAC, LAG, CAL_WEEKS = 500, 10, 0.2, 0.7, 4, 8
# [PAPER] App. Table 16: Fraudar on 100 k-means topics; MMD window sample of 2,000 against 20 reference batches of 500
N_TOPICS, N_REF_BATCH, REF_BATCH, MMD_WIN = 100, 20, 500, 2000
SOURCES = ["metamath", "magicoder", "alpaca", "wild_narrow", "random", "metamath_pad50"]
DETS = ["spec", "hc", "acct", "glob", "mmd", "fraudar", "vol", "twoscale"]
N_DISTILL = 25000
if QUICK:
    KS, N_TEST_W, NULL_PER_WEEK, REPS, RHO, SHARD_K = [10, 50], 3, 12, 2, 0.3, 20
    MMD_WIN, N_REF_BATCH, REF_BATCH = 400, 5, 200
else:
    # [PAPER] Table 1 caption: K = 50, 24 development weeks; App. Table 16: 40 null windows per week
    KS, N_TEST_W, NULL_PER_WEEK, REPS, RHO, SHARD_K = [10, 50, 200], 24, 40, 4, 0.3, 50
CACHE11 = f"{OUTDIR}/step11_wildchat_E384_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
CACHE12 = f"{OUTDIR}/step12_distill_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
N_CONV = 30000 if QUICK else None; DIMS = [384]
print(f"backend {'GPU ' + torch.cuda.get_device_name(0) if USE_GPU else 'CPU'} | QUICK={QUICK} | KS={KS} test weeks={N_TEST_W}")

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
# ===== Cell 3: real generated distillation prompt sets (MetaMathQA, Magicoder-OSS-Instruct, Alpaca), MiniLM, cached =====
if os.path.exists(CACHE12):
    _c = np.load(CACHE12); DIST = {k[2:]: _c[k] for k in _c.files if k.startswith("D_")}; print({k: v.shape for k, v in DIST.items()})
else:
    t0 = time.time(); DIST = {}
    if SYNTHETIC_TEST:
        r = np.random.default_rng(3)
        for i, nm in enumerate(["metamath", "magicoder", "alpaca"]):
            c = r.standard_normal(384) * 2; DIST[nm] = (c + r.standard_normal((3000, 384)) * 0.7).astype(np.float16)
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
        specs = {"metamath": ("meta-math/MetaMathQA", ["query"]), "magicoder": ("ise-uiuc/Magicoder-OSS-Instruct-75K", ["problem"]),
                 "alpaca": ("tatsu-lab/alpaca", ["instruction", "input"])}
        for nm, (repo, fields) in specs.items():
            try:
                ds = load_dataset(repo, split="train", token=tok); cols = [f for f in fields if f in ds.column_names]
                if not cols: print(f"{nm}: fields {fields} not in {ds.column_names}; skipped"); continue
                idx = np.random.default_rng(SEED).choice(len(ds), min(N_DISTILL, len(ds)), replace=False)
                sub = ds.select(idx.tolist()); texts = [" ".join(str(row[c]) for c in cols if row[c]).strip()[:1000] for row in sub]
                texts = list(dict.fromkeys(t for t in texts if t))          # exact duplicates removed
                DIST[nm] = enc.encode(texts, batch_size=512, convert_to_numpy=True, normalize_embeddings=True).astype(np.float16)
                print(f"{nm}: {len(texts)} prompts embedded ({time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"{nm}: could not load ({e}); skipped")
    np.savez(CACHE12, **{f"D_{k}": v for k, v in DIST.items()})
SOURCES = [s for s in SOURCES if s in ("wild_narrow", "random") or s.replace("_pad50", "") in DIST]
print("campaign sources:", SOURCES)

# %% [markdown]
# ## Part 1 — accounts, projection, topics

# %%
# ===== Cell 4: accounts (>= 10 queries), weeks, projection (full 384-d), topics, pool clusters =====
rng = np.random.default_rng(SEED + 1); n_acct = int(ACCT.max()) + 1; is_pool = rng.random(n_acct) < POOL_FRAC
order = np.lexsort((ACCT, WEEK)); keys = WEEK[order].astype(np.int64) * n_acct + ACCT[order]
groups = np.split(order, np.flatnonzero(np.diff(keys)) + 1); WEEK_ACCTS = {w: [] for w in range(n_w)}
for g in groups:
    a = int(ACCT[g[0]]); w = int(WEEK[g[0]])
    if is_pool[a] or len(g) < MIN_PER_ACCT: continue
    if len(g) > VOL_CAP: g = rng.choice(g, VOL_CAP, replace=False)
    WEEK_ACCTS[w].append(g)
print("accounts with >=10 queries per week: " + ", ".join(f"w{w}:{len(v)}" for w, v in WEEK_ACCTS.items() if v))
pool_idx = np.flatnonzero(is_pool[ACCT])
Z = make_Z(384); W384 = UB[:, :384] / math.sqrt(EW[:384].sum() / 4.0)
DZ = {k: ((v.astype(np.float32) - MU_E) @ W384).astype(np.float32) for k, v in DIST.items()}
EN = E16                                                    # normalised MiniLM embeddings (for MMD)
from sklearn.cluster import MiniBatchKMeans
t0 = time.time(); ref_all = np.concatenate([np.concatenate(WEEK_ACCTS[w]) for w in range(n_w) if WEEK_ACCTS[w]])
km = MiniBatchKMeans(n_clusters=N_TOPICS, random_state=SEED, batch_size=4096, n_init=3).fit(Z[np.random.default_rng(1).choice(ref_all, min(200000, len(ref_all)), replace=False)])
TOPIC = km.predict(Z).astype(np.int32)
CL30 = MiniBatchKMeans(n_clusters=30, random_state=SEED + 1, batch_size=4096, n_init=3).fit(Z[pool_idx]).labels_
s30 = np.bincount(CL30, minlength=30); NARROW = pool_idx[CL30 == int(np.argsort(-s30)[3])]
print(f"topics and pool clusters done ({time.time()-t0:.0f}s); wild_narrow source size {len(NARROW)}")
_sub = EN[np.random.default_rng(2).choice(len(EN), 2000, replace=False)].astype(np.float32)
_d2 = ((_sub[:1000, None, :] - _sub[None, 1000:, :]) ** 2).sum(-1); MMD_BW = float(np.sqrt(np.median(_d2) / 2)); del _d2
print(f"MMD Gaussian bandwidth (median heuristic) = {MMD_BW:.3f}")


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
    # [PAPER] App. Table 16 (Fraudar baseline): account × topic count matrix, column weights 1/ log(degree + 5), greedy peeling
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
                vol=float(np.sort(cnt)[-10:].sum() / cnt.sum()), twoscale=float(np.log(union_S(Yc, cnt, grp)) - np.log(union_S(Yf, cnt, grp))), v2=v2)
def hc_stat(x, ns):
    M_ = len(ns); p = np.sort((1 + M_ - np.searchsorted(ns, x, side="left")) / (M_ + 1)); A_ = len(p); k = max(1, A_ // 2)
    i = np.arange(1, k + 1); pp = p[:k]; return float(np.max(np.sqrt(A_) * (i / A_ - pp) / np.sqrt(pp * (1 - pp))))


# %% [markdown]
# ## Part 2 — per-week banks and null windows

# %%
# ===== Cell 6: per-week lagged banks, MMD reference batches, null windows (cached per week) =====
t0 = time.time(); okw = [w for w in range(CAL_WEEKS, n_w) if len(WEEK_ACCTS[w]) >= 40 and all(len(WEEK_ACCTS[v]) >= 20 for v in range(w - CAL_WEEKS, w))]
TW = [int(okw[i]) for i in np.linspace(0, len(okw) - 1, min(N_TEST_W, len(okw))).astype(int)] if okw else []
NEED = sorted(set(w for tw in TW for w in range(tw - CAL_WEEKS, tw + 1)))
print(f"test weeks {TW}; weeks needing banks/null windows: {len(NEED)}")
BANKS_W, REFS_W, NULLW = {}, {}, {}
def sub_window(w, r):
    k = max(20, int(SUB_FRAC * len(WEEK_ACCTS[w]))); pick = r.choice(len(WEEK_ACCTS[w]), k, replace=False); parts = [WEEK_ACCTS[w][i] for i in pick]
    vols = np.array([len(p) for p in parts]); return np.concatenate(parts), np.repeat(np.arange(k), vols), vols
for w in NEED:
    prev = [np.concatenate(WEEK_ACCTS[v]) for v in range(max(0, w - LAG), w) if WEEK_ACCTS[v]]
    # [PAPER] App. Table 16 (Reference): up to 100,000 benign turns from the four preceding weeks (lagged)
    ref = np.concatenate(prev) if prev else np.concatenate(WEEK_ACCTS[w]); ref = np.random.default_rng(w).choice(ref, min(100000, len(ref)), replace=False)
    BANKS_W[w] = (make_bank_aug(Z[ref], H_COARSE, SEED + 7), make_bank_aug(Z[ref], H_FINE, SEED + 8))
    rb = np.random.default_rng(w + 1).choice(ref, N_REF_BATCH * REF_BATCH, replace=len(ref) < N_REF_BATCH * REF_BATCH).reshape(N_REF_BATCH, REF_BATCH)
    REFS_W[w] = [torch.as_tensor(EN[b].astype(np.float32), device=DEV) if USE_GPU else EN[b].astype(np.float32) for b in rb]
    recs = []
    for t in range(NULL_PER_WEEK):
        idx, lab, vols = sub_window(w, np.random.default_rng(SEED + 100 * w + t)); recs.append(window_stats(idx, lab, np.bincount(lab), *BANKS_W[w], REFS_W[w]))
    NULLW[w] = recs
    print(f"week {w}: {len(WEEK_ACCTS[w])} accounts, bank r={BANKS_W[w][0]['r']} ({time.time()-t0:.0f}s)")


# %% [markdown]
# ## Part 3 — calibration

# %%
# ===== Cell 7: rolling calibration (8 preceding weeks) and held-out FPR with week-clustered bootstrap =====
def thresholds(tw):
    recs = [r for w in range(tw - CAL_WEEKS, tw) for r in NULLW[w]]; ns = np.sort(np.concatenate([r["r2"] for r in recs]))
    T = {d: float(np.quantile([r[d] for r in recs], 1 - ALPHA)) for d in DETS if d != "hc"}
    T["hc"] = float(np.quantile([hc_stat(r["r2"], ns) for r in recs], 1 - ALPHA)); T["ns"] = ns; return T
def fires(s, T):
    f = {d: s[d] > T[d] for d in DETS if d != "hc"}; f["hc"] = hc_stat(s["r2"], T["ns"]) > T["hc"]
    f["any_geo"] = f["spec"] or f["hc"] or f["glob"] or f["mmd"] or f["fraudar"] or f["twoscale"]; return f
THR = {tw: thresholds(tw) for tw in TW}; ALLD = DETS + ["any_geo"]
def boot_ci(per_week, B=1000, rng=np.random.default_rng(7)):
    x = np.array(per_week); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]; return float(x.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
FPR = {}
for d in ALLD:
    pw = [np.mean([fires(r, THR[tw])[d] for r in NULLW[tw]]) for tw in TW]; FPR[d] = boot_ci(pw)
    # [PAPER] Table 1 last row, Fig. 4b, §5, App. D.1: two-scale 0.7%, global 4.1%, MMD 11.6%, spectral 7.8%, per-account/HC/Fraudar 7.5 / 5.3 / 2.9%, volume 14.2%
    print(f"held-out FPR {d:>9}: {FPR[d][0]:.3f} [{FPR[d][1]:.3f}, {FPR[d][2]:.3f}]  (nominal {ALPHA}; 'any_geo' is the union)")


# %% [markdown]
# ## Part 4 — volume-matched campaigns

# %%
# ===== Cell 8: volume-matched campaigns, paired evaluation, partial linkage, precision at K =====
# Note (App. P "Corrections made during the study", Padding construction): for `metamath_pad50` the generated
# half comes first and `labc = np.repeat(np.arange(K_), cv)` below assigns points to colluders in order, so the
# natural padding lands on separate colluder accounts (segregated padding). Step 14 mixed padding into every account.
def campaign_points(src, n, r):
    if src == "random": ix = r.choice(pool_idx, n, replace=False); return Z[ix], EN[ix], TOPIC[ix]
    if src == "wild_narrow": ix = r.choice(NARROW, min(n, len(NARROW)), replace=n > len(NARROW)); return Z[ix], EN[ix], TOPIC[ix]
    base = src.replace("_pad50", ""); nd = n // 2 if src.endswith("_pad50") else n
    D = DIST[base]; ix = r.choice(len(D), min(nd, len(D)), replace=nd > len(D)); Zd = DZ[base][ix]; Ed = D[ix].astype(np.float32); Td = km.predict(Zd).astype(np.int32)
    if nd < n:
        jx = r.choice(pool_idx, n - nd, replace=False); Zd, Ed, Td = np.vstack([Zd, Z[jx]]), np.vstack([Ed, EN[jx].astype(np.float32)]), np.concatenate([Td, TOPIC[jx]])
    return Zd, Ed, Td
t0 = time.time(); rowsB = []
for src in SOURCES:
    for K_ in KS:
        per_week = {f"{d}|{rho}": [] for d in ALLD for rho in [0.0, RHO]}; prec, Ns = [], []
        for tw in TW:
            Bc, Bf = BANKS_W[tw]; T = THR[tw]; acc = {k: [] for k in per_week}
            for rep in range(REPS):
                r = np.random.default_rng(SEED + 9_000_000 + 1000 * tw + 10 * rep + K_)
                idx, lab, vols = sub_window(tw, r); A_ = int(lab.max()) + 1
                if K_ >= A_ // 2: continue
                base = fires(window_stats(idx, lab, np.bincount(lab, minlength=A_), Bc, Bf, REFS_W[tw]), T)
                cv = r.choice(np.array([len(g) for g in WEEK_ACCTS[tw]]), K_, replace=True); N = int(cv.sum()); Ns.append(N)
                Zc, Ec, Tc = campaign_points(src, N, np.random.default_rng(SEED + 17 * tw + rep + K_ + SOURCES.index(src)))
                keep = lab >= K_; lab_k = lab[keep]
                for rho in [0.0, RHO]:
                    labc = np.repeat(np.arange(K_), cv)
                    if rho > 0: nl = max(2, int(round(rho * K_))); labc = np.where(labc < nl, 0, labc)   # first nl colluders merged (known link)
                    _, inv = np.unique(np.concatenate([labc, lab_k]), return_inverse=True)              # compact labels: no empty accounts
                    lc, lk = inv[:len(labc)], inv[len(labc):]; cnt = np.bincount(inv)
                    s = window_stats(idx[keep], lk, cnt, Bc, Bf, REFS_W[tw], extra=(Zc, lc), en_extra=Ec, top_extra=Tc); f = fires(s, T)
                    for d in ALLD: acc[f"{d}|{rho}"].append(float(f[d] and not base[d]))
                    if rho == 0.0:
                        top = np.argsort(-s["r2"])[:K_]; prec.append(float(np.mean(top < K_)))      # labels 0..K-1 are the colluders when rho = 0
            for k in per_week:
                if acc[k]: per_week[k].append(np.mean(acc[k]))
        row = dict(source=src, K=K_, N_mean=float(np.mean(Ns)) if Ns else 0, prec_at_K=float(np.mean(prec)) if prec else float("nan"))
        for k, v in per_week.items():
            if v: m, lo, hi = boot_ci(v); row[k] = m; row[k + "_ci"] = (lo, hi)
        rowsB.append(row)
        # [PAPER] Table 1 rows (K = 50): MetaMathQA 0.93/0.71/0.60/0.19/≤0.01; Magicoder 0.62/0.46/0.32/0.00/0.00; Alpaca 0.01/0.00/0.01/0.00/0.00;
        # narrow cluster 0.12/0.04/0.27/0.00/0.00; random 0.01/0.00/0.01/0.00/0.00 (twoscale/glob/mmd/spec/acct-hc-fraudar)
        print(f"{src:>15} K={K_:4d} N~{row['N_mean']:7.0f} prec@K={row['prec_at_K']:.2f} | " + " ".join(f"{d}={row.get(d+'|0.0', float('nan')):.2f}" for d in ALLD)
              + f" | linked(rho={RHO}): acct={row.get('acct|'+str(RHO), float('nan')):.2f} hc={row.get('hc|'+str(RHO), float('nan')):.2f}  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Part 5 — attacker-aware sharding

# %%
# ===== Cell 9: attacker-aware sharding (100-account shards, one colluder per shard) =====
t0 = time.time(); SH = 100; rowsS = []
def shard_null(w, r):
    k = min(SH, len(WEEK_ACCTS[w])); pick = r.choice(len(WEEK_ACCTS[w]), k, replace=False); parts = [WEEK_ACCTS[w][i] for i in pick]
    vols = np.array([len(p) for p in parts]); return np.concatenate(parts), np.repeat(np.arange(k), vols), vols
for src in [s for s in ["metamath", "wild_narrow"] if s in SOURCES]:
    hits = {d: [] for d in ALLD}
    for tw in TW[: max(4, len(TW) // 2)]:
        Bc, Bf = BANKS_W[tw]; recs = []
        for w in range(tw - CAL_WEEKS, tw):
            for t in range(10):
                idx, lab, vols = shard_null(w, np.random.default_rng(SEED + 31 * w + t)); recs.append(window_stats(idx, lab, np.bincount(lab), Bc, Bf, REFS_W[tw]))
        ns = np.sort(np.concatenate([r["r2"] for r in recs])); n_sh = max(1, len(WEEK_ACCTS[tw]) // SH)
        a_sh = ALPHA / n_sh                                   # weekly level alpha, Bonferroni over shards
        Ts = {d: float(np.quantile([r[d] for r in recs], 1 - a_sh)) for d in DETS if d != "hc"}; Ts["hc"] = float(np.quantile([hc_stat(r["r2"], ns) for r in recs], 1 - a_sh)); Ts["ns"] = ns
        for rep in range(REPS):
            r = np.random.default_rng(SEED + 7_000 * tw + rep); fired = {d: False for d in ALLD}
            for sh in range(min(SHARD_K, n_sh)):                # one colluder in each of the first SHARD_K shards
                idx, lab, vols = shard_null(tw, r); keep = lab >= 1; cv = int(r.choice(vols))
                Zc, Ec, Tc = campaign_points(src, cv, r); _, inv = np.unique(np.concatenate([np.zeros(cv, int), lab[keep]]), return_inverse=True)
                cnt = np.bincount(inv); s = window_stats(idx[keep], inv[cv:], cnt, Bc, Bf, REFS_W[tw], extra=(Zc, inv[:cv]), en_extra=Ec, top_extra=Tc); f = fires(s, Ts)
                for d in ALLD: fired[d] |= bool(f[d])
            for d in ALLD: hits[d].append(float(fired[d]))
    # [PAPER] App. P (Step 12): sharding Untestable (1–5 shards; unpaired); "Corrections": identical numbers for natural and generated campaigns
    row = dict(source=src, colluders=SHARD_K, **{d: float(np.mean(v)) for d, v in hits.items()}); rowsS.append(row)
    print(f"sharded {src:>12}: " + " ".join(f"{d}={row[d]:.2f}" for d in ALLD) + f"  (weekly detection, colluders spread one per shard; {time.time()-t0:.0f}s)")

# %% [markdown]
# ## Predictions, figure, VERDICT REPORT — copy into the chat

# %%
# ===== Cell 10 (fixed): predictions, figure, verdict — robust to cells that were skipped =====
fmt = lambda x: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
g = lambda src, K_, d, rho=0.0: next((r.get(f"{d}|{rho}") for r in rowsB if r["source"] == src and r["K"] == K_), None)
done = lambda src, K_: g(src, K_, "spec") is not None
skipped = [(r["source"], r["K"]) for r in rowsB if not done(r["source"], r["K"])]
print("cells skipped (K >= half of a 70% weekly subsample):", skipped if skipped else "none")
# [PAPER] §2 "Data": windows hold 64–364 accounts (plausibly from this print; not isolated)
print("accounts per test-week window (70% subsample):", {tw: int(SUB_FRAC * len(WEEK_ACCTS[tw])) for tw in TW})
# [PAPER] App. P (Step 12): P12.1 Pass (≤ 0.02)
print("P12.1 volume test max paired power:", fmt(max((g(s, k, "vol") or 0) for s in SOURCES for k in KS)))
# [PAPER] App. P (Step 12): P12.2 Fail (MMD 0.27); Table 1 "Narrow cluster of real prompts", MMD 0.27
print("P12.2 natural campaigns, max geometry detector power:",
      {s: fmt(max((g(s, k, d) or 0) for d in ["spec", "hc", "acct", "glob", "mmd", "fraudar", "twoscale"] for k in KS)) for s in ["wild_narrow", "random"] if s in SOURCES})
for s in ["metamath", "magicoder", "alpaca"]:
    if s not in SOURCES: continue
    ks = [k for k in KS if done(s, k)]
    if not ks: print(f"P12.3/4 {s}: no evaluated cells"); continue
    # [PAPER] App. P (Step 12): P12.3 Partial (K = 200 infeasible); P12.4 Pass (0.93 vs 0.19) = Table 1 MetaMathQA two-scale vs spectral
    k = max(ks); print(f"P12.3/4 {s} K={k} (largest evaluated): mmd {fmt(g(s, k, 'mmd'))} glob {fmt(g(s, k, 'glob'))} spec {fmt(g(s, k, 'spec'))} twoscale {fmt(g(s, k, 'twoscale'))}")
# [PAPER] App. P (Step 12): P12.5 Untestable (base ≈ 0)
for r in rowsB:
    for d in ["acct", "hc"]:
        b, l = r.get(f"{d}|0.0"), r.get(f"{d}|{RHO}")
        if b is not None and 0.1 <= b <= 0.9: print(f"P12.5 {r['source']} K={r['K']} {d}: {fmt(b)} -> {fmt(l)} with linkage")
# [PAPER] App. P (Step 12): P12.6 Untestable (1–5 shards; unpaired)
for r in rowsS: print(f"P12.6 sharded {r['source']}: spectral {fmt(r['spec'])}, per-account {fmt(r['acct'])}, any {fmt(r['any_geo'])}")
# [PAPER] App. P (Step 12): P12.7 Fail (only two-scale); values as in Table 1 last row
print("P12.7 FPR:", {d: fmt(FPR[d][0]) for d in ALLD})
import matplotlib.pyplot as plt
KS_PLOT = [k for k in KS if any(done(s, k) for s in SOURCES)]
fig, ax = plt.subplots(1, len(KS_PLOT), figsize=(7 * len(KS_PLOT), 5), squeeze=False)
for i, K_ in enumerate(KS_PLOT):
    xs = np.arange(len(SOURCES)); w_ = 0.8 / len(DETS)
    for j, d in enumerate(DETS):
        ax[0, i].bar(xs + (j - len(DETS) / 2) * w_, [g(s, K_, d) or 0 for s in SOURCES], w_, label=d)
    ax[0, i].set_xticks(xs); ax[0, i].set_xticklabels(SOURCES, rotation=30, fontsize=8); ax[0, i].set_ylim(0, 1.02)
    ax[0, i].set_title(f"volume-matched, whole-week populations, K={K_}"); ax[0, i].set_ylabel("paired detection power")
    if i == 0: ax[0, i].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/step12_figure.png", dpi=130); plt.show()
print("=" * 90); print("STEP 12 VERDICT REPORT"); print("=" * 90)
print("config:", json.dumps(dict(QUICK=QUICK, KS=KS, TW=TW, ALPHA=ALPHA, H=(H_COARSE, H_FINE), MIN_PER_ACCT=MIN_PER_ACCT, sources=SOURCES)))
for d in ALLD: print(f"[FPR] {d:>9}: {FPR[d][0]:.3f} [{FPR[d][1]:.3f}, {FPR[d][2]:.3f}]")
for r in rowsB:
    if not done(r["source"], r["K"]): print(f"[B] {r['source']:>15} K={r['K']:4d} skipped"); continue
    ci = lambda d: r.get(d + "|0.0_ci", (float("nan"), float("nan")))
    print(f"[B] {r['source']:>15} K={r['K']:4d} N~{r['N_mean']:.0f} prec@K={fmt(r['prec_at_K'])} | " + " ".join(f"{d}={fmt(r.get(d+'|0.0'))}[{fmt(ci(d)[0])},{fmt(ci(d)[1])}]" for d in ALLD))
    print(f"    linked rho={RHO}: " + " ".join(f"{d}={fmt(r.get(d+'|'+str(RHO)))}" for d in ALLD))
for r in rowsS: print(f"[S] {r['source']}: " + " ".join(f"{d}={fmt(r[d])}" for d in ALLD))
print("=" * 90)
with open(f"{OUTDIR}/step12_results.json", "w") as fh: json.dump(dict(FPR=FPR, B=rowsB, S=rowsS, TW=TW), fh, indent=1, default=float)
