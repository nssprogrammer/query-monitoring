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
# | Fig. 4a (App. D.1); App. Table 5, last row "Held-out false alarms at A = 100 / 500" | held-out suite false alarms at nominal 5%, schemes fixed / rolling / rolling + lagged ref. | 18.8% / 56.2%; 5.0% / 24.7%; 3.4% / 28.7% | Part A cell, print "held-out suite FPR" (`rowsA[...]["fpr"]`); Part C print "P10.1"; verdict "[A]" lines |
# | App. Table 5, row "100, 20 random natural 20,000" | share; paired detection Fixed / Rolling / Rolling + lag | 0.94; 0.00 / 0.00 / 0.00 | Part B cell, `share` and `{scheme}_attrib` in the per-row print; verdict "[B]" lines (`att`) |
# | App. Table 5, row "100, 20 broad capability (1 of 10) 5,000" | share; Fixed / Rolling / Rolling + lag | 0.81; 0.03 / 0.04 / 0.21 | Part B cell, as above (`capability`, N = 5000) |
# | App. Table 5, row "100, 20 broad capability 20,000" | share; Fixed / Rolling / Rolling + lag | 0.94; 0.82 / 0.90 / 0.93 | Part B cell, as above (`capability`, N = 20000) |
# | App. Table 5, row "100, 20 narrow capability (1 of 30) 1,000" | share; Fixed / Rolling / Rolling + lag | 0.46; 0.00 / 0.00 / 0.03 | Part B cell, as above (`capability_narrow`, N = 1000) |
# | App. Table 5, row "100, 20 narrow capability 5,000"; App. F "Small monitored populations" | share; Fixed / Rolling / Rolling + lag | 0.81; 0.78 / 0.93 / 0.94 ("caught 94% of the time in 100-account windows") | Part B cell, as above (`capability_narrow`, N = 5000) |
# | App. Table 5, row "100, 20 broad, 50% natural padding 20,000" | share; Fixed / Rolling / Rolling + lag | 0.94; 0.25 / 0.04 / 0.38 | Part B cell, as above (`capability_pad50`, N = 20000) |
# | App. Table 5, row "500, 100 broad capability 5,000" | share; Fixed / Rolling / Rolling + lag | 0.46; 0.10 / 0.07 / 0.29 | Part B cell, as above (A = 500, `capability`, N = 5000) |
# | App. Table 5, row "500, 100 narrow capability 5,000" | share; Fixed / Rolling / Rolling + lag | 0.46; 0.44 / 0.78 / 0.71 | Part B cell, as above (A = 500, `capability_narrow`, N = 5000) |
# | App. F "Small monitored populations" | equal-volume colluders' fraction of the window | 46–94% | Part B cell, `share` |
# | Prediction log (Step 10) | locked predictions P10.1–P10.5 | P10.1 Partial (A = 100 yes; A = 500 no); P10.2 Partial (lagged 0.21, 0.29); P10.3 Pass (0.94; 0.71); P10.4 Pass; P10.5 Partial (A = 100 yes) | Part C "predictions" cell (prints "P10.1"–"P10.5") |
# | App. Table 5 caption; App. Table 16 (Calibration) | 8 held-out weeks, 3 campaigns × 3 windows per week, α = 5%; 250 windows from the 4 preceding weeks (small populations) | as quoted | Cell 1 configuration (`N_TW`, `TRIALS9`, `W_TW`, `ALPHA`, `N_CAL_ROLL`, `LAG`) |
#
# Not reported in the paper: raw power and `excess` columns (the paper reports only the paired/attributable
# detection), cells not in Table 5 (e.g. N = 1,000 broad and random at smaller N, A = 500 random/padded/
# N = 20,000 rows), and the figure. Fig. 4a and the last row of Table 5 are the same numbers.

# %% [markdown]
# # Step 10 — Real LLM traffic: is the failure drift, granularity, or identifiability?
# **Kaggle:** GPU + Internet + `HF_TOKEN` secret → **Run All** (≈40–60 min; reuses `step9_wildchat_full.npz` if present, otherwise re-downloads and embeds). `QUICK=True` first.
#
# Step 9 found (i) 12% false alarms on held-out weeks and (ii) detection only when a broad-capability campaign dominated a week's traffic. Step 10 separates the explanations.
#
# **Calibration schemes** (each evaluated on 8 held-out test weeks; overall suite level **α = 5%**, i.e. 1.67% per detector, so that rolling calibration with 250 windows has stable quantiles):
# 1. `fixed` — first 60% of weeks, global reference (Step 9).
# 2. `rolling` — the 4 weeks before each test week, global reference.
# 3. `rolling_lag` — the 4 weeks before, and every window whitened against the benign traffic of *its own* preceding 4 weeks (removes slow topical drift).
#
# **Paired evaluation.** Each campaign is injected into a window; the *same* window without the campaign is also scored. Reported: `excess` = P(detect | campaign) − P(detect | no campaign) and `attrib` = P(detect with campaign and not without). Campaign **share** of the window's traffic is reported.
#
# **Campaigns:** `random`, `capability` (1 of 10 clusters), `capability_narrow` (1 of 30 clusters), `capability_pad50`.
#
# | ID | Locked prediction |
# |---|---|
# | **P10.1** | `rolling` brings held-out suite FPR to ≤ 2× nominal (≤ 10%); `rolling_lag` to ≤ 1.5× (≤ 7.5%) |
# | **P10.2** | Broad `capability` at N = 5,000: attributable power ≤ 0.2 under every scheme (the failure is not drift) |
# | **P10.3** | `capability_narrow` at N = 5,000 under `rolling_lag`: attributable power ≥ 0.5 |
# | **P10.4** | Across campaigns, attributable power ≥ 0.5 occurs only when the campaign is ≥ 50% of window traffic, except for `capability_narrow` |
# | **P10.5** | 50% padding at N = 20,000 reduces attributable power by ≥ 0.5 under every scheme |

# %%
# ===== Cell 1: configuration =====
import math, json, time, os, gc
import numpy as np
from scipy.stats import rankdata
QUICK = False
SYNTHETIC_TEST = False          # True only for offline plumbing tests (no download)
SEED = 9091
OUTDIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
try:
    import torch
    HAS_TORCH = True; DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    HAS_TORCH, DEV = False, None
USE_GPU = HAS_TORCH and DEV.type == "cuda"
ALLOW_CPU = False
if not USE_GPU and not ALLOW_CPU and not SYNTHETIC_TEST:
    raise RuntimeError("No GPU visible. Kaggle: Settings -> Accelerator -> GPU, then restart.")
CHUNK = 65536 if USE_GPU else 8192
# [PAPER] App. Table 5 caption / Fig. 4a: α = 5% (suite level); App. Table 16: h = 4, m = 400 complex modes (small-population protocol)
D_DEF, H_BW, M_MODES, ALPHA, FLOOR = 32, 4.0, 400, 0.05, 1e-4
ALPHA_D = ALPHA / 3
VOL_CAP, MIN_PER_ACCT, POOL_FRAC, CAL_FRAC, N_CLUSTERS = 500, 3, 0.2, 0.6, 10
STRATS10 = ["random", "capability", "capability_narrow", "capability_pad50"]
if QUICK:
    N_CONV, AK9, N9, N_CAL9, N_TEST9, TRIALS9, W9 = 30000, [(50, 10)], [300, 1000], 80, 40, 1, 2
    N_TW, N_CAL_ROLL, N_NULL_TW, W_TW, LAG = 2, 40, 10, 2, 2
else:
    N_CONV, AK9, N9, N_CAL9, N_TEST9, TRIALS9, W9 = None, [(100, 20), (500, 100)], [1000, 5000, 20000], 600, 300, 3, 8
    # [PAPER] App. Table 5 caption: 8 held-out weeks, 3 campaigns × 3 windows per week; App. Table 16: 250 windows from the 4 preceding weeks
    N_TW, N_CAL_ROLL, N_NULL_TW, W_TW, LAG = 8, 250, 40, 3, 4
CACHE9 = f"{OUTDIR}/step9_wildchat_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
print(f"backend {'GPU ' + torch.cuda.get_device_name(0) if USE_GPU else 'CPU'} | QUICK={QUICK} SYNTHETIC_TEST={SYNTHETIC_TEST} | cache {CACHE9}")
N_NARROW_CLUSTERS = 30


# %% [markdown]
# ## Part 0 — WildChat-1M (cached from Step 9)

# %%
# ===== Cell 2: load WildChat-1M, extract English user turns, embed (cached) =====
def _synthetic_rows(n_ip=3000, n_rows=60000, rng=np.random.default_rng(0)):
    # offline plumbing test only: fake accounts/weeks/embeddings with topical structure
    ip = rng.integers(0, n_ip, n_rows); wk = rng.integers(0, 30, n_rows); topic = rng.integers(0, 40, n_ip)[ip]
    base = rng.standard_normal((40, 384)); E = base[topic] * 0.8 + rng.standard_normal((n_rows, 384)) * 0.6
    return ip.astype(np.int32), wk.astype(np.int16), E.astype(np.float32)

if os.path.exists(CACHE9):
    _c = np.load(CACHE9); Z_all, ACCT, WEEK = _c["Z_all"], _c["ACCT"], _c["WEEK"]; print(f"loaded cache: {len(Z_all)} queries")
else:
    t0 = time.time()
    if SYNTHETIC_TEST:
        ACCT, WEEK, E = _synthetic_rows()
    else:
        from datasets import load_dataset
        tok = None
        try:
            from kaggle_secrets import UserSecretsClient; tok = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            tok = os.environ.get("HF_TOKEN")
        split = "train" if N_CONV is None else f"train[:{N_CONV}]"
        ds = load_dataset("allenai/WildChat-1M", split=split, token=tok).select_columns(["conversation", "hashed_ip", "timestamp", "language"])
        print(f"loaded {len(ds)} conversations ({time.time()-t0:.0f}s)")
        texts, ips, tss = [], [], []
        for batch in ds.iter(batch_size=2000):
            for conv, ip, ts, lang in zip(batch["conversation"], batch["hashed_ip"], batch["timestamp"], batch["language"]):
                if lang != "English" or ip is None or ts is None: continue
                for turn in conv:
                    if turn.get("role") == "user" and turn.get("content"):
                        texts.append(turn["content"][:1000]); ips.append(ip); tss.append(ts)
        print(f"{len(texts)} English user turns from {len(set(ips))} IPs ({time.time()-t0:.0f}s)")
        uniq, ACCT = np.unique(np.array(ips), return_inverse=True); ACCT = ACCT.astype(np.int32)
        days = np.array([np.datetime64(t.replace(tzinfo=None) if hasattr(t, "replace") else t, "D") for t in tss])
        WEEK = ((days - days.min()).astype(np.int64) // 7).astype(np.int16)
        del ds; gc.collect()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            os.system("pip install -q sentence-transformers"); from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda" if USE_GPU else "cpu"); enc.max_seq_length = 256
        E = enc.encode(texts, batch_size=512, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        del texts; gc.collect()
    rng = np.random.default_rng(SEED); fit = E[rng.choice(len(E), min(100000, len(E)), replace=False)]
    mu = fit.mean(0); ew, U = np.linalg.eigh(np.cov(fit - mu, rowvar=False)); W_ = U[:, ::-1][:, :D_DEF]
    Z_all = ((E - mu) @ W_).astype(np.float32); Z_all /= math.sqrt(Z_all.var(0).sum() / 4.0)
    print(f"PCA-32 explains {ew[::-1][:D_DEF].sum()/ew.sum():.3f} of embedding variance")
    del E; gc.collect()
    np.savez_compressed(CACHE9, Z_all=Z_all, ACCT=ACCT, WEEK=WEEK); print(f"cached ({time.time()-t0:.0f}s)")
n_w = int(WEEK.max()) + 1; print(f"{len(Z_all)} queries, {ACCT.max()+1} accounts, {n_w} weeks")

# %% [markdown]
# ## Part 1 — accounts, weeks, global reference

# %%
# ===== Cell 3: account split, weekly windows, reference whitening, statistics =====
rng = np.random.default_rng(SEED + 1); n_acct = int(ACCT.max()) + 1
is_pool = rng.random(n_acct) < POOL_FRAC
cal_weeks = np.arange(int(round(CAL_FRAC * n_w))); test_weeks = np.arange(len(cal_weeks), n_w)
# per (week, account) query indices for benign (non-pool) accounts
order = np.lexsort((ACCT, WEEK)); W_s, A_s = WEEK[order], ACCT[order]
keys = W_s.astype(np.int64) * n_acct + A_s; brk = np.flatnonzero(np.diff(keys)) + 1
groups = np.split(order, brk)
WEEK_ACCTS = {w: [] for w in range(n_w)}
for g in groups:
    a = int(ACCT[g[0]]); w = int(WEEK[g[0]])
    if is_pool[a] or len(g) < MIN_PER_ACCT: continue
    if len(g) > VOL_CAP: g = rng.choice(g, VOL_CAP, replace=False)
    WEEK_ACCTS[w].append(g)
print("eligible benign accounts per week: " + ", ".join(f"w{w}:{len(v)}" for w, v in WEEK_ACCTS.items() if len(v)))
pool_idx = np.flatnonzero(is_pool[ACCT]); Z_pool = Z_all[pool_idx]
ref_idx = np.concatenate([np.concatenate(WEEK_ACCTS[w]) for w in cal_weeks if WEEK_ACCTS[w]])
print(f"attacker pool {len(Z_pool)} queries | reference {len(ref_idx)} queries (calibration weeks, benign accounts)")

def make_bank(Zref, h, seed):
    r = np.random.default_rng(seed); modes = r.standard_normal((M_MODES, Zref.shape[1])) / h / (2 * np.pi); m = M_MODES
    if USE_GPU:
        Mt = torch.as_tensor(modes, dtype=torch.float64, device=DEV)
        Gt = torch.zeros((m, m), dtype=torch.complex128, device=DEV); mut = torch.zeros(m, dtype=torch.complex128, device=DEV)
        for s0 in range(0, len(Zref), 16384):
            th = (2 * math.pi) * (torch.as_tensor(Zref[s0:s0 + 16384], dtype=torch.float64, device=DEV) @ Mt.T)
            E_ = torch.complex(torch.cos(th), -torch.sin(th)); Gt += E_.T @ E_.conj(); mut += E_.sum(0)
        G, mu = Gt.cpu().numpy(), mut.cpu().numpy()
    else:
        G = np.zeros((m, m), dtype=np.complex128); mu = np.zeros(m, dtype=np.complex128)
        for s0 in range(0, len(Zref), 4096):
            E_ = np.exp(-2j * np.pi * (Zref[s0:s0 + 4096] @ modes.T)); G += E_.T @ E_.conj(); mu += E_.sum(0)
    G /= len(Zref); mu /= len(Zref); Sig = G - np.outer(mu, mu.conj()); Sig = (Sig + Sig.conj().T) / 2
    ew, U = np.linalg.eigh(Sig); keep = ew > FLOOR * ew.max()
    return dict(modes=modes, mu=mu, Wk=U[:, keep].conj() / np.sqrt(ew[keep]), meff=int(keep.sum()))
t0 = time.time(); BANK = make_bank(Z_all[ref_idx], H_BW, SEED + 7); MEFF = BANK["meff"]; print(f"bank m_eff={MEFF} ({time.time()-t0:.0f}s)")
if USE_GPU:
    MODES_T = torch.as_tensor(BANK["modes"], dtype=torch.float64, device=DEV)
    MU_T = torch.as_tensor(BANK["mu"], dtype=torch.complex128, device=DEV); WK_T = torch.as_tensor(BANK["Wk"], dtype=torch.complex128, device=DEV)

def _fourier_numpy(pts, labels, nacc):
    import scipy.sparse as sp
    F_ = np.zeros((nacc, M_MODES), dtype=np.complex128)
    for s in range(0, len(pts), CHUNK):
        P, lab = pts[s:s + CHUNK], labels[s:s + CHUNK]; E_ = np.exp(-2j * np.pi * (P @ BANK["modes"].T))
        F_ += sp.csr_matrix((np.ones(len(P)), (lab, np.arange(len(P)))), shape=(nacc, len(P))) @ E_
    return F_
def _fourier_t(pts, labels, nacc):
    Fr = torch.zeros((nacc, M_MODES), dtype=torch.float64, device=DEV); Fi = torch.zeros_like(Fr)
    for s0 in range(0, len(pts), CHUNK):
        P = torch.as_tensor(pts[s0:s0 + CHUNK], dtype=torch.float64, device=DEV)
        ix = torch.as_tensor(labels[s0:s0 + CHUNK], dtype=torch.long, device=DEV); th = (2.0 * math.pi) * (P @ MODES_T.T)
        Fr.index_add_(0, ix, torch.cos(th)); Fi.index_add_(0, ix, -torch.sin(th))
    return torch.complex(Fr, Fi)
def geo_stats(pts, lab, cnt, Kc=0):
    if USE_GPU:
        c = torch.as_tensor(cnt, dtype=torch.float64, device=DEV); sq = torch.sqrt(c)
        Fw = ((_fourier_t(pts, lab, len(cnt)) - c[:, None] * MU_T[None, :]) / sq[:, None]) @ WK_T
        r2 = (Fw.abs() ** 2).mean(1).cpu().numpy(); lm = float(torch.linalg.eigvalsh(Fw.conj().T @ Fw / MEFF)[-1])
        ga = float(((Fw * sq[:, None]).sum(0).abs() ** 2).mean() / c.sum())
        su = float(((Fw[:Kc] * sq[:Kc, None]).sum(0).abs() ** 2).mean() / c[:Kc].sum()) if Kc else float("nan")
    else:
        sq = np.sqrt(cnt); Fw = ((_fourier_numpy(pts, lab, len(cnt)) - cnt[:, None] * BANK["mu"][None, :]) / sq[:, None]) @ BANK["Wk"]
        r2 = (np.abs(Fw) ** 2).mean(1); lm = float(np.linalg.eigvalsh(Fw.conj().T @ Fw / MEFF)[-1])
        ga = float((np.abs((Fw * sq[:, None]).sum(0)) ** 2).mean() / cnt.sum())
        su = float((np.abs((Fw[:Kc] * sq[:Kc, None]).sum(0)) ** 2).mean() / cnt[:Kc].sum()) if Kc else float("nan")
    return dict(lm=lm, r2=r2, ga=ga, su=su)
def hc_stat(x, null_sorted):
    M_ = len(null_sorted); p = np.sort((1 + M_ - np.searchsorted(null_sorted, x, side="left")) / (M_ + 1)); A_ = len(p); k = max(1, A_ // 2)
    i = np.arange(1, k + 1); pp = p[:k]; return float(np.max(np.sqrt(A_) * (i / A_ - pp) / np.sqrt(pp * (1 - pp))))
def calibrate(recs):
    q_ = lambda a: float(np.quantile(a, 1 - ALPHA_D)); ns = np.sort(np.concatenate([r["r2"] for r in recs]))
    return dict(tmax=q_([r["lm"] for r in recs]), tglo=q_([r["ga"] for r in recs]), tacct=q_([r["r2"].max() for r in recs]),
                thc=q_([hc_stat(r["r2"], ns) for r in recs]), ns=ns)
def decide(g, T_):
    d = dict(spec=g["lm"] > T_["tmax"], hc=hc_stat(g["r2"], T_["ns"]) > T_["thc"], glob=g["ga"] > T_["tglo"], acct=g["r2"].max() > T_["tacct"])
    d["suite"] = d["spec"] or d["hc"] or d["glob"]; return d
DKEYS = ["suite", "spec", "hc", "glob", "acct"]

def sample_window(weeks, A_, rng, shuffle=False):
    ok = [w for w in weeks if len(WEEK_ACCTS[w]) >= A_]
    if not ok: return None
    w = int(rng.choice(ok)); pick = rng.choice(len(WEEK_ACCTS[w]), A_, replace=False); parts = [WEEK_ACCTS[w][i] for i in pick]
    vols = np.array([len(p) for p in parts]); idx = np.concatenate(parts); lab = np.repeat(np.arange(A_), vols)
    if shuffle: idx = idx[rng.permutation(len(idx))]          # same prompts, account structure destroyed
    return Z_all[idx], lab, vols


# %% [markdown]
# ## Part 2 — campaigns

# %%
# ===== Cell 4: campaigns (broad = 1 of 10 clusters, narrow = 1 of 30) =====
from sklearn.cluster import MiniBatchKMeans
t0 = time.time()
km10 = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED, batch_size=4096, n_init=3).fit(Z_pool); CL10 = km10.labels_
km30 = MiniBatchKMeans(n_clusters=N_NARROW_CLUSTERS, random_state=SEED + 1, batch_size=4096, n_init=3).fit(Z_pool); CL30 = km30.labels_
s10, s30 = np.bincount(CL10, minlength=N_CLUSTERS), np.bincount(CL30, minlength=N_NARROW_CLUSTERS)
print(f"broad cluster sizes {sorted(s10.tolist(), reverse=True)}\nnarrow cluster sizes {sorted(s30.tolist(), reverse=True)} ({time.time()-t0:.0f}s)")
Nmax = max(N9); CAMP = {}
for tr_ in range(TRIALS9):
    rng = np.random.default_rng(SEED + 100 + tr_)
    b = np.flatnonzero(CL10 == int(np.argsort(-s10)[tr_ % 5])); nn_ok = np.flatnonzero(s30 >= sorted(N9)[len(N9) // 2])
    c30 = int(nn_ok[tr_ % len(nn_ok)]) if len(nn_ok) else int(np.argmax(s30)); n_ = np.flatnonzero(CL30 == c30)
    CAMP[("random", tr_)] = rng.choice(len(Z_pool), Nmax, replace=False)
    CAMP[("capability", tr_)] = b[rng.choice(len(b), Nmax, replace=len(b) < Nmax)]
    CAMP[("capability_narrow", tr_)] = n_[rng.permutation(len(n_))]                       # only budgets <= cluster size are used
    half = rng.choice(len(Z_pool), Nmax, replace=False)
    CAMP[("capability_pad50", tr_)] = np.where(np.arange(Nmax) % 2 == 0, CAMP[("capability", tr_)], half)
    print(f"trial {tr_}: broad cluster size {len(b)}, narrow cluster size {len(n_)}")


# %%
# ===== Cell 5: lagged references, bank-parametrised statistics, per-week calibration =====
def bank_t(b):
    if USE_GPU:
        b["MT"] = torch.as_tensor(b["modes"], dtype=torch.float64, device=DEV)
        b["MU_T"] = torch.as_tensor(b["mu"], dtype=torch.complex128, device=DEV); b["WK_T"] = torch.as_tensor(b["Wk"], dtype=torch.complex128, device=DEV)
    return b
BANK = bank_t(BANK)
def week_traffic(weeks, cap=100000, rng=np.random.default_rng(5)):
    parts = [np.concatenate(WEEK_ACCTS[w]) for w in weeks if w >= 0 and WEEK_ACCTS.get(w)]
    if not parts: return None
    idx = np.concatenate(parts); return idx if len(idx) <= cap else rng.choice(idx, cap, replace=False)
LAG_BANK = {}
def lag_bank(w):
    if w not in LAG_BANK:
        idx = week_traffic(range(w - LAG, w)); LAG_BANK[w] = bank_t(make_bank(Z_all[idx], H_BW, SEED + 7)) if idx is not None and len(idx) > 2000 else BANK
    return LAG_BANK[w]

def stats_b(pts, lab, cnt, B):
    if USE_GPU:
        Fr = torch.zeros((len(cnt), M_MODES), dtype=torch.float64, device=DEV); Fi = torch.zeros_like(Fr)
        for s0 in range(0, len(pts), CHUNK):
            P = torch.as_tensor(pts[s0:s0 + CHUNK], dtype=torch.float64, device=DEV); ix = torch.as_tensor(lab[s0:s0 + CHUNK], dtype=torch.long, device=DEV)
            th = (2.0 * math.pi) * (P @ B["MT"].T); Fr.index_add_(0, ix, torch.cos(th)); Fi.index_add_(0, ix, -torch.sin(th))
        c = torch.as_tensor(cnt, dtype=torch.float64, device=DEV); sq = torch.sqrt(c)
        Fw = ((torch.complex(Fr, Fi) - c[:, None] * B["MU_T"][None, :]) / sq[:, None]) @ B["WK_T"]
        return dict(lm=float(torch.linalg.eigvalsh(Fw.conj().T @ Fw / B["meff"])[-1]), r2=(Fw.abs() ** 2).mean(1).cpu().numpy(),
                    ga=float(((Fw * sq[:, None]).sum(0).abs() ** 2).mean() / c.sum()))
    import scipy.sparse as sp
    F_ = np.zeros((len(cnt), M_MODES), dtype=np.complex128)
    for s0 in range(0, len(pts), CHUNK):
        P, l_ = pts[s0:s0 + CHUNK], lab[s0:s0 + CHUNK]; E_ = np.exp(-2j * np.pi * (P @ B["modes"].T))
        F_ += sp.csr_matrix((np.ones(len(P)), (l_, np.arange(len(P)))), shape=(len(cnt), len(P))) @ E_
    sq = np.sqrt(cnt); Fw = ((F_ - cnt[:, None] * B["mu"][None, :]) / sq[:, None]) @ B["Wk"]
    return dict(lm=float(np.linalg.eigvalsh(Fw.conj().T @ Fw / B["meff"])[-1]), r2=(np.abs(Fw) ** 2).mean(1), ga=float((np.abs((Fw * sq[:, None]).sum(0)) ** 2).mean() / cnt.sum()))

def window_at(w, A_, rng):
    if len(WEEK_ACCTS[w]) < A_: return None
    pick = rng.choice(len(WEEK_ACCTS[w]), A_, replace=False); parts = [WEEK_ACCTS[w][i] for i in pick]
    vols = np.array([len(p) for p in parts]); return np.concatenate(parts), np.repeat(np.arange(A_), vols), vols
def win_stats(idx, lab, A_, w, scheme, extra=None):
    pts = Z_all[idx] if extra is None else np.vstack([extra[0], Z_all[idx]])
    labs = lab if extra is None else np.concatenate([extra[1], lab])
    return stats_b(pts, labs, np.bincount(labs, minlength=A_), lag_bank(w) if scheme == "rolling_lag" else BANK)

SCHEMES = ["fixed", "rolling", "rolling_lag"]
okw = [w for w in test_weeks if len(WEEK_ACCTS[w]) >= max(a for a, _ in AK9) and w - LAG >= 0]
TW = [okw[i] for i in np.linspace(0, len(okw) - 1, min(N_TW, len(okw))).astype(int)] if okw else []
print(f"test weeks used: {TW}; eligible accounts there: {[len(WEEK_ACCTS[w]) for w in TW]}")

# %% [markdown]
# ## Part A — calibration schemes

# %%
# ===== Part A: calibration per scheme and held-out null FPR =====
t0 = time.time(); THR = {}; rowsA = []
for A_, K_ in AK9:
    recs = []
    for t in range(N_CAL9):
        r = np.random.default_rng(SEED + 5_000_000 + 10 * A_ + t); cw = [w for w in cal_weeks if len(WEEK_ACCTS[w]) >= A_]
        if not cw: break
        w = int(r.choice(cw)); idx, lab, vols = window_at(w, A_, r); recs.append(win_stats(idx, lab, A_, w, "fixed"))
    THR[("fixed", A_)] = calibrate(recs)
    for tw in TW:
        for scheme in ["rolling", "rolling_lag"]:
            recs = []; prev = [w for w in range(tw - LAG, tw) if len(WEEK_ACCTS[w]) >= A_]
            for t in range(N_CAL_ROLL):
                r = np.random.default_rng(SEED + 6_000_000 + 1000 * tw + 10 * A_ + t + (7 if scheme == "rolling_lag" else 0))
                w = int(r.choice(prev)); idx, lab, vols = window_at(w, A_, r); recs.append(win_stats(idx, lab, A_, w, scheme))
            THR[(scheme, A_, tw)] = calibrate(recs)
    for scheme in SCHEMES:
        fp = 0; n = 0
        for tw in TW:
            for t in range(N_NULL_TW):
                r = np.random.default_rng(SEED + 7_000_000 + 1000 * tw + 10 * A_ + t); idx, lab, vols = window_at(tw, A_, r)
                T_ = THR[("fixed", A_)] if scheme == "fixed" else THR[(scheme, A_, tw)]
                fp += int(decide(win_stats(idx, lab, A_, tw, scheme), T_)["suite"]); n += 1
        rowsA.append(dict(A=A_, scheme=scheme, fpr=fp / max(n, 1), n=n))
        # [PAPER] Fig. 4a and App. Table 5 (last row): fixed 18.8% / 56.2%, rolling 5.0% / 24.7%, rolling + lag 3.4% / 28.7% (A = 100 / 500)
        print(f"A={A_:4d} {scheme:>12}: held-out suite FPR {fp/max(n,1):.3f} (nominal {ALPHA})  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Part B — paired campaign evaluation

# %%
# ===== Part B: paired campaign evaluation =====
t0 = time.time(); rowsB = []
for A_, K_ in AK9:
    for name in STRATS10:
        for N in N9:
            if name == "capability_narrow" and N > min(len(CAMP[(name, t)]) for t in range(TRIALS9)): continue
            agg = {s: dict(att=0, base=0, attrib=0) for s in SCHEMES}; n = 0; shares = []
            for tw in TW:
                for tr_ in range(TRIALS9):
                    campZ = Z_pool[CAMP[(name, tr_)][:N]]
                    for w in range(W_TW):
                        r = np.random.default_rng(SEED + 8_000_000 + 1000 * tw + 100 * tr_ + w + A_ + N); idx, lab, vols = window_at(tw, A_, r)
                        keep = lab >= K_; n_per = N // K_; cz = campZ[r.permutation(N)][: K_ * n_per]
                        extra = (cz, np.arange(K_ * n_per) % K_); shares.append(len(cz) / (len(cz) + keep.sum())); n += 1
                        for s in SCHEMES:
                            T_ = THR[("fixed", A_)] if s == "fixed" else THR[(s, A_, tw)]
                            d_base = decide(win_stats(idx, lab, A_, tw, s), T_)["suite"]
                            d_att = decide(win_stats(idx[keep], lab[keep], A_, tw, s, extra), T_)["suite"]
                            agg[s]["att"] += int(d_att); agg[s]["base"] += int(d_base); agg[s]["attrib"] += int(d_att and not d_base)
            row = dict(A=A_, K=K_, strategy=name, N=N, n=n, share=float(np.mean(shares)))
            for s in SCHEMES:
                row[f"{s}_power"] = agg[s]["att"] / n; row[f"{s}_excess"] = (agg[s]["att"] - agg[s]["base"]) / n; row[f"{s}_attrib"] = agg[s]["attrib"] / n
            rowsB.append(row)
            # [PAPER] App. Table 5 (cols "Share", "Fixed", "Rolling", "Rolling + lag" = `share`, `{scheme}_attrib`); App. F: colluders 46–94% of the window
            print(f"A={A_:4d} {name:>18} N={N:6d} share {row['share']:.2f} | " + " ".join(f"{s}: attrib {row[s+'_attrib']:.2f} excess {row[s+'_excess']:+.2f}" for s in SCHEMES) + f"  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Part C — predictions

# %%
# ===== Part C: predictions =====
get = lambda A_, s_, N: next((r for r in rowsB if r["A"] == A_ and r["strategy"] == s_ and r["N"] == N), None)
# [PAPER] Prediction log (Step 10): P10.1 Partial (A = 100 yes; A = 500 no)
for r in rowsA: print(f"P10.1 A={r['A']} {r['scheme']}: FPR {r['fpr']:.3f}")
mid = sorted(N9)[len(N9) // 2]
for A_, K_ in AK9:
    a = get(A_, "capability", mid); b = get(A_, "capability_narrow", mid)
    # [PAPER] Prediction log (Step 10): P10.2 Partial (lagged 0.21, 0.29); P10.3 Pass (0.94; 0.71)
    if a: print(f"P10.2 A={A_} capability N={mid}: attrib " + " ".join(f"{s} {a[s+'_attrib']:.2f}" for s in SCHEMES) + "  (pass if all <= 0.2)")
    if b: print(f"P10.3 A={A_} capability_narrow N={mid}: rolling_lag attrib {b['rolling_lag_attrib']:.2f} (pass if >= 0.5)")
    c, d = get(A_, "capability", max(N9)), get(A_, "capability_pad50", max(N9))
    # [PAPER] Prediction log (Step 10): P10.5 Partial (A = 100 yes)
    if c and d: print(f"P10.5 A={A_} N={max(N9)}: pad50 drop " + " ".join(f"{s} {c[s+'_attrib']-d[s+'_attrib']:+.2f}" for s in SCHEMES) + "  (pass if all >= 0.5)")
hits = [r for r in rowsB if r["rolling_lag_attrib"] >= 0.5]
# [PAPER] Prediction log (Step 10): P10.4 Pass
print("P10.4 cells with rolling_lag attrib >= 0.5: " + "; ".join(f"A={r['A']} {r['strategy']} N={r['N']} share={r['share']:.2f}" for r in hits))
RESULTS10 = dict(config=dict(QUICK=QUICK, AK=AK9, N=N9, TW=[int(w) for w in TW], ALPHA=ALPHA, LAG=LAG), A=rowsA, B=rowsB)

# %% [markdown]
# ## Figure

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
cols = dict(fixed="tab:gray", rolling="tab:blue", rolling_lag="tab:red"); mk = dict(random="x", capability="o", capability_narrow="s", capability_pad50="^")
for r in rowsB:
    for s in SCHEMES: ax[0].scatter(r["share"], r[s + "_attrib"], color=cols[s], marker=mk[r["strategy"]], s=30, alpha=.8)
ax[0].axvline(0.5, color="k", ls=":"); ax[0].set_xlabel("campaign share of window traffic"); ax[0].set_ylabel("attributable power")
ax[0].set_title("P10.4: does detection need dominance? (colour = scheme, marker = strategy)")
for s, c_ in cols.items(): ax[0].scatter([], [], color=c_, label=s)
for k_, m_ in mk.items(): ax[0].scatter([], [], color="k", marker=m_, label=k_)
ax[0].legend(fontsize=7)
xs = np.arange(len(rowsA)); ax[1].bar(xs, [r["fpr"] for r in rowsA], color=[cols[r["scheme"]] for r in rowsA])
ax[1].axhline(ALPHA, color="k", ls=":"); ax[1].set_xticks(xs); ax[1].set_xticklabels([f"A={r['A']}\n{r['scheme']}" for r in rowsA], fontsize=8)
ax[1].set_ylabel("held-out null FPR"); ax[1].set_title("P10.1: drift-aware calibration")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/step10_figure.png", dpi=130); plt.show()

# %% [markdown]
# ## VERDICT REPORT — copy into the chat

# %%
print("=" * 86); print("STEP 10 VERDICT REPORT"); print("=" * 86); print("config:", json.dumps(RESULTS10["config"]))
for r in RESULTS10["A"]: print(f"[A] A={r['A']:4d} {r['scheme']:>12} FPR={r['fpr']:.3f} n={r['n']}")
for r in RESULTS10["B"]:
    print(f"[B] A={r['A']:4d} {r['strategy']:>18} N={r['N']:6d} share={r['share']:.2f} " + " ".join(f"{s}:pow {r[s+'_power']:.2f}/exc {r[s+'_excess']:+.2f}/att {r[s+'_attrib']:.2f}" for s in SCHEMES))
print("=" * 86)
with open(f"{OUTDIR}/step10_results.json", "w") as fh: json.dump(RESULTS10, fh, indent=1, default=float)
