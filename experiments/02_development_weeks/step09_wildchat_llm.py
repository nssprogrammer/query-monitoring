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
# Run on Kaggle (see "Kaggle setup:" line below for hardware/time). Seeds are fixed in the configuration cell;
# re-runs vary by a few points (the paper reports run-to-run spread up to ~0.07-0.14 for weekly detection).
#
# | Paper location | Quantity | Value in paper | Where computed here |
# |---|---|---|---|
# | §5 "Calibration breaks first"; App. D.1; Prediction log (Step 9, "Fixed calibration transfers to later weeks") | false alarms on later (held-out) null weeks of thresholds set once on early weeks, nominal 1% | 12% (Prediction log: "Fail (12%)") | Part A cell, `row["suite"]` in print "held-out-week FPR ... suite="; Part C print "P9.1" |
# | Prediction log (Step 9, "Real users structured: λmax real/shuffled ≥ 2") | median λ_max of real windows / same windows with prompts shuffled across accounts | Pass (4.3–5.1) | Part A cell, print "(ratio ...)" from `lm_real`/`lm_shuf`; Part C print "P9.2" |
# | Prediction log (Step 9, "Capability campaigns detected at 5,000 and 20,000") | suite power of `capability`/`capability_kcenter` campaigns | Partial (20,000 via global only) | Part B cell, `suite` and `glob` columns of the per-row print; Part C print "P9.3" |
# | Prediction log (Step 9, "Random campaigns ≤ 2× nominal") | suite power of `random` campaigns | Fail (8–21% floor) | Part B cell, `random` rows; Part C print "P9.4" |
# | Prediction log (Step 9, "Padding costs ≥ 0.3 power at 5,000") | power cost of 50% padding, `capability` vs `capability_pad50` | Fail as specified; 0.79 at 20,000 | Part C print "P9.5" (N = 5,000); the N = 20,000 value from Part B rows `capability` vs `capability_pad50` at N = 20000 (not printed as a difference) |
# | §2 "Data" | development data: hashed IPs and weeks of WildChat-1M | 98,331 hashed IPs over 56 weeks | plausibly the account/week counts printed at the end of Cell 2 (same loading code in Steps 9–12); not isolated |
# | §2 "Data"; App. Table 16 (Embedding, Fourier modes, Whitening floor) | attacker pool share; MiniLM → PCA rescaled to total variance 4; m = 400 complex modes, h = 4, floor 10−4 | 20%; "rescaled to total variance 4"; "m = 400"; "h = 4"; "10−4" | Cell 1 configuration (`POOL_FRAC`, `D_DEF`, `H_BW`, `M_MODES`, `FLOOR`) and the PCA step in Cell 2 |
#
# Not reported in the paper: per-detector held-out FPRs other than the suite (spec/hc/glob/acct), the S_all
# real-vs-shuffled medians, the `broad_kcenter` (P9.6, exploratory) and `capability_kcenter` cells individually,
# the per-N power curves and the figure. The consequences recorded in Prediction log (drift-aware calibration;
# "Detection needs dominance") motivated Step 10.

# %% [markdown]
# # Step 9 — Real benign LLM traffic (WildChat-1M) vs simulated distillation campaigns
# **Kaggle setup:** GPU + Internet ON. *Add-ons → Secrets*: add your Hugging Face token as **`HF_TOKEN`** and tick it for this notebook. **Run All** (≈30–50 min; first run downloads WildChat-1M and embeds ≈1M English user turns, then caches). `QUICK=True` uses 30k conversations.
#
# **Accounts** = hashed IPs; **windows** = calendar weeks; **queries** = English user turns, embedded with `all-MiniLM-L6-v2` → PCA-32. IPs are split 80/20: the 80% are benign users, the disjoint 20% form the attacker's pool of natural prompts. **Temporal holdout:** detectors are calibrated on the first 60% of weeks and evaluated on the rest. Per-account weekly volume capped at 500 (shared/NAT IPs).
#
# **Attacker campaigns** (N prompts split equally over K accounts that replace K benign accounts in a held-out week):
# `random` natural prompts · `capability` = random prompts from one semantic cluster (k-means, 10 clusters ≈ broad capabilities) · `capability_kcenter` = farthest-point coverage *inside* that cluster · `broad_kcenter` = farthest-point coverage of the whole pool · `capability_pad50` = half capability, half random.
#
# **Detectors** (h = 4, as before): spectral upper edge, Higher Criticism, global stream, per-account max; suite = spectral OR HC OR global at α/3 each.
#
# | ID | Locked prediction |
# |---|---|
# | **P9.1** | Temporal holdout: suite FPR on held-out weeks ≤ 2% (nominal ≤ 1%) |
# | **P9.2** | Real benign traffic is coordinated: median λ_max of real windows ≥ 2× that of the same windows with prompts shuffled across accounts |
# | **P9.3** | `capability` and `capability_kcenter`: suite power ≥ 0.8 at N = 20,000 for both (A, K); ≥ 0.5 at N = 5,000 for (100, 20) |
# | **P9.4** | `random` campaigns: suite power ≤ 2× nominal |
# | **P9.5** | `capability_pad50` loses ≥ 0.3 power relative to `capability` at N = 5,000 |
# | **P9.6** | `broad_kcenter`: exploratory (no prediction) |

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
# [PAPER] App. Table 16: PCA-32 of MiniLM, h = 4, m = 400 complex modes, whitening floor 10−4; nominal 1% (§5 "Calibration breaks first")
D_DEF, H_BW, M_MODES, ALPHA, FLOOR = 32, 4.0, 400, 0.01, 1e-4
ALPHA_D = ALPHA / 3
# [PAPER] §2 "Data": a random 20% of IPs supplies the attacker's pool of natural prompts (POOL_FRAC)
VOL_CAP, MIN_PER_ACCT, POOL_FRAC, CAL_FRAC, N_CLUSTERS = 500, 3, 0.2, 0.6, 10
STRATS9 = ["random", "capability", "capability_kcenter", "broad_kcenter", "capability_pad50"]
if QUICK:
    N_CONV, AK9, N9, N_CAL9, N_TEST9, TRIALS9, W9 = 30000, [(50, 10)], [300, 1000], 80, 40, 1, 2
else:
    N_CONV, AK9, N9, N_CAL9, N_TEST9, TRIALS9, W9 = None, [(100, 20), (500, 100)], [1000, 5000, 20000], 600, 300, 3, 8
CACHE9 = f"{OUTDIR}/step9_wildchat_{'quick' if QUICK else 'full'}{'_synth' if SYNTHETIC_TEST else ''}.npz"
print(f"backend {'GPU ' + torch.cuda.get_device_name(0) if USE_GPU else 'CPU'} | QUICK={QUICK} SYNTHETIC_TEST={SYNTHETIC_TEST} | cache {CACHE9}")


# %% [markdown]
# ## Part 0 — WildChat-1M: English user turns → MiniLM → PCA-32 (cached)

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
# [PAPER] §2 "Data": 98,331 hashed IPs over 56 weeks (plausibly these printed counts; not isolated)
n_w = int(WEEK.max()) + 1; print(f"{len(Z_all)} queries, {ACCT.max()+1} accounts, {n_w} weeks")

# %% [markdown]
# ## Part 1 — accounts, weeks, whitening, statistics

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
# ## Part 2 — distillation campaigns

# %%
# ===== Cell 4: distillation campaigns from the attacker pool =====
from sklearn.cluster import MiniBatchKMeans
t0 = time.time(); km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED, batch_size=4096, n_init=3).fit(Z_pool)
CL = km.labels_; sizes = np.bincount(CL, minlength=N_CLUSTERS); print(f"pool clusters (capabilities) sizes: {sorted(sizes.tolist(), reverse=True)} ({time.time()-t0:.0f}s)")
def fps(X, N, rng):
    if USE_GPU:
        Xt = torch.as_tensor(X, dtype=torch.float32, device=DEV); i0 = int(rng.integers(len(X))); d2 = ((Xt - Xt[i0]) ** 2).sum(1)
        sel = torch.empty(N, dtype=torch.long, device=DEV); sel[0] = i0
        for t in range(1, N):
            i = torch.argmax(d2); sel[t] = i; d2 = torch.minimum(d2, ((Xt - Xt[i]) ** 2).sum(1))
        return sel.cpu().numpy()
    sel = [int(rng.integers(len(X)))]; d2 = ((X - X[sel[0]]) ** 2).sum(1)
    for _ in range(N - 1):
        i = int(np.argmax(d2)); sel.append(i); d2 = np.minimum(d2, ((X - X[i]) ** 2).sum(1))
    return np.array(sel)
Nmax = max(N9); CAMP = {}
big = np.argsort(-sizes)
for tr_ in range(TRIALS9):
    rng = np.random.default_rng(SEED + 100 + tr_); c = int(big[tr_ % max(1, min(5, N_CLUSTERS))]); members = np.flatnonzero(CL == c)
    CAMP[("random", tr_)] = rng.choice(len(Z_pool), Nmax, replace=False)
    CAMP[("capability", tr_)] = members[rng.choice(len(members), min(Nmax, len(members)), replace=len(members) < Nmax)]
    CAMP[("capability_kcenter", tr_)] = members[fps(Z_pool[members], min(Nmax, len(members)), rng)] if len(members) >= Nmax else CAMP[("capability", tr_)]
    CAMP[("broad_kcenter", tr_)] = fps(Z_pool, Nmax, rng)
    half = rng.choice(len(Z_pool), Nmax, replace=False)
    CAMP[("capability_pad50", tr_)] = np.where(np.arange(Nmax) % 2 == 0, CAMP[("capability", tr_)][:Nmax], half)
    print(f"trial {tr_}: capability cluster {c} (size {len(members)}) ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Part A — temporal-holdout calibration and real benign coordination

# %%
# ===== Part A: temporal-holdout calibration and real benign coordination =====
t0 = time.time(); CAL = {}; rowsA = []
for A_, K_ in AK9:
    recs = []
    for t in range(N_CAL9):
        win = sample_window(cal_weeks, A_, np.random.default_rng(SEED + 1_000_000 + 10 * A_ + t))
        if win is None: break
        recs.append(geo_stats(*win))
    if len(recs) < 20: print(f"A={A_}: too few eligible calibration weeks; skipped"); continue
    CAL[A_] = T_ = calibrate(recs); fp = {k: 0 for k in DKEYS}; n = 0; lm_real, lm_shuf, ga_real, ga_shuf = [], [], [], []
    for t in range(N_TEST9):
        r = np.random.default_rng(SEED + 2_000_000 + 10 * A_ + t); win = sample_window(test_weeks, A_, r)
        if win is None: break
        g = geo_stats(*win); d = decide(g, T_); n += 1
        for k in DKEYS: fp[k] += int(d[k])
        lm_real.append(g["lm"]); ga_real.append(g["ga"])
        if t < 100:
            gs = geo_stats(*sample_window(test_weeks, A_, np.random.default_rng(SEED + 2_000_000 + 10 * A_ + t), shuffle=True))
            lm_shuf.append(gs["lm"]); ga_shuf.append(gs["ga"])
    row = dict(A=A_, n_cal=len(recs), n_test=n, **{k: fp[k] / max(n, 1) for k in DKEYS},
               lm_real=float(np.median(lm_real)), lm_shuf=float(np.median(lm_shuf)), ga_real=float(np.median(ga_real)), ga_shuf=float(np.median(ga_shuf)))
    rowsA.append(row)
    # [PAPER] §5, App. D.1, Prediction log (Step 9): fixed thresholds fired on 12% of later null weeks (suite, nominal 1%)
    # [PAPER] Prediction log (Step 9): λmax real/shuffled ratio 4.3–5.1 (the "(ratio ...)" part of this print)
    print(f"A={A_}: held-out-week FPR " + " ".join(f"{k}={row[k]:.3f}" for k in DKEYS) +
          f" | median lam_max real {row['lm_real']:.2f} vs shuffled {row['lm_shuf']:.2f} (ratio {row['lm_real']/row['lm_shuf']:.2f}); "
          f"S_all real {row['ga_real']:.2f} vs shuffled {row['ga_shuf']:.2f}  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Part B — campaigns injected into held-out weeks

# %%
# ===== Part B: distillation campaigns injected into held-out weeks =====
t0 = time.time(); rowsB = []
for A_, K_ in AK9:
    if A_ not in CAL: continue
    for name in STRATS9:
        for N in N9:
            hits = {k: 0 for k in DKEYS}; n = 0; Su = []
            for tr_ in range(TRIALS9):
                campZ = Z_pool[CAMP[(name, tr_)][:N]]
                for w in range(W9):
                    r = np.random.default_rng(SEED + 3_000_000 + 1000 * tr_ + w + A_ + N); win = sample_window(test_weeks, A_, r)
                    if win is None: continue
                    pts, lab, vols = win; keep = lab >= K_; n_per = N // K_
                    pts = np.vstack([campZ[r.permutation(N)][: K_ * n_per], pts[keep]]); lab = np.concatenate([np.arange(K_ * n_per) % K_, lab[keep]])
                    g = geo_stats(pts, lab, np.bincount(lab, minlength=A_), K_); d = decide(g, CAL[A_]); n += 1; Su.append(g["su"])
                    for k in DKEYS: hits[k] += int(d[k])
            row = dict(A=A_, K=K_, strategy=name, N=N, n_per=N // K_, n=n, S_un=float(np.mean(Su)), **{k: hits[k] / max(n, 1) for k in DKEYS})
            rowsB.append(row)
            # [PAPER] Prediction log (Step 9): capability detected at 20,000 via global only; random floor 8–21%; padding "0.79 at 20,000" (P9.5)
            print(f"A={A_:4d} K={K_:3d} {name:>18} N={N:6d} (n/acct {N//K_:5d}) S_un={row['S_un']:7.2f} | " + " ".join(f"{k}={row[k]:.2f}" for k in DKEYS) + f"  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Part C — predictions

# %%
# ===== Part C: predictions =====
get = lambda A_, s_, N: next((r for r in rowsB if r["A"] == A_ and r["strategy"] == s_ and r["N"] == N), None)
# [PAPER] Prediction log (Step 9): P9.1 Fail (12%); P9.2 Pass (4.3–5.1)
for r in rowsA: print(f"P9.1 A={r['A']}: held-out suite FPR {r['suite']:.3f} (pass if <= 0.02) | P9.2 lam_max ratio real/shuffled {r['lm_real']/r['lm_shuf']:.2f} (pass if >= 2)")
for A_, K_ in AK9:
    for s_ in ["capability", "capability_kcenter"]:
        a, b = get(A_, s_, max(N9)), get(A_, s_, sorted(N9)[len(N9) // 2])
        # [PAPER] Prediction log (Step 9): P9.3 Partial (20,000 via global only)
        if a: print(f"P9.3 A={A_} {s_}: N={max(N9)} power {a['suite']:.2f} (>=0.8) | N={b['N']} power {b['suite']:.2f}")
    rr = [r for r in rowsB if r["A"] == A_ and r["strategy"] == "random"]
    # [PAPER] Prediction log (Step 9): P9.4 Fail (8–21% floor)
    if rr: print(f"P9.4 A={A_} random: max suite power {max(r['suite'] for r in rr):.3f} (pass if <= 0.02)")
    mid = sorted(N9)[len(N9) // 2]; a, b = get(A_, "capability", mid), get(A_, "capability_pad50", mid)
    # [PAPER] Prediction log (Step 9): P9.5 Fail as specified (N = 5,000); 0.79 at 20,000 (from Part B rows, not printed here)
    if a and b: print(f"P9.5 A={A_} N={mid}: capability {a['suite']:.2f} vs pad50 {b['suite']:.2f} (drop {a['suite']-b['suite']:+.2f}; pass if >= 0.3)")
RESULTS9 = dict(config=dict(QUICK=QUICK, AK9=AK9, N9=N9, MEFF=MEFF, n_queries=int(len(Z_all)), n_weeks=int(n_w)), A=rowsA, B=rowsB)

# %% [markdown]
# ## Figure

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, len(AK9), figsize=(8 * len(AK9), 5), squeeze=False)
for i, (A_, K_) in enumerate(AK9):
    for s_ in STRATS9:
        rr = sorted([r for r in rowsB if r["A"] == A_ and r["strategy"] == s_], key=lambda r: r["N"])
        if rr: ax[0, i].plot([r["N"] for r in rr], [r["suite"] for r in rr], "o-", label=s_)
    ax[0, i].set_xscale("log"); ax[0, i].set_ylim(-0.03, 1.03); ax[0, i].set_xlabel("campaign size N (prompts)"); ax[0, i].set_ylabel("suite power")
    ax[0, i].set_title(f"WildChat held-out weeks, A={A_}, K={K_}"); ax[0, i].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/step9_figure.png", dpi=130); plt.show()

# %% [markdown]
# ## VERDICT REPORT — copy into the chat

# %%
print("=" * 86); print("STEP 9 VERDICT REPORT"); print("=" * 86); print("config:", json.dumps(RESULTS9["config"]))
for r in RESULTS9["A"]: print("[A]", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
for r in RESULTS9["B"]: print(f"[B] A={r['A']:4d} K={r['K']:3d} {r['strategy']:>18} N={r['N']:6d} S_un={r['S_un']:.2f} " + " ".join(f"{k}={r[k]:.2f}" for k in DKEYS))
print("=" * 86)
with open(f"{OUTDIR}/step9_results.json", "w") as fh: json.dump(RESULTS9, fh, indent=1, default=float)
