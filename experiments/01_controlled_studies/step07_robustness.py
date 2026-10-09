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
# | App. P (Step 7), row "Only whole-window resampling calibrates" (P7.1) | suite false-positive rate on realistic null windows, window-resampled calibration | Pass (suite 0.0–0.6%) | Part A print "calibrated by window_resample" (`suite=`) / VERDICT `[A]` lines |
# | App. E 'The dose–response study' | window-resampled suite false alarms at s = 1; spectral and global tests under independent-account resampling | 0.6%; 31%; 100% | Part A prints (`window_resample` suite; `acct_resample` spec / glob). Plausibly from here: Step 7's realistic windows have the structure of Step 8's s = 1, and account resampling exists only in this script |
# | App. P (Step 7), row "Evading the suite needs padding f ≥ 0.75" (P7.2) | suite detection vs padding fraction f | Fail (no detection even unpadded) | Part B print "suite: equal=..." at f=0.00 / VERDICT `[B]` lines |
# | App. P (Step 7), row "Heterogeneous volumes cost ≤ 0.2 power" (P7.3) | suite power, heterogeneous colluder volumes | Untestable (power 0) | Part B `hetero` variant |
# | App. P (Step 7), row "Time-splitting evades; aggregation restores" (P7.4) | any-window suite detection, split / aggregated | Untestable (power 0) | Part B `split2` / `split4` / `agg4` variants |
# | App. P (Step 7), row "Upper-edge rule predicts power at A ≤ 10^4" (P7.5) | agreement of rule (S̄ − 1)√m_eff/√A > 1.36 with spectral power > 0.5 | Fail (10/24 cells) | Part C print "P7.5 rule ... agreement" (fraction of the 24 cells) |
#
# Not reported in the paper: the iid calibration scheme's false-positive rates, per-detector rates other than those above,
# fidelity under padding (`FIDP`), the Part C powers per detector and global-stream power at A = 10,000, the unused
# Step-6 evaluation cell, and the figure `step7_figure.png`.

# %% [markdown]
# # Step 7 — Robustness: correlated benign traffic, adaptive attackers, production-scale A
# **Kaggle:** GPU + Internet → **Run All** (≈60–80 min; reuses `step5_cache_full.npz` if present). `QUICK=True` first.
#
# Answers the three empirical concerns of the review. One bandwidth (h = 4); four detectors: spectral upper edge, Higher Criticism (HC), per-account max, global stream; the **suite** = spectral OR HC OR global, each at α/3, so suite FPR ≤ 1%.
#
# **Realistic benign windows.** Account volumes lognormal (median 100, σ = 1). 30% of accounts sit in *topic clusters* of 10 (shared script / tutorial): half their queries come from one narrow region. Each window has a *trend*: 20% of accounts send 20% of their queries to one trending region.
#
# Splitting over T windows is scored as detection in *any* of the T windows (per-window false-positive rate unchanged; over T windows it compounds to about T × 1%). The defender's aggregated window is calibrated on its own null.
#
# **Calibration schemes compared:** (i) iid benign users (what Steps 5–6 assumed), (ii) resampling historical *accounts* independently, (iii) resampling whole historical *windows*.
#
# | ID | Locked prediction |
# |---|---|
# | **P7.1** | On realistic null windows: (i) inflates FPR for every detector; (ii) calibrates per-account and HC but leaves spectral and global inflated; (iii) keeps every detector ≤ 1.5× nominal and the suite ≤ 1.5% |
# | **P7.2** | Padding with natural queries: suite detection falls with padding fraction f; evading the suite (power ≤ 0.1) requires f ≥ 0.75, where k-center's fidelity gain over random is ≤ 0.5 points |
# | **P7.3** | Heterogeneous colluder volumes reduce suite power by ≤ 0.2 |
# | **P7.4** | Splitting a campaign over T = 4 windows drops any-window suite detection below 0.5 at A = 400, N = 5,000; aggregating the T windows into one (defender's longer window) restores it to within 0.1 of T = 1 |
# | **P7.5** | At A up to 10,000 (window-calibrated), the rule (S̄ − 1)√m_eff/√A > 1.36 predicts spectral power > 0.5 in ≥ 80% of cells; global-stream power ≤ 0.1 everywhere at A = 10,000 |

# %%
# ===== Cell 1: configuration =====
import math, json, time, os
import numpy as np
from scipy.stats import rankdata, spearmanr
QUICK = False
SEED = 5051
OUTDIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
try:
    import torch
    HAS_TORCH = True; DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    HAS_TORCH, DEV = False, None
USE_GPU = HAS_TORCH and DEV.type == "cuda"
ALLOW_CPU = False   # set True only for a CPU smoke test
if not USE_GPU and not ALLOW_CPU:
    raise RuntimeError("No GPU visible to PyTorch. Kaggle: Settings -> Accelerator -> GPU T4 x2, then restart the session.")
CHUNK = 65536 if USE_GPU else 8192
D_DEF, HS, M_MODES, ALPHA, FLOOR = 32, [4.0], 400, 0.01, 1e-4
STRATS = ["random", "entropy", "kcenter"]
if QUICK:
    EPOCHS, N_REF, N_NULLB, POOL, N_FIT = 2, 40000, 30000, 12000, 10000
    SEED_N, TARGETS, CHECKS, TRIALS, N_NULL, W = 300, [600, 1000], [600, 1000], 1, 60, 2
    AK = [(25, 5), (100, 20)]
else:
    EPOCHS, N_REF, N_NULLB, POOL, N_FIT = 15, 300000, 200000, 60000, 50000
    SEED_N, TARGETS, CHECKS, TRIALS, N_NULL, W = 1000, [2000, 3500, 5000, 7500, 10000], [2000, 5000, 10000], 3, 1000, 8
    AK = [(25, 5), (100, 5), (100, 20), (400, 20), (400, 80)]
AS = sorted({a for a, _ in AK})
CACHE = f"{OUTDIR}/step5_cache_{'quick' if QUICK else 'full'}.npz"
print(f"backend: {'GPU ' + torch.cuda.get_device_name(0) if USE_GPU else 'CPU'} | QUICK={QUICK} | cache={CACHE}")
# ---- Step 7 parameters
ALPHA_D = ALPHA / 3                      # per detector -> suite FPR <= 1% by the union bound
N0_MED, VOL_SIGMA = 100, 1.0
CLUSTER_FRAC, CLUSTER_SIZE, TOPIC_Q = 0.3, 10, 0.5
TREND_FRAC, TREND_Q = 0.2, 0.2
R_TOPICS, NN_TOPIC = 400, 1500
if QUICK:
    A_STRESS, N_CAL, N_TEST, HIST_WIN = [50], 60, 40, 20
    A_ADAPT, K_ADAPT, N_ADAPT, FS, TS, W7 = [50], 10, [600], [0.0, 0.75], [1, 2], 2
    A_SCALE, K_SCALE, N_SCALE, N_CAL_SCALE = [200], [10], [1000], 40
    R_TOPICS, NN_TOPIC = 60, 300
else:
    A_STRESS, N_CAL, N_TEST, HIST_WIN = [100, 400], 1000, 500, 200
    A_ADAPT, K_ADAPT, N_ADAPT, FS, TS, W7 = [100, 400], 20, [2000, 5000], [0.0, 0.5, 0.75, 0.9], [1, 2, 4], 8
    A_SCALE, K_SCALE, N_SCALE, N_CAL_SCALE = [1000, 10000], [20, 100], [2000, 5000, 10000], 800
print(f"Step 7: A_STRESS={A_STRESS} A_ADAPT={A_ADAPT} A_SCALE={A_SCALE} N_CAL={N_CAL} N_CAL_SCALE={N_CAL_SCALE}")

# %% [markdown]
# ## Part 0 — victim & embeddings (cached)

# %%
# ===== Cell 2: data, victim, attacker encoder, embeddings =====
if os.path.exists(CACHE):
    _c = np.load(CACHE); globals().update({k: _c[k] for k in _c.files}); VICTIM_ACC = float(VICTIM_ACC)
    print(f"loaded cache: victim acc {VICTIM_ACC:.4f}, Z_ref {Z_ref.shape}, Z_pool {Z_pool.shape}")
else:
    import torch.nn as nn, torch.nn.functional as F, torchvision
    from torchvision.models import resnet18, ResNet18_Weights
    torch.backends.cudnn.benchmark = True
    t0 = time.time(); DATA = os.path.join(OUTDIR, "data")
    tr = torchvision.datasets.CIFAR10(DATA, train=True, download=True); te = torchvision.datasets.CIFAR10(DATA, train=False, download=True)
    X_tr = torch.tensor(tr.data).permute(0, 3, 1, 2).contiguous().to(DEV); y_tr = torch.tensor(tr.targets, device=DEV)
    X_te = torch.tensor(te.data).permute(0, 3, 1, 2).contiguous().to(DEV); y_te = np.array(te.targets)
    VICTIM_IDX, BASES = np.arange(25000), np.arange(25000, 50000)
    MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=DEV).view(1, 3, 1, 1); STD = torch.tensor([0.2470, 0.2435, 0.2616], device=DEV).view(1, 3, 1, 1)
    IMEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); ISTD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    AR = torch.arange(32, device=DEV)

    def make_batch(X, base, ox, oy, flip):     # reflect-pad 4, crop 32x32 at (oy, ox), optional h-flip -> float in [0,1]
        xb = X[torch.as_tensor(base, device=DEV, dtype=torch.long)].float()
        pad = F.pad(xb, (4, 4, 4, 4), mode="reflect"); B = xb.shape[0]
        rows = torch.as_tensor(oy, device=DEV, dtype=torch.long)[:, None] + AR[None, :]
        cols = torch.as_tensor(ox, device=DEV, dtype=torch.long)[:, None] + AR[None, :]
        bi = torch.arange(B, device=DEV)[:, None, None, None]; ci = torch.arange(3, device=DEV)[None, :, None, None]
        out = pad[bi, ci, rows[:, None, :, None], cols[:, None, None, :]]
        f = torch.as_tensor(np.asarray(flip), device=DEV, dtype=torch.bool)
        return torch.where(f[:, None, None, None], out.flip(-1), out) / 255.0
    def draw(n, rng, bases):
        return dict(base=rng.choice(bases, n), ox=rng.integers(0, 9, n), oy=rng.integers(0, 9, n), flip=rng.random(n) < 0.5)
    def victim_net():
        m = resnet18(num_classes=10); m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity(); return m
    def victim_feats(m, x):
        x = m.relu(m.bn1(m.conv1(x))); x = m.maxpool(x)
        x = m.layer4(m.layer3(m.layer2(m.layer1(x)))); return torch.flatten(m.avgpool(x), 1)

    # sanity: un-augmented crop reproduces the original image
    chk = make_batch(X_te, np.arange(4), np.full(4, 4), np.full(4, 4), np.zeros(4, bool))
    assert torch.allclose(chk * 255.0, X_te[:4].float()), "augmentation crop check failed"

    # ---- train victim
    rng = np.random.default_rng(SEED)
    victim = victim_net().to(DEV).to(memory_format=torch.channels_last)
    BS = 256; spe = len(VICTIM_IDX) // BS
    opt = torch.optim.SGD(victim.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.2, total_steps=EPOCHS * spe, pct_start=0.25)
    try: scaler = torch.amp.GradScaler("cuda", enabled=USE_GPU)
    except Exception: scaler = torch.cuda.amp.GradScaler(enabled=USE_GPU)
    @torch.no_grad()
    def test_acc():
        victim.eval(); pr = []
        for s in range(0, 10000, 1000):
            x = make_batch(X_te, np.arange(s, s + 1000), np.full(1000, 4), np.full(1000, 4), np.zeros(1000, bool))
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_GPU):
                pr.append(victim(((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)).argmax(1).cpu().numpy())
        return float((np.concatenate(pr) == y_te).mean())
    for ep in range(EPOCHS):
        victim.train(); perm = rng.permutation(VICTIM_IDX)
        for s in range(spe):
            idx = perm[s * BS:(s + 1) * BS]
            xb = ((make_batch(X_tr, idx, rng.integers(0, 9, BS), rng.integers(0, 9, BS), rng.random(BS) < 0.5) - MEAN) / STD).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_GPU):
                loss = F.cross_entropy(victim(xb), y_tr[torch.as_tensor(idx, device=DEV)], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if ep == EPOCHS - 1 or ep % 5 == 0: print(f"epoch {ep+1}/{EPOCHS} loss {loss.item():.3f} test acc {test_acc():.4f} ({time.time()-t0:.0f}s)")
    VICTIM_ACC = test_acc(); victim.eval()

    # ---- attacker's own encoder (ImageNet ResNet-18)
    enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); enc.fc = nn.Identity(); enc = enc.to(DEV).eval()
    PCA_MU = PCA_W = None
    @torch.no_grad()
    def extract(X, q, want_def=True, want_att=False, bs=1024):
        n = len(q["base"]); D, Lb, At = [], [], []
        for s in range(0, n, bs):
            sl = slice(s, s + bs); x01 = make_batch(X, q["base"][sl], q["ox"][sl], q["oy"][sl], q["flip"][sl])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_GPU):
                if want_def:
                    f = victim_feats(victim, ((x01 - MEAN) / STD).contiguous(memory_format=torch.channels_last))
                    Lb.append(victim.fc(f).argmax(1).cpu().numpy())
                if want_att:
                    xi = (F.interpolate(x01, size=64, mode="bilinear", align_corners=False) - IMEAN) / ISTD
                    At.append(enc(xi).float().cpu().numpy())
            if want_def:                                   # projection in full fp32, outside autocast
                f = f.float(); D.append(((f - PCA_MU) @ PCA_W).cpu().numpy() if PCA_W is not None else f.cpu().numpy())
        cat = lambda L: np.concatenate(L) if L else None
        return cat(D), cat(Lb), cat(At)

    # ---- defender PCA (victim penultimate -> 32 dims, total variance 4)
    Ffit, _, _ = extract(X_tr, draw(N_FIT, rng, BASES))
    mu = Ffit.mean(0); ew, U = np.linalg.eigh(np.cov(Ffit - mu, rowvar=False)); Wp = U[:, ::-1][:, :D_DEF]
    Zf = (Ffit - mu) @ Wp; Wp = Wp / math.sqrt(Zf.var(0).sum() / 4.0)
    PCA_MU = torch.as_tensor(mu, device=DEV, dtype=torch.float32); PCA_W = torch.as_tensor(Wp.copy(), device=DEV, dtype=torch.float32)
    print(f"PCA-32 explains {ew[::-1][:D_DEF].sum()/ew.sum():.3f} of victim-feature variance")

    Z_ref, _, _ = extract(X_tr, draw(N_REF, rng, BASES))
    Z_null, _, _ = extract(X_tr, draw(N_NULLB, rng, BASES))
    Z_pool, vlab_pool, A_pool_raw = extract(X_tr, draw(POOL, rng, BASES), want_att=True)
    qt = dict(base=np.arange(10000), ox=np.full(10000, 4), oy=np.full(10000, 4), flip=np.zeros(10000, bool))
    _, vlab_test, A_test_raw = extract(X_te, qt, want_def=True, want_att=True)
    ytest_true = y_te
    # attacker-side PCA-128 + standardisation (fit on the attacker's own pool)
    am = A_pool_raw.mean(0); aew, aU = np.linalg.eigh(np.cov(A_pool_raw - am, rowvar=False)); aW = aU[:, ::-1][:, :128]
    A_pool = (A_pool_raw - am) @ aW; asd = A_pool.std(0); A_pool = A_pool / asd; A_test = ((A_test_raw - am) @ aW) / asd
    np.savez_compressed(CACHE, Z_ref=Z_ref, Z_null=Z_null, Z_pool=Z_pool, vlab_pool=vlab_pool, A_pool=A_pool, A_test=A_test,
                        vlab_test=vlab_test, ytest_true=ytest_true, VICTIM_ACC=VICTIM_ACC)
    print(f"victim test acc {VICTIM_ACC:.4f} | embeddings done in {time.time()-t0:.0f}s, cached to {CACHE}")
print(f"victim-vs-truth agreement on test: {(vlab_test == ytest_true).mean():.4f} | pool label histogram: {np.bincount(vlab_pool, minlength=10)}")


# %% [markdown]
# ## Part 1 — defender machinery

# %%
# ===== Cell 3: Fourier, reference whitening, evaluation =====
def _fourier_numpy(pts, labels, nacc, modes):
    import scipy.sparse as sp
    F_ = np.zeros((nacc, len(modes)), dtype=np.complex128)
    for s in range(0, len(pts), CHUNK):
        P, lab = pts[s:s + CHUNK], labels[s:s + CHUNK]
        E = np.exp(-2j * np.pi * (P @ modes.T))
        F_ += sp.csr_matrix((np.ones(len(P)), (lab, np.arange(len(P)))), shape=(nacc, len(P))) @ E
    return F_
def _fourier_torch(pts, labels, nacc, modes):
    Mt = torch.as_tensor(modes, dtype=torch.float64, device=DEV)
    Fr = torch.zeros((nacc, len(modes)), dtype=torch.float64, device=DEV); Fi = torch.zeros_like(Fr)
    for s in range(0, len(pts), CHUNK):
        P = torch.as_tensor(pts[s:s + CHUNK], dtype=torch.float64, device=DEV)
        idx = torch.as_tensor(labels[s:s + CHUNK], dtype=torch.long, device=DEV)
        th = (2.0 * math.pi) * (P @ Mt.T)
        Fr.index_add_(0, idx, torch.cos(th)); Fi.index_add_(0, idx, -torch.sin(th))
    return Fr.cpu().numpy() + 1j * Fi.cpu().numpy()
def fourier_accounts(pts, labels, nacc, modes):
    return _fourier_torch(pts, labels, nacc, modes) if USE_GPU else _fourier_numpy(pts, labels, nacc, modes)

def make_bank(Zref, h, seed):
    # random Fourier modes in the defender's PCA space; mu and Sigma estimated from the historical reference
    r = np.random.default_rng(seed); modes = r.standard_normal((M_MODES, Zref.shape[1])) / h / (2 * np.pi); m = M_MODES
    if USE_GPU:
        Mt = torch.as_tensor(modes, dtype=torch.float64, device=DEV)
        Gt = torch.zeros((m, m), dtype=torch.complex128, device=DEV); mut = torch.zeros(m, dtype=torch.complex128, device=DEV)
        for s0 in range(0, len(Zref), 16384):
            th = (2 * math.pi) * (torch.as_tensor(Zref[s0:s0 + 16384], dtype=torch.float64, device=DEV) @ Mt.T)
            E = torch.complex(torch.cos(th), -torch.sin(th)); Gt += E.T @ E.conj(); mut += E.sum(0)
        G, mu = Gt.cpu().numpy(), mut.cpu().numpy()
    else:
        G = np.zeros((m, m), dtype=np.complex128); mu = np.zeros(m, dtype=np.complex128)
        for s0 in range(0, len(Zref), 4096):
            E = np.exp(-2j * np.pi * (Zref[s0:s0 + 4096] @ modes.T)); G += E.T @ E.conj(); mu += E.sum(0)
    G /= len(Zref); mu /= len(Zref)
    Sig = G - np.outer(mu, mu.conj()); Sig = (Sig + Sig.conj().T) / 2
    ew, U = np.linalg.eigh(Sig); keep = ew > FLOOR * ew.max()
    return dict(h=h, modes=modes, mu=mu, Wk=U[:, keep].conj() / np.sqrt(ew[keep]), meff=int(keep.sum()))

def _auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg])); n1 = len(pos); return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg))

def evaluate(pts, lab, cnt, banks, Kc, thr=None):
    A_ = len(cnt); iu = np.triu_indices(A_, 1); pc = (iu[0] < Kc) & (iu[1] < Kc)
    Fall = fourier_accounts(pts, lab, A_, np.vstack([b["modes"] for b in banks])); sq = np.sqrt(cnt); Ncol = cnt[:Kc].sum(); out = []
    for j, b in enumerate(banks):
        Fr = Fall[:, j * M_MODES:(j + 1) * M_MODES]
        Fw = ((Fr - cnt[:, None] * b["mu"][None, :]) / sq[:, None]) @ b["Wk"]
        S_un = float((np.abs((Fw[:Kc] * sq[:Kc, None]).sum(0)) ** 2).mean() / Ncol)
        rms2 = (np.abs(Fw) ** 2).mean(1)                       # per-account whitened discrepancy (~1 under null)
        ev_raw, V_raw = np.linalg.eigh(Fw @ Fw.conj().T / b["meff"])        # positive side: NO row normalisation
        Fn = Fw / np.sqrt(rms2)[:, None]
        ev, V = np.linalg.eigh(Fn @ Fn.conj().T / b["meff"])                 # missing mode: row-normalised
        r = dict(lam_min=ev[0], lam_max_raw=ev_raw[-1], acct_max=float(rms2.max()), S_union=S_un, z_oracle=(1 - S_un) * math.sqrt(b["meff"]))
        if thr is not None:
            t = thr[j]; nb_ = int((ev < t["tmin"]).sum()); na_ = int((ev_raw > t["tmax_raw"]).sum())
            r.update(mm=float(nb_ == 1 and na_ == 0), max=float(na_ >= 1), acct=float(rms2.max() > t["tacct"]),
                     oracle_low=float(S_un < t["olow"]), oracle_high=float(S_un > t["ohigh"]))
            v = V_raw[:, -1] if (na_ >= 1 and nb_ == 0) else V[:, 0]
            r["auc_spectral"] = _auc(np.abs(v[:Kc]) ** 2, np.abs(v[Kc:]) ** 2)
            r["auc_acct"] = _auc(rms2[:Kc], rms2[Kc:])            # per-account ranking baseline
            mu_a = Fr / cnt[:, None]; G = mu_a @ mu_a.conj().T; nrm = np.real(np.diag(G))
            D2 = (nrm[:, None] + nrm[None, :] - 2 * np.real(G))[iu]; r["auc_pairwise"] = _auc(-D2[pc], -D2[~pc])
        out.append(r)
    return out

def world(campZ, A_, rng):
    Kc = A_ // 5; camp = campZ[rng.permutation(len(campZ))]; lab_c = rng.integers(0, Kc, len(camp)); n_per = len(camp) // Kc
    ben = Z_ref[rng.integers(0, len(Z_ref), (A_ - Kc) * n_per)]; lab_b = Kc + np.repeat(np.arange(A_ - Kc), n_per)
    pts = np.vstack([camp, ben]); lab = np.concatenate([lab_c, lab_b]); return pts, lab, np.bincount(lab, minlength=A_), Kc

t0 = time.time(); BANKS = [make_bank(Z_ref, h, 777 + int(10 * h)) for h in HS]
print("banks: " + ", ".join(f"h={b['h']}: m_eff={b['meff']}" for b in BANKS) + f"  ({time.time()-t0:.0f}s)")
RESULTS = {"config": dict(QUICK=QUICK, SEED=SEED, EPOCHS=EPOCHS, N_REF=N_REF, POOL=POOL, CHECKS=CHECKS, TRIALS=TRIALS, AS=AS,
                          N_NULL=N_NULL, W=W, HS=HS, backend="gpu" if USE_GPU else "cpu"),
           "victim_acc": float(VICTIM_ACC), "banks": [dict(h=b["h"], meff=b["meff"]) for b in BANKS]}

# %% [markdown]
# ## Part 2 — extraction campaigns

# %%
# ===== Cell 4: attacker strategies =====
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings("ignore")
def fit_sub(idx):
    clf = LogisticRegression(max_iter=300, C=1.0); clf.fit(A_pool[idx], vlab_pool[idx]); return clf
def fidelity(clf): return float((clf.predict(A_test) == vlab_test).mean())
def proba(clf, X):
    P = np.zeros((len(X), 10)); P[:, clf.classes_] = clf.predict_proba(X); return P
def kcenter_pick(X, sel, k):
    dmin = np.full(len(X), np.inf); S = X[sel]; xn = (X ** 2).sum(1)
    for s0 in range(0, len(X), 4096):
        Xi = X[s0:s0 + 4096]; d2 = xn[s0:s0 + 4096, None] + (S ** 2).sum(1)[None, :] - 2 * Xi @ S.T; dmin[s0:s0 + 4096] = d2.min(1)
    dmin[sel] = -1; pick = []
    for _ in range(k):
        i = int(np.argmax(dmin)); pick.append(i); dmin = np.minimum(dmin, ((X - X[i]) ** 2).sum(1)); dmin[i] = -1
    return np.array(pick)
def _herd_numpy(Phi, N):
    base = Phi @ Phi.mean(0); Ps = np.zeros(len(Phi)); taken = np.zeros(len(Phi), bool); sel = np.empty(N, int)
    for t in range(N):
        sc = base - Ps / (t + 1); sc[taken] = -np.inf; i = int(np.argmax(sc)); sel[t] = i; taken[i] = True; Ps += Phi @ Phi[i]
    return sel
def _herd_torch(Phi, N):
    P = torch.as_tensor(Phi, dtype=torch.float32, device=DEV); base = P @ P.mean(0)
    Ps = torch.zeros(P.shape[0], dtype=torch.float32, device=DEV); taken = torch.zeros(P.shape[0], dtype=torch.bool, device=DEV)
    sel = torch.empty(N, dtype=torch.long, device=DEV)
    for t in range(N):
        sc = (base - Ps / (t + 1)).masked_fill(taken, float("-inf")); i = torch.argmax(sc); sel[t] = i; taken[i] = True; Ps += P @ P[i]
    return sel.cpu().numpy()
def herd_order(X, N, rng, D=256):
    sub = X[rng.choice(len(X), 2000, replace=False)]; d = np.sqrt(((sub[:1000] - sub[1000:]) ** 2).sum(1)); h = np.median(d)
    W_ = rng.standard_normal((X.shape[1], D)) / h; b = rng.uniform(0, 2 * np.pi, D); Phi = np.sqrt(2.0 / D) * np.cos(X @ W_ + b)
    return _herd_torch(Phi, N) if USE_GPU else _herd_numpy(Phi, N)

def run_strategy(name, trial):
    rng = np.random.default_rng(SEED + 1000 * trial + STRATS.index(name)); P = len(A_pool); Nmax = max(TARGETS); fid = {}
    if name in ("random", "herding"):
        order = rng.permutation(P)[:Nmax] if name == "random" else herd_order(A_pool, Nmax, rng)
        for N in CHECKS: fid[N] = fidelity(fit_sub(order[:N]))
        return order, fid
    sel = list(rng.choice(P, SEED_N, replace=False)); taken = np.zeros(P, bool); taken[sel] = True; clf = fit_sub(np.array(sel))
    for tgt in TARGETS:
        k = tgt - len(sel); cand = np.where(~taken)[0]
        if name == "entropy":
            pr = proba(clf, A_pool[cand]); ent = -(pr * np.log(pr + 1e-12)).sum(1); pick = cand[np.argsort(-ent)[:k]]
        elif name == "kcenter":
            pick = kcenter_pick(proba(clf, A_pool), np.array(sel), k)
        else:  # class_balanced
            pred = clf.predict(A_pool[cand]); pick = []
            for c in range(10):
                pool_c = cand[pred == c]; pick += list(rng.choice(pool_c, min(len(pool_c), k // 10), replace=False))
            rest = np.setdiff1d(cand, pick); pick = np.array(pick + list(rng.choice(rest, k - len(pick), replace=False)))
        sel += list(pick); taken[pick] = True; clf = fit_sub(np.array(sel))
        if tgt in CHECKS: fid[tgt] = fidelity(clf)
    return np.array(sel), fid

t0 = time.time(); CAMPAIGNS = {}
for name in STRATS:
    for tr_ in range(TRIALS):
        order, fid = run_strategy(name, tr_); CAMPAIGNS[(name, tr_)] = (order, fid)
        print(f"{name:>15} trial {tr_}: fidelity " + " ".join(f"N={N}:{fid[N]:.4f}" for N in CHECKS) + f"  ({time.time()-t0:.0f}s)")
FID = {(n, N): (float(np.mean([CAMPAIGNS[(n, t)][1][N] for t in range(TRIALS)])), float(np.std([CAMPAIGNS[(n, t)][1][N] for t in range(TRIALS)])))
       for n in STRATS for N in CHECKS}
print("\nmean fidelity (agreement with victim on test):")
for n in STRATS: print(f"{n:>15}: " + "  ".join(f"N={N}: {FID[(n, N)][0]:.4f}±{FID[(n, N)][1]:.4f} (Δ {100*(FID[(n, N)][0]-FID[('random', N)][0]):+.2f} pt)" for N in CHECKS))
RESULTS["fidelity"] = {f"{n}|{N}": FID[(n, N)] for n in STRATS for N in CHECKS}

# %%
# ===== Cell 5: Step-6 evaluation (equal-size colluders, global stream, Higher Criticism, Bonferroni over bandwidths) =====
ALPHA_B = ALPHA / len(HS)
def world(campZ, A_, K_, rng):
    n_per = len(campZ) // K_; camp = campZ[rng.permutation(len(campZ))][:K_ * n_per]; lab_c = np.arange(K_ * n_per) % K_
    ben = Z_ref[rng.integers(0, len(Z_ref), (A_ - K_) * n_per)]; lab_b = K_ + np.repeat(np.arange(A_ - K_), n_per)
    pts = np.vstack([camp, ben]); lab = np.concatenate([lab_c, lab_b]); return pts, lab, np.bincount(lab, minlength=A_)

def stats(pts, lab, cnt, Kc, vectors=False):
    A_ = len(cnt); Fall = fourier_accounts(pts, lab, A_, np.vstack([b["modes"] for b in BANKS])); sq = np.sqrt(cnt)
    out = []
    for j, b in enumerate(BANKS):
        Fr = Fall[:, j * M_MODES:(j + 1) * M_MODES]; Fw = ((Fr - cnt[:, None] * b["mu"][None, :]) / sq[:, None]) @ b["Wk"]
        rms2 = (np.abs(Fw) ** 2).mean(1); Fn = Fw / np.sqrt(rms2)[:, None]
        Craw, Cn = Fw @ Fw.conj().T / b["meff"], Fn @ Fn.conj().T / b["meff"]
        if vectors: evr, Vr = np.linalg.eigh(Craw); evn = np.linalg.eigvalsh(Cn)
        else: evr, Vr, evn = np.linalg.eigvalsh(Craw), None, np.linalg.eigvalsh(Cn)
        S_un = float((np.abs((Fw[:Kc] * sq[:Kc, None]).sum(0)) ** 2).mean() / cnt[:Kc].sum())
        S_all = float((np.abs((Fw * sq[:, None]).sum(0)) ** 2).mean() / cnt.sum())
        out.append(dict(rms2=rms2, evr=evr, evn=evn, Vr=Vr, S_union=S_un, S_all=S_all, mm_ok=A_ <= 0.8 * b["meff"]))
    return out

def hc_stat(rms2, null_sorted):
    M_ = len(null_sorted); p = (1 + M_ - np.searchsorted(null_sorted, rms2, side="left")) / (M_ + 1)
    p = np.sort(p); A_ = len(p); k = A_ // 2; i = np.arange(1, k + 1); pp = p[:k]
    return float(np.max(np.sqrt(A_) * (i / A_ - pp) / np.sqrt(pp * (1 - pp))))

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den; hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - hw), min(1.0, c + hw))
DETECTORS = ["spec", "glob_hi", "hc", "acct", "mm", "glob_lo"]
print("Step-6 evaluation ready; Bonferroni alpha per bandwidth =", round(ALPHA_B, 5))

# %%
# ===== Cell 5: realistic benign windows + fast statistics (GPU path, numpy fallback) =====
B0 = BANKS[0]; MEFF = B0["meff"]
if USE_GPU:
    MODES_T = torch.as_tensor(B0["modes"], dtype=torch.float64, device=DEV)
    MU_T = torch.as_tensor(B0["mu"], dtype=torch.complex128, device=DEV); WK_T = torch.as_tensor(B0["Wk"], dtype=torch.complex128, device=DEV)

# topic regions = nearest neighbours (in the defender embedding) of random reference points
t0 = time.time(); rng0 = np.random.default_rng(SEED + 77)
centers = Z_ref[rng0.choice(len(Z_ref), R_TOPICS, replace=False)]
if USE_GPU:
    Zt = torch.as_tensor(Z_ref, dtype=torch.float32, device=DEV); Ct = torch.as_tensor(centers, dtype=torch.float32, device=DEV)
    TOPIC_NN = torch.cdist(Ct, Zt).topk(NN_TOPIC, largest=False).indices.cpu().numpy(); del Zt
else:
    D_ = ((centers[:, None, :] - Z_ref[None, :, :]) ** 2).sum(-1); TOPIC_NN = np.argpartition(D_, NN_TOPIC, axis=1)[:, :NN_TOPIC]; del D_
print(f"topic regions: {TOPIC_NN.shape} ({time.time()-t0:.0f}s)")

def gen_window(A_, rng, structured=True, vol_scale=1):
    vols = (np.clip(np.round(N0_MED * np.exp(VOL_SIGMA * rng.standard_normal(A_))), 5, 20 * N0_MED) * vol_scale).astype(int)
    lab = np.repeat(np.arange(A_), vols); T_ = len(lab); idx = rng.integers(0, len(Z_ref), T_)
    if structured:
        nclu = int(CLUSTER_FRAC * A_) // CLUSTER_SIZE; topic = np.full(A_, -1)
        if nclu > 0:
            mem = rng.permutation(A_)[:nclu * CLUSTER_SIZE].reshape(nclu, CLUSTER_SIZE)
            topic[mem] = rng.integers(0, R_TOPICS, nclu)[:, None]
        tq = topic[lab]; m1 = (tq >= 0) & (rng.random(T_) < TOPIC_Q)
        idx[m1] = TOPIC_NN[tq[m1], rng.integers(0, NN_TOPIC, m1.sum())]
        trend = int(rng.integers(R_TOPICS)); fol = rng.random(A_) < TREND_FRAC
        m2 = fol[lab] & (rng.random(T_) < TREND_Q); idx[m2] = TOPIC_NN[trend, rng.integers(0, NN_TOPIC, m2.sum())]
    return idx, lab, vols

def _fourier_t(pts, labels, nacc):
    Fr = torch.zeros((nacc, M_MODES), dtype=torch.float64, device=DEV); Fi = torch.zeros_like(Fr)
    for s in range(0, len(pts), CHUNK):
        P = torch.as_tensor(pts[s:s + CHUNK], dtype=torch.float64, device=DEV)
        ix = torch.as_tensor(labels[s:s + CHUNK], dtype=torch.long, device=DEV); th = (2.0 * math.pi) * (P @ MODES_T.T)
        Fr.index_add_(0, ix, torch.cos(th)); Fi.index_add_(0, ix, -torch.sin(th))
    return torch.complex(Fr, Fi)

def wstats(pts, lab, cnt, Kc=0, want_vec=False, force_numpy=False):
    # returns lam_max (upper-edge spectral), rms2 (per-account discrepancy), S_all (global), S_un (colluder union), v (account eigvec)
    A_ = len(cnt)
    if USE_GPU and not force_numpy:
        Fr = _fourier_t(pts, lab, A_); c = torch.as_tensor(cnt, dtype=torch.float64, device=DEV); sq = torch.sqrt(c)
        Fw = ((Fr - c[:, None] * MU_T[None, :]) / sq[:, None]) @ WK_T
        rms2 = (Fw.abs() ** 2).mean(1); G = Fw.conj().T @ Fw / MEFF; ev, U = torch.linalg.eigh(G)
        S_all = float(((Fw * sq[:, None]).sum(0).abs() ** 2).mean() / c.sum())
        S_un = float(((Fw[:Kc] * sq[:Kc, None]).sum(0).abs() ** 2).mean() / c[:Kc].sum()) if Kc > 0 else float("nan")
        v = None
        if want_vec: v = (Fw @ U[:, -1]).abs().cpu().numpy() ** 2
        return float(ev[-1]), rms2.cpu().numpy(), S_all, S_un, v
    Fr = _fourier_numpy(pts, lab, A_, B0["modes"]); sq = np.sqrt(cnt)
    Fw = ((Fr - cnt[:, None] * B0["mu"][None, :]) / sq[:, None]) @ B0["Wk"]
    rms2 = (np.abs(Fw) ** 2).mean(1); ev, U = np.linalg.eigh(Fw.conj().T @ Fw / MEFF)
    S_all = float((np.abs((Fw * sq[:, None]).sum(0)) ** 2).mean() / cnt.sum())
    S_un = float((np.abs((Fw[:Kc] * sq[:Kc, None]).sum(0)) ** 2).mean() / cnt[:Kc].sum()) if Kc > 0 else float("nan")
    v = np.abs(Fw @ U[:, -1]) ** 2 if want_vec else None
    return float(ev[-1]), rms2, S_all, S_un, v

def thresholds(recs):
    lm = np.array([r[0] for r in recs]); ga = np.array([r[2] for r in recs]); ac = np.array([r[1].max() for r in recs])
    ns = np.sort(np.concatenate([r[1] for r in recs])); hcs = np.array([hc_stat(r[1], ns) for r in recs])
    q_ = lambda a: float(np.quantile(a, 1 - ALPHA_D))
    return dict(tmax=q_(lm), tglo=q_(ga), tacct=q_(ac), thc=q_(hcs), null=ns)

def decide(st, T_):
    lm, r2, ga, _, _ = st
    d = dict(spec=lm > T_["tmax"], hc=hc_stat(r2, T_["null"]) > T_["thc"], acct=r2.max() > T_["tacct"], glob=ga > T_["tglo"])
    d["suite"] = d["spec"] or d["hc"] or d["glob"]; return d

def realistic_attack_world(campZ, A_, K_, rng, hetero=False, vol_scale=1):
    idx, lab, vols = gen_window(A_, rng, True, vol_scale)
    keep = lab >= K_; idx, lab = idx[keep], lab[keep]
    if hetero:
        w_ = np.exp(VOL_SIGMA * rng.standard_normal(K_)); cnts = np.floor(len(campZ) * w_ / w_.sum()).astype(int)
        cnts[: len(campZ) - cnts.sum()] += 1; lab_c = np.repeat(np.arange(K_), cnts)
    else:
        n_per = len(campZ) // K_; campZ = campZ[: K_ * n_per]; lab_c = np.arange(K_ * n_per) % K_
    camp = campZ[rng.permutation(len(campZ))]
    pts = np.vstack([camp, Z_ref[idx]]); labs = np.concatenate([lab_c, lab]); return pts, labs, np.bincount(labs, minlength=A_)

# self-check: GPU and numpy statistics agree
rng = np.random.default_rng(3); idx, lab, vols = gen_window(min(A_STRESS), rng)
a = wstats(Z_ref[idx], lab, np.bincount(lab), 5); b = wstats(Z_ref[idx], lab, np.bincount(lab), 5, force_numpy=True)
print(f"stats self-check: lam_max {a[0]:.6f}/{b[0]:.6f}  S_all {a[2]:.6f}/{b[2]:.6f}  max|rms2 diff| {np.abs(a[1]-b[1]).max():.2e}")
RESULTS7 = {"config": dict(QUICK=QUICK, MEFF=MEFF, A_STRESS=A_STRESS, A_ADAPT=A_ADAPT, A_SCALE=A_SCALE, N_CAL=N_CAL, ALPHA_D=ALPHA_D)}

# %% [markdown]
# ## Part A — realistic benign traffic and calibration

# %%
# ===== Part A: realistic benign traffic vs calibration scheme =====
t0 = time.time(); CAL = {}; rowsA = []
for A_ in A_STRESS:
    rng = np.random.default_rng(SEED + 21_000 + A_)
    hist = []                                  # historical accounts (each = array of Z_ref indices)
    for h_ in range(HIST_WIN):
        idx, lab, vols = gen_window(A_, rng); hist += np.split(idx, np.cumsum(vols)[:-1])
    def world_iid(r): idx, lab, vols = gen_window(A_, r, structured=False); return Z_ref[idx], lab, vols
    def world_acct(r):
        pick = r.integers(0, len(hist), A_); parts = [hist[i] for i in pick]; vols = np.array([len(p) for p in parts])
        return Z_ref[np.concatenate(parts)], np.repeat(np.arange(A_), vols), vols
    def world_win(r): idx, lab, vols = gen_window(A_, r); return Z_ref[idx], lab, vols
    for scheme, fn in [("iid", world_iid), ("acct_resample", world_acct), ("window_resample", world_win)]:
        recs = []
        for t in range(N_CAL):
            r = np.random.default_rng(SEED + 22_000_000 + 1000 * A_ + t + 7 * len(scheme)); pts, lab, vols = fn(r)
            recs.append(wstats(pts, lab, np.bincount(lab, minlength=A_)))
        CAL[(scheme, A_)] = thresholds(recs)
    test = []
    for t in range(N_TEST):
        r = np.random.default_rng(SEED + 23_000_000 + 1000 * A_ + t); idx, lab, vols = gen_window(A_, r)
        test.append(wstats(Z_ref[idx], lab, np.bincount(lab, minlength=A_)))
    for scheme in ["iid", "acct_resample", "window_resample"]:
        fp = {k: np.mean([decide(s, CAL[(scheme, A_)])[k] for s in test]) for k in ["spec", "hc", "acct", "glob", "suite"]}
        rowsA.append(dict(A=A_, scheme=scheme, **{k: float(v) for k, v in fp.items()}))
        # [PAPER] App. P (Step 7, "Only whole-window resampling calibrates"): Pass (suite 0.0–0.6%)
        # [PAPER] App. E 'The dose–response study': 0.6% (window-resampled suite); account resampling: spectral 31%, global 100% (plausibly from here, see header)
        print(f"A={A_:4d} calibrated by {scheme:>15}: FPR on realistic windows  " + " ".join(f"{k}={v:.3f}" for k, v in fp.items())
              + f"   (nominal {ALPHA_D:.4f} per detector, suite <= {ALPHA})  ({time.time()-t0:.0f}s)")
RESULTS7["A"] = rowsA

# %% [markdown]
# ## Part B — adaptive attacker

# %%
# ===== Part B: adaptive attacker (padding, heterogeneous volumes, time-splitting, defender aggregation) =====
t0 = time.time(); rowsB = []; FIDP = {}
for name in ["kcenter", "entropy"]:
    for N in N_ADAPT:
        for f in FS:
            fids = []
            for tr_ in range(TRIALS):
                order = CAMPAIGNS[(name, tr_)][0]; nk = int(round((1 - f) * N)); sel = order[:nk]
                rest = np.setdiff1d(np.arange(len(A_pool)), sel); pad = np.random.default_rng(SEED + 31 * tr_ + int(100 * f)).choice(rest, N - nk, replace=False)
                fids.append(fidelity(fit_sub(np.concatenate([sel, pad]).astype(int))))
            FIDP[(name, N, f)] = float(np.mean(fids))
        print(f"{name} N={N}: fidelity by padding f: " + " ".join(f"{f}:{FIDP[(name, N, f)]:.4f} (Δ {100*(FIDP[(name, N, f)]-FID[('random', N)][0]):+.2f}pt)" for f in FS))
for A_ in A_ADAPT:
    T_ = CAL[("window_resample", A_)]
    recs = []                       # separate null for the defender's aggregated (T-times longer) window
    for t in range(N_CAL):
        r = np.random.default_rng(SEED + 24_000_000 + 1000 * A_ + t); idx, lab, vols = gen_window(A_, r, True, vol_scale=max(TS))
        recs.append(wstats(Z_ref[idx], lab, np.bincount(lab, minlength=A_)))
    T_AGG = thresholds(recs); print(f"A={A_}: aggregated-window null calibrated ({time.time()-t0:.0f}s)")
    for name in ["kcenter", "entropy"]:
        for N in N_ADAPT:
            for f in FS:
                variants = [("equal", 1, False, 1), ("hetero", 1, True, 1)] + [(f"split{T}", T, False, 1) for T in TS if T > 1] + [(f"agg{max(TS)}", 1, False, max(TS))]
                for vname, T, het, vs in variants:
                    hits = {k: 0 for k in ["spec", "hc", "acct", "glob", "suite"]}; n = 0
                    for tr_ in range(TRIALS):
                        order = CAMPAIGNS[(name, tr_)][0]; nk = int(round((1 - f) * N)); sel = order[:nk]
                        rest = np.setdiff1d(np.arange(len(A_pool)), sel); pad = np.random.default_rng(SEED + 31 * tr_ + int(100 * f)).choice(rest, N - nk, replace=False)
                        campZ_full = Z_pool[np.concatenate([sel, pad]).astype(int)]
                        for w in range(W7):
                            any_ = {k: False for k in hits}
                            for t in range(T):
                                r = np.random.default_rng(SEED + 41_000_000 + 100_000 * tr_ + 1000 * w + 10 * t + A_ + N + int(1000 * f))
                                pts, lab, cnt = realistic_attack_world(campZ_full[t::T], A_, K_ADAPT, r, hetero=het, vol_scale=vs)
                                d = decide(wstats(pts, lab, cnt, K_ADAPT), T_AGG if vs > 1 else T_)
                                for k in hits: any_[k] |= bool(d[k])
                            for k in hits: hits[k] += int(any_[k])
                            n += 1
                    row = dict(A=A_, strategy=name, N=N, f=f, variant=vname, n=n, dfid=FIDP[(name, N, f)] - FID[("random", N)][0], **{k: v / n for k, v in hits.items()})
                    rowsB.append(row)
                # [PAPER] App. P (Step 7): padding f ≥ 0.75 "Fail (no detection even unpadded)"; hetero volumes and split/aggregate "Untestable (power 0)"
                print(f"A={A_:3d} {name:>8} N={N:5d} f={f:.2f} Δfid={100*rowsB[-1]['dfid']:+.2f}pt | suite: " +
                      " ".join(f"{r['variant']}={r['suite']:.2f}" for r in rowsB[-len(variants):]) + f"  ({time.time()-t0:.0f}s)")
RESULTS7["B"] = rowsB

# %% [markdown]
# ## Part C — production-scale A

# %%
# ===== Part C: production-scale account counts =====
t0 = time.time(); rowsC = []
for A_ in A_SCALE:
    recs = []
    for t in range(N_CAL_SCALE):
        r = np.random.default_rng(SEED + 51_000_000 + 10 * A_ + t); idx, lab, vols = gen_window(A_, r)
        recs.append(wstats(Z_ref[idx], lab, np.bincount(lab, minlength=A_)))
    T_ = thresholds(recs); print(f"A={A_}: calibrated on {N_CAL_SCALE} realistic windows ({time.time()-t0:.0f}s)")
    for name in ["kcenter", "entropy"]:
        for N in N_SCALE:
            for K_ in K_SCALE:
                hits = {k: 0 for k in ["spec", "hc", "acct", "glob", "suite"]}; n = 0; Ss = []
                for tr_ in range(TRIALS):
                    campZ = Z_pool[CAMPAIGNS[(name, tr_)][0][:N]]
                    for w in range(W7):
                        r = np.random.default_rng(SEED + 61_000_000 + 1000 * tr_ + w + A_ + N + K_)
                        pts, lab, cnt = realistic_attack_world(campZ, A_, K_, r); st = wstats(pts, lab, cnt, K_); d = decide(st, T_)
                        for k in hits: hits[k] += int(d[k])
                        n += 1; Ss.append(st[3])
                S_ = float(np.mean(Ss)); xu = (S_ - 1) * math.sqrt(MEFF) / math.sqrt(A_)
                row = dict(A=A_, strategy=name, N=N, K=K_, n_per=N // K_, S=S_, x_up=xu, pred_spec=xu > 1.36, **{k: v / n for k, v in hits.items()})
                rowsC.append(row)
                print(f"A={A_:5d} {name:>8} N={N:5d} K={K_:3d} (n/acct {N//K_:4d}) S={S_:6.2f} x_up={xu:5.2f} pred={'Y' if xu > 1.36 else 'n'} | "
                      + " ".join(f"{k}={row[k]:.2f}" for k in ["spec", "hc", "acct", "glob", "suite"]) + f"  ({time.time()-t0:.0f}s)")
agree = np.mean([r["pred_spec"] == (r["spec"] > 0.5) for r in rowsC])
# [PAPER] App. P (Step 7, "Upper-edge rule predicts power at A ≤ 10^4"): Fail (10/24 cells)
print(f"P7.5 rule (S-1)sqrt(m_eff)/sqrt(A) > 1.36 vs spectral power>0.5: agreement {agree:.3f} over {len(rowsC)} cells; "
      f"max global power at A={max(A_SCALE)}: {max(r['glob'] for r in rowsC if r['A']==max(A_SCALE)):.2f}")
RESULTS7["C"] = rowsC; RESULTS7["C_agree"] = float(agree)

# %% [markdown]
# ## Figures

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(19, 5))
rs = RESULTS7["A"]; labs_ = [f"A={r['A']}\n{r['scheme']}" for r in rs]; xs = np.arange(len(rs))
for i, k in enumerate(["spec", "hc", "acct", "glob", "suite"]):
    ax[0].bar(xs + (i - 2) * 0.16, [r[k] for r in rs], 0.16, label=k)
ax[0].axhline(ALPHA_D, color="k", ls=":", lw=1); ax[0].axhline(ALPHA, color="r", ls=":", lw=1); ax[0].set_yscale("symlog", linthresh=0.01)
ax[0].set_xticks(xs); ax[0].set_xticklabels(labs_, fontsize=7); ax[0].set_ylabel("false-positive rate"); ax[0].set_title("P7.1: calibration under realistic benign traffic"); ax[0].legend(fontsize=7)
for name, mk in [("kcenter", "o"), ("entropy", "s")]:
    for A_ in A_ADAPT:
        r_ = sorted([r for r in RESULTS7["B"] if r["strategy"] == name and r["A"] == A_ and r["N"] == max(N_ADAPT) and r["variant"] == "equal"], key=lambda r: r["f"])
        ax[1].plot([100 * r["dfid"] for r in r_], [r["suite"] for r in r_], marker=mk, label=f"{name} A={A_}")
        for r in r_: ax[1].annotate(f"f={r['f']}", (100 * r["dfid"], r["suite"]), fontsize=6)
ax[1].set_xlabel("fidelity gain over random (points)"); ax[1].set_ylabel("suite detection"); ax[1].set_title(f"P7.2: evasion cost of padding (N={max(N_ADAPT)})"); ax[1].legend(fontsize=7)
for name, mk in [("kcenter", "o"), ("entropy", "s")]:
    r_ = [r for r in RESULTS7["C"] if r["strategy"] == name]
    ax[2].scatter([max(r["x_up"], 1e-2) for r in r_], [r["spec"] for r in r_], marker=mk, label=f"{name} spectral")
    ax[2].scatter([max(r["x_up"], 1e-2) for r in r_], [r["hc"] for r in r_], marker=mk, facecolors="none", edgecolors="gray", label=f"{name} HC")
ax[2].axvline(1.36, color="k", ls=":", label="rule x=1.36"); ax[2].set_xscale("log"); ax[2].set_xlabel("(S̄−1)√m_eff/√A"); ax[2].set_ylabel("power")
ax[2].set_title("P7.5: production-scale A"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/step7_figure.png", dpi=130); plt.show()

# %% [markdown]
# ## VERDICT REPORT — copy into the chat

# %%
print("=" * 86); print("STEP 7 VERDICT REPORT"); print("=" * 86)
print("config:", json.dumps(RESULTS7["config"]))
for r in RESULTS7["A"]: print(f"[A] A={r['A']:4d} {r['scheme']:>15}: " + " ".join(f"{k}={r[k]:.3f}" for k in ["spec", "hc", "acct", "glob", "suite"]))
for r in RESULTS7["B"]: print(f"[B] A={r['A']:3d} {r['strategy']:>8} N={r['N']:5d} f={r['f']:.2f} {r['variant']:>7} Δfid={100*r['dfid']:+.2f}pt " + " ".join(f"{k}={r[k]:.2f}" for k in ["spec", "hc", "acct", "glob", "suite"]))
for r in RESULTS7["C"]: print(f"[C] A={r['A']:5d} {r['strategy']:>8} N={r['N']:5d} K={r['K']:3d} S={r['S']:.2f} x_up={r['x_up']:.2f} " + " ".join(f"{k}={r[k]:.2f}" for k in ["spec", "hc", "acct", "glob", "suite"]))
print(f"[C] rule agreement {RESULTS7['C_agree']:.3f}"); print("=" * 86)
with open(f"{OUTDIR}/step7_results.json", "w") as fh: json.dump(RESULTS7, fh, indent=1, default=float)
print(f"saved {OUTDIR}/step7_results.json and step7_figure.png")
