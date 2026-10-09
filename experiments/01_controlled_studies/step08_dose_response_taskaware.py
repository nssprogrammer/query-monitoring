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
# Table 4 columns s = 0 / 0.05 / 0.1 / 0.25 / 0.5 / 1; 10,000-query campaigns, 24 windows per cell (`TRIALS` × `W8` = 3 × 8),
# window-resampled calibration. Geometry suite = `geo_suite`, task-aware suite = `dir_suite` in Part B.
#
# | Paper location | Quantity | Value in paper | Where computed here |
# |---|---|---|---|
# | Table 4, row "100, 20 / entropy / geometry" | power vs s | 1.00 / 1.00 / 1.00 / 0.00 / 0.00 / 0.00 | Part B print "geo/dir by s" (first value), A=100 entropy N=10000; VERDICT `[B]` `geo_suite=` |
# | Table 4, row "100, 20 / entropy / task-aware" | power vs s | 1.00 / 1.00 / 1.00 / 1.00 / 0.12 / 0.17 | Part B print "geo/dir by s" (second value), A=100 entropy N=10000; VERDICT `[B]` `dir_suite=` |
# | Table 4, row "100, 20 / k-center / geometry" | power vs s | 1.00 / 1.00 / 1.00 / 0.00 / 0.00 / 0.00 | Part B, A=100 kcenter N=10000, `geo_suite` |
# | Table 4, row "100, 20 / k-center / task-aware" | power vs s | 1.00 / 1.00 / 1.00 / 0.79 / 0.04 / 0.08 | Part B, A=100 kcenter N=10000, `dir_suite` |
# | Table 4, row "1000, 100 / entropy / geometry" | power vs s | 1.00 / 0.67 / 0.04 / 0.00 / 0.04 / 0.04 | Part B, A=1000 entropy N=10000, `geo_suite` |
# | Table 4, row "1000, 100 / entropy / task-aware" | power vs s | 1.00 / 0.96 / 0.67 / 0.33 / 0.04 / 0.04 | Part B, A=1000 entropy N=10000, `dir_suite` |
# | Table 4, row "1000, 100 / k-center / geometry" | power vs s | 1.00 / 0.38 / 0.04 / 0.00 / 0.04 / 0.04 | Part B, A=1000 kcenter N=10000, `geo_suite` |
# | Table 4, row "1000, 100 / k-center / task-aware" | power vs s | 1.00 / 0.21 / 0.25 / 0.08 / 0.04 / 0.04 | Part B, A=1000 kcenter N=10000, `dir_suite` |
# | Fig. 3 (App. B, "The barrier in a controlled setting") | geometry detection power vs strength of benign coordination, 100 and 1,000 accounts | power erased at strength 0.25 (100 accounts); falls at 0.05 and vanishes at 0.1 (1,000 accounts) | Part B `geo_suite` at N=10000 (same cells as Table 4 geometry rows); the paper figure is drawn by a separate figure script, and how the two strategies are combined per curve is not isolated here |
# | App. E 'The dose–response study' | window-resampled suite false alarms at s = 1 | 0.6% | plausibly Part A print "null FPR" at s=1.00 (`geo_suite`); the companion 31% / 100% account-resampling figures come from Step 7 Part A |
# | App. E 'The dose–response study' | union S̄ of benign ten-account clusters vs attacks | 83–159, against 26–30 for the attacks | Part C prints "benign 10-account clusters ... union S" and "attacks at N=10000: union S" |
# | App. E 'The dose–response study' | task-aware suite tolerated one cluster at 100 accounts and a few at 1,000, then failed | qualitative (see Table 4) | Part B `dir_suite` |
# | Prediction log (Step 8), row "Geometry breaks at s ∈ [0.05, 0.5]" (P8.1) | interpolated 50% point s50 | Partial (0.175 at A = 100; ≤ 0.06 at 1,000) | prediction print "P8.1/P8.2 ... (s50 ...)" |
# | Prediction log (Step 8), row "Task-aware suite ≥ 0.8 at s = 1" (P8.2) | directional-suite power at s = 1 | Fail (≤ 0.17) | prediction print "P8.1/P8.2 ... dir s=1" |
# | Prediction log (Step 8), row "Calibration ≤ 1.5× nominal" (P8.3) | null / random-campaign false-positive rate | Partial (mean 1.1%, worst 3.3%) | worst: print "P8.3 max null FPR ... random-campaign max"; mean: not isolated (summary over Part A / Part B random rows) |
# | Prediction log (Step 8), row "Clusters out-shine attacks in geometry, not margin" (P8.4) | benign-cluster vs attack union S̄ and low-margin share | Partial (geometry yes, margin no) | Part C diagnostic prints |
#
# Not reported in the paper: N = 5,000 cells, random-campaign power curves, the combined `both` suite, individual
# detectors (`spec`, `hc`, `glob`, `d_hc`, `d_glob`, `d_sum`), the low-margin threshold and per-topic low-margin shares,
# and the figure `step8_figure.png`.

# %% [markdown]
# # Step 8 — How much benign structure can detection survive, and do task-aware statistics help?
# **Kaggle:** GPU + Internet → **Run All** (≈45–60 min; rebuilds a new cache with the victim's logits, ≈6 min). `QUICK=True` first.
#
# Step 7: under strongly clustered benign traffic every detector stayed calibrated but lost all power — benign topic clusters were *more concentrated* than the attacks in query geometry. Step 8 asks two questions.
#
# **1. Dose–response.** Benign structure strength s scales the share of clustered accounts (30%·s) and trend followers (20%·s); s = 0 is unstructured traffic. Geometry suite (spectral + HC + global on the victim embedding) is measured at s ∈ {0, 0.05, 0.1, 0.25, 0.5, 1}.
#
# **2. Task-aware, directional statistic.** Extraction seeks queries the model is unsure about; most topical benign use does not. Per query we flag *low victim margin* (top-1 minus top-2 logit below the reference 10% quantile). Per account: z-score of the low-margin share. Detectors: HC over accounts, pooled (global) share, and Σ max(z,0)² (many weak accounts); **directional suite** = their OR, each at α/3. This is boundary-proximity monitoring in the spirit of PRADA, aggregated across accounts and calibrated on whole historical windows.
#
# Everything is window-calibrated (whole historical windows, the only scheme that passed Step 7).
#
# | ID | Locked prediction |
# |---|---|
# | **P8.1** | Geometry suite: power for k-center/entropy at N ≥ 5,000 is high at s = 0 and falls to ≈ 0 by s = 1; the 50% point lies at s ∈ [0.05, 0.5] |
# | **P8.2** | Directional suite keeps power ≥ 0.8 at s = 1 for entropy at N ≥ 5,000 (both A); k-center exploratory |
# | **P8.3** | Every detector calibrated: random-campaign and null FPR ≤ 1.5× nominal at every s |
# | **P8.4** | Diagnostic at s = 1: in geometry, the 90th percentile of benign-cluster union S̄ exceeds the attack union S̄; in low-margin share, benign clusters (90th percentile) sit below the attacks |

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
CACHE = f"{OUTDIR}/step8_cache_{'quick' if QUICK else 'full'}.npz"
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
# ---- Step 8 parameters
STRENGTHS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
LOWQ = 0.10
if QUICK:
    A8, K8, N8, N_CAL8, N_TEST8, W8, STRENGTHS = [50], {50: 10}, [600, 1000], 60, 40, 2, [0.0, 1.0]
else:
    A8, K8, N8, N_CAL8, N_TEST8, W8 = [100, 1000], {100: 20, 1000: 100}, [5000, 10000], 500, 300, 8

# %% [markdown]
# ## Part 0 — victim, embeddings and logits (new cache)

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
        n = len(q["base"]); D, Lb, At, Lg = [], [], [], []
        for s in range(0, n, bs):
            sl = slice(s, s + bs); x01 = make_batch(X, q["base"][sl], q["ox"][sl], q["oy"][sl], q["flip"][sl])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_GPU):
                if want_def:
                    f = victim_feats(victim, ((x01 - MEAN) / STD).contiguous(memory_format=torch.channels_last))
                    lg = victim.fc(f).float(); Lb.append(lg.argmax(1).cpu().numpy()); Lg.append(lg.cpu().numpy())
                if want_att:
                    xi = (F.interpolate(x01, size=64, mode="bilinear", align_corners=False) - IMEAN) / ISTD
                    At.append(enc(xi).float().cpu().numpy())
            if want_def:                                   # projection in full fp32, outside autocast
                f = f.float(); D.append(((f - PCA_MU) @ PCA_W).cpu().numpy() if PCA_W is not None else f.cpu().numpy())
        cat = lambda L: np.concatenate(L) if L else None
        globals()["LAST_LOGITS"] = cat(Lg)
        return cat(D), cat(Lb), cat(At)

    # ---- defender PCA (victim penultimate -> 32 dims, total variance 4)
    Ffit, _, _ = extract(X_tr, draw(N_FIT, rng, BASES))
    mu = Ffit.mean(0); ew, U = np.linalg.eigh(np.cov(Ffit - mu, rowvar=False)); Wp = U[:, ::-1][:, :D_DEF]
    Zf = (Ffit - mu) @ Wp; Wp = Wp / math.sqrt(Zf.var(0).sum() / 4.0)
    PCA_MU = torch.as_tensor(mu, device=DEV, dtype=torch.float32); PCA_W = torch.as_tensor(Wp.copy(), device=DEV, dtype=torch.float32)
    print(f"PCA-32 explains {ew[::-1][:D_DEF].sum()/ew.sum():.3f} of victim-feature variance")

    Z_ref, _, _ = extract(X_tr, draw(N_REF, rng, BASES)); L_ref = LAST_LOGITS
    Z_null, _, _ = extract(X_tr, draw(N_NULLB, rng, BASES))
    Z_pool, vlab_pool, A_pool_raw = extract(X_tr, draw(POOL, rng, BASES), want_att=True); L_pool = LAST_LOGITS
    qt = dict(base=np.arange(10000), ox=np.full(10000, 4), oy=np.full(10000, 4), flip=np.zeros(10000, bool))
    _, vlab_test, A_test_raw = extract(X_te, qt, want_def=True, want_att=True)
    ytest_true = y_te
    # attacker-side PCA-128 + standardisation (fit on the attacker's own pool)
    am = A_pool_raw.mean(0); aew, aU = np.linalg.eigh(np.cov(A_pool_raw - am, rowvar=False)); aW = aU[:, ::-1][:, :128]
    A_pool = (A_pool_raw - am) @ aW; asd = A_pool.std(0); A_pool = A_pool / asd; A_test = ((A_test_raw - am) @ aW) / asd
    np.savez_compressed(CACHE, Z_ref=Z_ref, Z_null=Z_null, Z_pool=Z_pool, vlab_pool=vlab_pool, A_pool=A_pool, A_test=A_test,
                        vlab_test=vlab_test, ytest_true=ytest_true, VICTIM_ACC=VICTIM_ACC, L_ref=L_ref, L_pool=L_pool)
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
# ===== Cell 5: Step-8 machinery (banks per space, structured windows with strength s, geometry + directional statistics) =====
from scipy.stats import norm as _norm
def margin_of(L): s_ = np.sort(L, 1); return s_[:, -1] - s_[:, -2]
M_REF, M_POOL = margin_of(L_ref), margin_of(L_pool); M_THR = float(np.quantile(M_REF, LOWQ))
LOW_REF, LOW_POOL = (M_REF < M_THR).astype(np.float64), (M_POOL < M_THR).astype(np.float64)
print(f"victim margin threshold (ref {LOWQ:.0%} quantile) = {M_THR:.3f}; attacker pool low-margin share = {LOW_POOL.mean():.3f}")
EB = BANKS[0]; MEFF = EB["meff"]
if USE_GPU:
    MODES_T = torch.as_tensor(EB["modes"], dtype=torch.float64, device=DEV)
    MU_T = torch.as_tensor(EB["mu"], dtype=torch.complex128, device=DEV); WK_T = torch.as_tensor(EB["Wk"], dtype=torch.complex128, device=DEV)
rng0 = np.random.default_rng(SEED + 77); centers = Z_ref[rng0.choice(len(Z_ref), R_TOPICS, replace=False)]
if USE_GPU:
    Zt = torch.as_tensor(Z_ref, dtype=torch.float32, device=DEV); Ct = torch.as_tensor(centers, dtype=torch.float32, device=DEV)
    TOPIC_NN = torch.cdist(Ct, Zt).topk(NN_TOPIC, largest=False).indices.cpu().numpy(); del Zt
else:
    D_ = ((centers[:, None, :] - Z_ref[None, :, :]) ** 2).sum(-1); TOPIC_NN = np.argpartition(D_, NN_TOPIC, axis=1)[:, :NN_TOPIC]; del D_
print(f"topic regions {TOPIC_NN.shape}; low-margin share per topic region: median {np.median(LOW_REF[TOPIC_NN].mean(1)):.3f}, 90th pct {np.quantile(LOW_REF[TOPIC_NN].mean(1), 0.9):.3f}")

def gen_window(A_, rng, s):
    vols = np.clip(np.round(N0_MED * np.exp(VOL_SIGMA * rng.standard_normal(A_))), 5, 20 * N0_MED).astype(int)
    lab = np.repeat(np.arange(A_), vols); T_ = len(lab); idx = rng.integers(0, len(Z_ref), T_); topic = np.full(A_, -1)
    if s > 0:
        nclu = int(round(CLUSTER_FRAC * s * A_ / CLUSTER_SIZE))
        if nclu > 0:
            mem = rng.permutation(A_)[:nclu * CLUSTER_SIZE].reshape(nclu, CLUSTER_SIZE); topic[mem] = np.arange(nclu)[:, None]
            tid = rng.integers(0, R_TOPICS, nclu); tq = topic[lab]; m1 = (tq >= 0) & (rng.random(T_) < TOPIC_Q)
            idx[m1] = TOPIC_NN[tid[tq[m1]], rng.integers(0, NN_TOPIC, m1.sum())]
        trend = int(rng.integers(R_TOPICS)); fol = rng.random(A_) < TREND_FRAC * s
        m2 = fol[lab] & (rng.random(T_) < TREND_Q); idx[m2] = TOPIC_NN[trend, rng.integers(0, NN_TOPIC, m2.sum())]
    return idx, lab, vols, topic

def _fourier_t(pts, labels, nacc):
    Fr = torch.zeros((nacc, M_MODES), dtype=torch.float64, device=DEV); Fi = torch.zeros_like(Fr)
    for s0 in range(0, len(pts), CHUNK):
        P = torch.as_tensor(pts[s0:s0 + CHUNK], dtype=torch.float64, device=DEV)
        ix = torch.as_tensor(labels[s0:s0 + CHUNK], dtype=torch.long, device=DEV); th = (2.0 * math.pi) * (P @ MODES_T.T)
        Fr.index_add_(0, ix, torch.cos(th)); Fi.index_add_(0, ix, -torch.sin(th))
    return torch.complex(Fr, Fi)
def whitened_rows(pts, lab, cnt):
    if USE_GPU:
        Fr = _fourier_t(pts, lab, len(cnt)); c = torch.as_tensor(cnt, dtype=torch.float64, device=DEV)
        return ((Fr - c[:, None] * MU_T[None, :]) / torch.sqrt(c)[:, None]) @ WK_T
    Fr = _fourier_numpy(pts, lab, len(cnt), EB["modes"]); return ((Fr - cnt[:, None] * EB["mu"][None, :]) / np.sqrt(cnt)[:, None]) @ EB["Wk"]
def geo_stats(pts, lab, cnt, Kc=0, groups=()):
    Fw = whitened_rows(pts, lab, cnt); be = torch if (USE_GPU and torch.is_tensor(Fw)) else None
    if be is not None:
        c = torch.as_tensor(cnt, dtype=torch.float64, device=DEV); sq = torch.sqrt(c)
        rms2 = (Fw.abs() ** 2).mean(1).cpu().numpy(); lm = float(torch.linalg.eigvalsh(Fw.conj().T @ Fw / MEFF)[-1])
        uni = lambda ix: float(((Fw[ix] * sq[ix, None]).sum(0).abs() ** 2).mean() / c[ix].sum())
    else:
        sq = np.sqrt(cnt); rms2 = (np.abs(Fw) ** 2).mean(1); lm = float(np.linalg.eigvalsh(Fw.conj().T @ Fw / MEFF)[-1])
        uni = lambda ix: float((np.abs((Fw[ix] * sq[ix, None]).sum(0)) ** 2).mean() / cnt[ix].sum())
    out = dict(lm=lm, r2=rms2, ga=uni(np.arange(len(cnt))), S_un=uni(np.arange(Kc)) if Kc > 0 else float("nan"))
    out["S_groups"] = [uni(np.array(g)) for g in groups]
    return out
def dir_stats(low, lab, cnt):
    c_ = np.bincount(lab, weights=low, minlength=len(cnt)); z = (c_ - LOWQ * cnt) / np.sqrt(LOWQ * (1 - LOWQ) * cnt)
    zg = (c_.sum() - LOWQ * cnt.sum()) / math.sqrt(LOWQ * (1 - LOWQ) * cnt.sum())
    return dict(z=z, zg=float(zg), zsum=float((np.maximum(z, 0) ** 2).sum()))

def calibrate(recs_g, recs_d):
    q_ = lambda a: float(np.quantile(a, 1 - ALPHA_D)); ns = np.sort(np.concatenate([r["r2"] for r in recs_g]))
    nz = np.sort(np.concatenate([r["z"] for r in recs_d]))
    return dict(tmax=q_([r["lm"] for r in recs_g]), tglo=q_([r["ga"] for r in recs_g]), thc=q_([hc_stat(r["r2"], ns) for r in recs_g]), ns=ns,
                dz_hc=q_([hc_stat(r["z"], nz) for r in recs_d]), dz_g=q_([r["zg"] for r in recs_d]), dz_sum=q_([r["zsum"] for r in recs_d]), nz=nz)
def decide8(g, d, T_):
    out = dict(spec=g["lm"] > T_["tmax"], hc=hc_stat(g["r2"], T_["ns"]) > T_["thc"], glob=g["ga"] > T_["tglo"],
               d_hc=hc_stat(d["z"], T_["nz"]) > T_["dz_hc"], d_glob=d["zg"] > T_["dz_g"], d_sum=d["zsum"] > T_["dz_sum"])
    out["geo_suite"] = out["spec"] or out["hc"] or out["glob"]; out["dir_suite"] = out["d_hc"] or out["d_glob"] or out["d_sum"]
    out["both"] = out["geo_suite"] or out["dir_suite"]; return out
KEYS8 = ["geo_suite", "dir_suite", "both", "spec", "hc", "glob", "d_hc", "d_glob", "d_sum"]

def attack_window(pool_idx, A_, K_, rng, s):
    idx, lab, vols, topic = gen_window(A_, rng, s); keep = lab >= K_; idx, lab = idx[keep], lab[keep]
    n_per = len(pool_idx) // K_; pi = pool_idx[rng.permutation(len(pool_idx))][: K_ * n_per]; lab_c = np.arange(K_ * n_per) % K_
    pts = np.vstack([Z_pool[pi], Z_ref[idx]]); low = np.concatenate([LOW_POOL[pi], LOW_REF[idx]]); labs = np.concatenate([lab_c, lab])
    return pts, low, labs, np.bincount(labs, minlength=A_)
RESULTS8 = {"config": dict(QUICK=QUICK, MEFF=MEFF, STRENGTHS=STRENGTHS, A8=A8, K8={str(k): v for k, v in K8.items()}, N8=N8, N_CAL8=N_CAL8, M_THR=M_THR)}

# %% [markdown]
# ## Part A — null calibration across structure strength

# %%
# ===== Part A: dose-response calibration + null FPR at each structure strength =====
t0 = time.time(); CAL8 = {}; rowsA = []
for A_ in A8:
    for s in STRENGTHS:
        rg, rd = [], []
        for t in range(N_CAL8):
            r = np.random.default_rng(SEED + 81_000_000 + 1000 * A_ + int(1000 * s) * 7 + t); idx, lab, vols, _ = gen_window(A_, r, s)
            cnt = np.bincount(lab, minlength=A_); rg.append(geo_stats(Z_ref[idx], lab, cnt)); rd.append(dir_stats(LOW_REF[idx], lab, cnt))
        CAL8[(A_, s)] = T_ = calibrate(rg, rd); fp = {k: 0 for k in KEYS8}
        for t in range(N_TEST8):
            r = np.random.default_rng(SEED + 82_000_000 + 1000 * A_ + int(1000 * s) * 7 + t); idx, lab, vols, _ = gen_window(A_, r, s)
            cnt = np.bincount(lab, minlength=A_); d = decide8(geo_stats(Z_ref[idx], lab, cnt), dir_stats(LOW_REF[idx], lab, cnt), T_)
            for k in KEYS8: fp[k] += int(d[k])
        rowsA.append(dict(A=A_, s=s, **{k: v / N_TEST8 for k, v in fp.items()}))
        # [PAPER] App. E 'The dose–response study': 0.6% false alarms at s = 1 (plausibly this geo_suite value; see header)
        print(f"A={A_:5d} s={s:.2f}: null FPR " + " ".join(f"{k}={rowsA[-1][k]:.3f}" for k in ["geo_suite", "dir_suite", "both"]) + f"  ({time.time()-t0:.0f}s)")
RESULTS8["A"] = rowsA

# %% [markdown]
# ## Part B — power vs structure strength

# %%
# ===== Part B: power vs benign structure strength =====
t0 = time.time(); rowsB = []
for A_ in A8:
    K_ = K8[A_]
    for name in ["random", "entropy", "kcenter"]:
        for N in N8:
            for s in STRENGTHS:
                hits = {k: 0 for k in KEYS8}; n = 0; Su, lowsh = [], []
                for tr_ in range(TRIALS):
                    pidx = CAMPAIGNS[(name, tr_)][0][:N]; lowsh.append(LOW_POOL[pidx].mean())
                    for w in range(W8):
                        r = np.random.default_rng(SEED + 83_000_000 + 1000 * tr_ + w + 7 * A_ + N + int(1000 * s))
                        pts, low, lab, cnt = attack_window(pidx, A_, K_, r, s)
                        g = geo_stats(pts, lab, cnt, K_); d = decide8(g, dir_stats(low, lab, cnt), CAL8[(A_, s)])
                        for k in KEYS8: hits[k] += int(d[k])
                        n += 1; Su.append(g["S_un"])
                row = dict(A=A_, K=K_, strategy=name, N=N, s=s, S_un=float(np.mean(Su)), low_share=float(np.mean(lowsh)), n=n, **{k: v / n for k, v in hits.items()})
                rowsB.append(row)
            # [PAPER] Table 4 (all rows, N = 10,000, 24 windows per cell): geo_suite = "geometry", dir_suite = "task-aware"; Fig. 3 (geometry power vs strength)
            print(f"A={A_:5d} {name:>8} N={N:5d} low-share {row['low_share']:.3f} (ref {LOWQ}) | geo/dir by s: " +
                  " ".join(f"s{r['s']}:{r['geo_suite']:.2f}/{r['dir_suite']:.2f}" for r in rowsB[-len(STRENGTHS):]) + f"  ({time.time()-t0:.0f}s)")
RESULTS8["B"] = rowsB

# %% [markdown]
# ## Part C — diagnostic and predictions

# %%
# ===== Part C: diagnostic — benign clusters vs attacks at full structure (s = 1) =====
A_ = A8[0]; s = max(STRENGTHS); clus_S, clus_low = [], []
for t in range(30 if not QUICK else 5):
    r = np.random.default_rng(SEED + 84_000_000 + t); idx, lab, vols, topic = gen_window(A_, r, s)
    groups = [np.where(topic == g)[0] for g in range(topic.max() + 1)] if topic.max() >= 0 else []
    cnt = np.bincount(lab, minlength=A_); g = geo_stats(Z_ref[idx], lab, cnt, groups=groups); clus_S += g["S_groups"]
    for gr in groups: m_ = np.isin(lab, gr); clus_low.append(LOW_REF[idx][m_].mean())
att_S = {n_: float(np.mean([r["S_un"] for r in rowsB if r["strategy"] == n_ and r["s"] == 0.0 and r["A"] == A_ and r["N"] == max(N8)])) for n_ in ["entropy", "kcenter"]}
att_low = {n_: float(np.mean([r["low_share"] for r in rowsB if r["strategy"] == n_ and r["A"] == A_ and r["N"] == max(N8)])) for n_ in ["entropy", "kcenter"]}
# [PAPER] App. E 'The dose–response study': benign ten-account clusters union S̄ 83–159, against 26–30 for the attacks
# [PAPER] Prediction log (Step 8, "Clusters out-shine attacks in geometry, not margin"): Partial (geometry yes, margin no)
print(f"benign 10-account clusters (n={len(clus_S)}): union S median {np.median(clus_S):.1f}, 90th pct {np.quantile(clus_S, .9):.1f} | low-margin share median {np.median(clus_low):.3f}, 90th pct {np.quantile(clus_low, .9):.3f}")
print(f"attacks at N={max(N8)}: union S " + ", ".join(f"{k} {v:.1f}" for k, v in att_S.items()) + " | low-margin share " + ", ".join(f"{k} {v:.3f}" for k, v in att_low.items()))
RESULTS8["C"] = dict(clusS_med=float(np.median(clus_S)), clusS_p90=float(np.quantile(clus_S, .9)), clusLow_med=float(np.median(clus_low)),
                     clusLow_p90=float(np.quantile(clus_low, .9)), attS=att_S, attLow=att_low)
# ---- predictions
def s50(rows):
    rows = sorted(rows, key=lambda r: r["s"]); xs, ys = [r["s"] for r in rows], [r["geo_suite"] for r in rows]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 >= 0.5 > y1: return x0 + (y0 - 0.5) * (x1 - x0) / (y0 - y1)
    return float("nan")
for A_ in A8:
    for n_ in ["entropy", "kcenter"]:
        for N in N8:
            rr = [r for r in rowsB if r["A"] == A_ and r["strategy"] == n_ and r["N"] == N]
            # [PAPER] Prediction log (Step 8): "Geometry breaks at s ∈ [0.05, 0.5]" Partial (0.175 at A = 100; ≤ 0.06 at 1,000) via s50; "Task-aware suite ≥ 0.8 at s = 1" Fail (≤ 0.17) via dir s=1
            print(f"P8.1/P8.2 A={A_} {n_} N={N}: geo s=0 {rr[0]['geo_suite']:.2f} -> s=1 {rr[-1]['geo_suite']:.2f} (s50 {s50(rr):.3f}) | dir s=1 {rr[-1]['dir_suite']:.2f} | both s=1 {rr[-1]['both']:.2f}")
# [PAPER] Prediction log (Step 8, "Calibration ≤ 1.5× nominal"): Partial (mean 1.1%, worst 3.3%); worst printed here, mean not isolated
print(f"P8.3 max null FPR: geo {max(r['geo_suite'] for r in rowsA):.3f} dir {max(r['dir_suite'] for r in rowsA):.3f} (nominal <= {ALPHA}); "
      f"random-campaign max: geo {max(r['geo_suite'] for r in rowsB if r['strategy']=='random'):.3f} dir {max(r['dir_suite'] for r in rowsB if r['strategy']=='random'):.3f}")

# %% [markdown]
# ## Figure

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, len(A8), figsize=(8 * len(A8), 5), squeeze=False)
for i, A_ in enumerate(A8):
    for n_, c_ in [("entropy", "tab:orange"), ("kcenter", "tab:green"), ("random", "tab:gray")]:
        rr = sorted([r for r in RESULTS8["B"] if r["A"] == A_ and r["strategy"] == n_ and r["N"] == max(N8)], key=lambda r: r["s"])
        xs = [max(r["s"], 0.01) for r in rr]
        ax[0, i].plot(xs, [r["geo_suite"] for r in rr], "o-", color=c_, label=f"{n_}: geometry suite")
        ax[0, i].plot(xs, [r["dir_suite"] for r in rr], "s--", color=c_, label=f"{n_}: directional (low-margin) suite")
    ax[0, i].set_xscale("log"); ax[0, i].set_xlabel("benign structure strength s (0 plotted at 0.01)"); ax[0, i].set_ylabel("power")
    ax[0, i].set_title(f"A={A_}, K={K8[A_]}, N={max(N8)}"); ax[0, i].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/step8_figure.png", dpi=130); plt.show()

# %% [markdown]
# ## VERDICT REPORT — copy into the chat

# %%
print("=" * 86); print("STEP 8 VERDICT REPORT"); print("=" * 86); print("config:", json.dumps(RESULTS8["config"]))
for r in RESULTS8["A"]: print(f"[null] A={r['A']:5d} s={r['s']:.2f}: " + " ".join(f"{k}={r[k]:.3f}" for k in KEYS8))
for r in RESULTS8["B"]: print(f"[B] A={r['A']:5d} {r['strategy']:>8} N={r['N']:5d} s={r['s']:.2f} S_un={r['S_un']:.2f} low={r['low_share']:.3f} " + " ".join(f"{k}={r[k]:.2f}" for k in KEYS8))
print("[C]", json.dumps(RESULTS8["C"])); print("=" * 86)
with open(f"{OUTDIR}/step8_results.json", "w") as fh: json.dump(RESULTS8, fh, indent=1, default=float)
