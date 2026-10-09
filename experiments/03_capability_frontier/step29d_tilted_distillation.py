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
# | App. Table 14, row "SmolLM2-360M, tilted (least detectable)", col "Half natural, mixed" | tilted_c05 gain at 3,000 / 10,000 queries | +2.2 ± 0.2 / +1.5 ± 0.2 | Cell 7, summary print "[A] tilted_c05 n=..." |
# | App. Table 14, tilted rows, col "Paired cost" | cost against the all-math campaign of the same size | −8.2 / −14.4 | not isolated: tilted gain here minus Step 19's all-math gain (+10.4, +15.9) |
# | §6 "The covert end"; App. G.2; Fig. 1a and Fig. 6a ("least detectable (tilted)") | gain at 3,000 queries, against 10.4; covertness cost | 2.2 points; 79% of the gain | Cell 7, tilted_c05 n = 3000; 79% = 1 − 2.2/10.4 with Step 19's reference (not computed here) |
# | §6 and App. G.2 "The frontier, in one sentence" | upper end of the share of gain given up by campaigns that evaded every detector | 58–79% | upper end only (79%), as above; the 58% end comes from Step 32 |
# | App. H (text) | on-task prompts in the attacker's pool; on-task share of the 10,000-query tilted campaign | 2,933; 29% on task | Cell 4, `len(ON_PAIRS)` in the "task region" print; the 29% follows from `tilted()` capping on-task draws at `len(ON_PAIRS)` |
# | Prediction log (Step 29), "Base-rate on-task prompts distil (≥ 8 points at 10k)" | P29d | Fail (+1.5) | Cell 9 "P29d" print |
#
# Not reported in the paper: the `tilted_c1` (all on-task) runs; the benign task-region share `P_TASK` (Prediction log Step 29 "Task region rare" reports 1.48% from the detection notebook, step27_31_bulk_edge_multioutlier_tilted.py, not from this script); the detection rates of the tilted campaign (from the Steps 27-31 and 35 notebooks); Part B (`RUN_B = False`; code carried over from Step 19, not run here).

# %% [markdown]
# # Step 29d — Do base-rate on-task prompts distil? The least-favourable campaign in the extraction regime
# **Kaggle:** GPU + Internet + `HF_TOKEN` → **Run All** (≈2–2.5 h; reuses the Step 11/14/19 caches). `QUICK=True` first.
#
# Proposition 4 says the least-detectable campaign is an exponential tilt of benign traffic: its on-task queries are what benign users already ask in the task region. Here the task region S is where a logistic MetaMathQA-vs-WildChat classifier gives posterior ≥ 0.5 (the same definition as the detection notebook). The tilted campaign takes its on-task prompts from attacker-pool WildChat prompts in S, **with their recorded ChatGPT replies**, and its off-task prompts from the rest of the pool.
#
# * **tilted_c05** — half on-task (natural, in S), half off-task; budgets 3,000 and 10,000; two seeds.
# * **tilted_c1** — all on-task natural prompts (up to the number available); two seeds.
# * Student: SmolLM2-360M, full fine-tuning, GSM8K (first 400 problems) — exactly the extraction-regime setup of Step 19, whose MetaMathQA campaigns gained +15.9 (all math) and +12.1 (half padding) at 10,000 queries.
#
# | ID | Locked prediction (listed in the prediction log (docs/prediction_log.md), Step 29) |
# |---|---|
# | **P29d** | Base-rate prompts still distil: tilted_c05 gain at 10,000 queries ≥ 8 points |

# %%
# ===== Cell 1: configuration =====
import math, json, time, os, gc, re, pickle
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
QUICK = False
SYNTHETIC_TEST = False
RUN_A, RUN_B = True, False
SEED = 9091
OUTDIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
import torch
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); USE_GPU = DEV.type == "cuda"
if not USE_GPU: raise RuntimeError("No GPU visible. Kaggle: Settings -> Accelerator -> GPU, then restart.")
POOL_FRAC = 0.2
STUDENT_A, STUDENT_B = "HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-1.5B"
if QUICK:
    BUD_A, SEEDS_A, N_EVAL_A, BUD_B, N_EVAL_B = [60, 180], [0], 40, [60], 16
    MAX_LEN, BS, GEN_BS, MAX_NEW_A, MAX_NEW_B, LR_A = 384, 8, 16, 128, 128, 5e-5
else:
    BUD_A, SEEDS_A, N_EVAL_A, BUD_B, N_EVAL_B = [3000, 10000], [0, 1], 400, [1000, 4000], 500
    MAX_LEN, BS, GEN_BS, MAX_NEW_A, MAX_NEW_B, LR_A = 512, 16, 32, 320, 512, 5e-5
BS_B = 8
CACHE11 = f"{OUTDIR}/step11_wildchat_E384_{'quick' if QUICK else 'full'}.npz"
CACHE14 = f"{OUTDIR}/step14_meta_{'quick' if QUICK else 'full'}.npz"
CACHE19 = f"{OUTDIR}/step19_mm_rows_{'quick' if QUICK else 'full'}.pkl"
N_CONV = 30000 if QUICK else None; DIMS = [384]; D_MMU = None
print(f"GPU {torch.cuda.get_device_name(0)} | QUICK={QUICK} | A: {STUDENT_A} budgets {BUD_A} seeds {SEEDS_A} | B: {STUDENT_B} budgets {BUD_B}")

# %% [markdown]
# ## Part 0 — caches (natural WildChat pairs) and MetaMathQA rows

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
# ===== Cell 4: all MetaMathQA rows by type (GSM / MATH) and natural pairs =====
if os.path.exists(CACHE19):
    with open(CACHE19, "rb") as fh: GSM_ROWS, MATH_ROWS = pickle.load(fh)
else:
    from datasets import load_dataset
    mmq = load_dataset("meta-math/MetaMathQA", split="train").select_columns(["query", "response", "type"])
    GSM_ROWS, MATH_ROWS = [], []
    for q, a, ty in zip(mmq["query"], mmq["response"], mmq["type"]):
        if q and a and ty: (GSM_ROWS if ty.startswith("GSM") else MATH_ROWS).append((q.strip()[:1000], a.strip()[:2000]))
    with open(CACHE19, "wb") as fh: pickle.dump((GSM_ROWS, MATH_ROWS), fh)
NATURAL = [POOL_TXT[i] for i in sorted(POOL_TXT) if POOL_TXT[i][0] and POOL_TXT[i][1]]
from sklearn.linear_model import LogisticRegression
neg = np.flatnonzero((~is_pool[ACCT]) & (WEEK < 8)); neg = np.random.default_rng(4).choice(neg, min(4 * len(D_MMU), len(neg)), replace=False)
clf_t = LogisticRegression(max_iter=2000, class_weight="balanced").fit(np.vstack([D_MMU.astype(np.float32), E16[neg].astype(np.float32)]), np.r_[np.ones(len(D_MMU)), np.zeros(len(neg))])
pool_keys = np.array(sorted(i for i in POOL_TXT if POOL_TXT[i][0] and POOL_TXT[i][1]))
post = clf_t.predict_proba(E16[pool_keys].astype(np.float32))[:, 1]
ON_PAIRS = [POOL_TXT[i] for i in pool_keys[post >= 0.5]]; OFF_PAIRS = [POOL_TXT[i] for i in pool_keys[post < 0.5]]
ben = np.flatnonzero(~is_pool[ACCT]); smp = np.random.default_rng(5).choice(ben, min(200000, len(ben)), replace=False)
P_TASK = float(np.mean(clf_t.predict_proba(E16[smp].astype(np.float32))[:, 1] >= 0.5))
# [PAPER] App. H: the attacker's pool held only 2,933 on-task prompts (so the 10,000-query tilted campaign was 29% on task)
print(f"task region: benign share p = {P_TASK:.4f}; attacker-pool on-task pairs {len(ON_PAIRS)}, off-task {len(OFF_PAIRS)}")
def tilted(var, n, rng):
    if var == "tilted_c1":
        k = min(n, len(ON_PAIRS)); return [ON_PAIRS[i] for i in rng.choice(len(ON_PAIRS), k, replace=False)]
    k = min((n + 1) // 2, len(ON_PAIRS))
    return [ON_PAIRS[i] for i in rng.choice(len(ON_PAIRS), k, replace=False)] + [OFF_PAIRS[i] for i in rng.choice(len(OFF_PAIRS), n - k, replace=False)]
print(f"MetaMathQA rows: GSM-type {len(GSM_ROWS)}, MATH-type {len(MATH_ROWS)}; natural pairs {len(NATURAL)}")
def campaign(rows_, var, n, rng):
    if var == "unique": return [rows_[i] for i in rng.choice(len(rows_), n, replace=False)]
    return [rows_[i] for i in rng.choice(len(rows_), (n + 1) // 2, replace=False)] + [NATURAL[i] for i in rng.choice(len(NATURAL), n // 2, replace=False)]


# %% [markdown]
# ## Training and evaluation utilities

# %%
# ===== Cell 5: LoRA fine-tuning and evaluation (merged fp16 weights for fast generation) =====
import sys, subprocess, importlib
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
for m in [m for m in list(sys.modules) if m in ("peft", "torchao") or m.startswith(("peft.", "torchao."))]: del sys.modules[m]
importlib.invalidate_caches()
try:
    import peft
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"]); importlib.invalidate_caches(); import peft
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from datasets import load_dataset
fmtq = lambda q: f"Question: {q.strip()}\nAnswer:"
def train_lora(student, pairs, seed, tok):
    torch.manual_seed(seed); model = AutoModelForCausalLM.from_pretrained(student, torch_dtype=torch.float32).to(DEV)
    model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4); steps = max(1, math.ceil(len(pairs) / BS))
    sch = get_linear_schedule_with_warmup(opt, max(1, steps // 20), steps); scaler = torch.cuda.amp.GradScaler()
    tok.padding_side = "right"; model.train(); order = np.random.default_rng(seed).permutation(len(pairs))
    for s0 in range(0, len(pairs), BS):
        batch = [pairs[k] for k in order[s0:s0 + BS]]
        enc = tok([fmtq(q) + " " + a.strip() + tok.eos_token for q, a in batch], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
        labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
        for k, (q, _) in enumerate(batch): labels[k, : len(tok(fmtq(q))["input_ids"])] = -100
        with torch.autocast("cuda", dtype=torch.float16): loss = model(**enc, labels=labels).loss
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
    model = model.merge_and_unload().half(); model.config.use_cache = True; return model
@torch.no_grad()
def generate(model, tok, questions, max_new):
    model.eval(); tok.padding_side = "left"; outs = []
    for i in range(0, len(questions), GEN_BS):
        enc = tok([fmtq(q) for q in questions[i:i + GEN_BS]], return_tensors="pt", padding=True).to(DEV)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).split("\nQuestion:")[0] for o in out]
    return outs
def last_number(t):
    m = re.search(r"answer is:?\s*\$?(-?[\d,]*\.?\d+)", t); s = m.group(1) if m else (re.findall(r"-?[\d,]*\.?\d+", t) or [None])[-1]
    try: return float(s.replace(",", "")) if s else None
    except ValueError: return None
def norm_math(a):
    if a is None: return None
    a = a.strip().strip("$").replace("\\left", "").replace("\\right", "").replace(" ", "").replace("\\!", "").replace("dfrac", "frac").replace("tfrac", "frac")
    a = re.sub(r"\\text\{(.*?)\}", r"\1", a); a = a.rstrip(".")
    try: return str(float(a)).rstrip("0").rstrip(".") if re.fullmatch(r"-?[\d.]+", a) else a
    except ValueError: return a
def extract_math(t):
    m = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", t)
    if m: return m[-1]
    m = re.search(r"answer is:?\s*(.+)", t); return m.group(1).split("\n")[0] if m else None


# %%
# ===== Cell 6: full fine-tuning for the small student =====
def train_full(student, pairs, seed, tok, lr):
    torch.manual_seed(seed); model = AutoModelForCausalLM.from_pretrained(student, torch_dtype=torch.float32).to(DEV)
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0); steps = max(1, math.ceil(len(pairs) / BS))
    steps = max(1, math.ceil(len(pairs) / BS)); sch = get_linear_schedule_with_warmup(opt, max(1, steps // 30), steps); scaler = torch.cuda.amp.GradScaler()
    tok.padding_side = "right"; model.train(); order = np.random.default_rng(seed).permutation(len(pairs))
    MICRO, ACC = 4, max(1, BS // 4); starts = list(range(0, len(pairs), MICRO))
    for i, s0 in enumerate(starts):
        batch = [pairs[k] for k in order[s0:s0 + MICRO]]
        enc = tok([fmtq(q) + " " + a.strip() + tok.eos_token for q, a in batch], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
        labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
        for k, (q, _) in enumerate(batch): labels[k, : len(tok(fmtq(q))["input_ids"])] = -100
        with torch.autocast("cuda", dtype=torch.float16): loss = model(**enc, labels=labels).loss / ACC
        scaler.scale(loss).backward(); del enc, labels, loss
        if (i + 1) % ACC == 0 or i == len(starts) - 1:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step()
    model = model.half(); model.config.use_cache = True; return model


# %% [markdown]
# ## Part A — a student that lacks the capability

# %%
# ===== Cell 7: Part A — SmolLM2-360M on GSM8K, budgets 1k-10k, unique vs mixed =====
RA, BASE_A = {}, None
if RUN_A:
    tokA = AutoTokenizer.from_pretrained(STUDENT_A); tokA.pad_token = tokA.pad_token or tokA.eos_token
    gsm = load_dataset("openai/gsm8k", "main", split="test").select(range(N_EVAL_A))
    GQ = list(gsm["question"]); GOLD = [float(a.split("####")[-1].strip().replace(",", "")) for a in gsm["answer"]]
    def gsm_correct(model, tok):
        preds = [last_number(t) for t in generate(model, tok, GQ, MAX_NEW_A)]
        return np.array([p is not None and abs(p - g) < 1e-4 for p, g in zip(preds, GOLD)], float)
    t0 = time.time(); bm = AutoModelForCausalLM.from_pretrained(STUDENT_A, torch_dtype=torch.float16).to(DEV); BASE_A = gsm_correct(bm, tokA); del bm; torch.cuda.empty_cache()
    print(f"[A] base {STUDENT_A} GSM8K accuracy {BASE_A.mean():.3f} on {N_EVAL_A} problems ({time.time()-t0:.0f}s)")
    for n in BUD_A:
        for var in ["tilted_c05", "tilted_c1"]:
            if var == "tilted_c1" and n > len(ON_PAIRS) and n != BUD_A[0]: continue
            for sd in SEEDS_A:
                pairs = tilted(var, n, np.random.default_rng(SEED + 1000 * sd + n))
                if var == "tilted_c1" and len(pairs) < n: print(f"[A] tilted_c1: only {len(pairs)} on-task pairs available for budget {n}")
                model = train_full(STUDENT_A, pairs, sd, tokA, LR_A); RA[(var, n, sd)] = gsm_correct(model, tokA); del model; gc.collect(); torch.cuda.empty_cache()
                print(f"[A] {var:>7} n={n:6d} seed={sd}: GSM8K {RA[(var, n, sd)].mean():.3f} (gain {100*(RA[(var, n, sd)].mean()-BASE_A.mean()):+.1f} pt)  ({time.time()-t0:.0f}s)")
    for (v, n) in sorted(set((k[0], k[1]) for k in RA)):
        g = [100 * (RA[(v, n, s)].mean() - BASE_A.mean()) for s in SEEDS_A if (v, n, s) in RA]
        # [PAPER] App. Table 14 (tilted rows): +2.2 ± 0.2 (3,000), +1.5 ± 0.2 (10,000); paired cost −8.2 / −14.4 uses Step 19's all-math gains
        # [PAPER] §6 "The covert end", App. G.2, Fig. 1a, Fig. 6a: 2.2 points at 3,000 queries against 10.4 (Step 19): covertness cost 79% of the gain
        print(f"[A] {v} n={n}: gain {np.mean(g):+.1f} (sd {np.std(g):.1f}) over {len(g)} seeds")

# %% [markdown]
# ## Part B — MATH-500 with a 1.5B student

# %%
# ===== Cell 8: Part B — Qwen2.5-1.5B + LoRA on MATH-500 (memory-safe: fp16 frozen base, fp32 LoRA, micro-batches) =====
for _n in ["model", "bm"]:
    if _n in globals(): del globals()[_n]
gc.collect(); torch.cuda.empty_cache()
MICRO_B, ACCUM_B, MAX_LEN_B = 2, 4, 512
def train_lora_lowmem(student, pairs, seed, tok):
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(student, torch_dtype=torch.float16).to(DEV)
    model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-4); steps = max(1, math.ceil(len(pairs) / (MICRO_B * ACCUM_B)))
    sch = get_linear_schedule_with_warmup(opt, max(1, steps // 20), steps); scaler = torch.cuda.amp.GradScaler()
    tok.padding_side = "right"; model.train(); order = np.random.default_rng(seed).permutation(len(pairs)); starts = list(range(0, len(pairs), MICRO_B))
    for i, s0 in enumerate(starts):
        batch = [pairs[k] for k in order[s0:s0 + MICRO_B]]
        enc = tok([fmtq(q) + " " + a.strip() + tok.eos_token for q, a in batch], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN_B).to(DEV)
        labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
        for k, (q, _) in enumerate(batch): labels[k, : len(tok(fmtq(q))["input_ids"])] = -100
        with torch.autocast("cuda", dtype=torch.float16): loss = model(**enc, labels=labels).loss / ACCUM_B
        scaler.scale(loss).backward(); del enc, labels, loss
        if (i + 1) % ACCUM_B == 0 or i == len(starts) - 1:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step()
    model = model.merge_and_unload(); model.config.use_cache = True; model.eval(); return model
RB, BASE_B = {}, None
if RUN_B:
    tokB = AutoTokenizer.from_pretrained(STUDENT_B); tokB.pad_token = tokB.pad_token or tokB.eos_token
    m5 = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(N_EVAL_B)); MQ, MA = list(m5["problem"]), [norm_math(a) for a in m5["answer"]]
    def math_correct(model):
        preds = generate(model, tokB, MQ, MAX_NEW_B); return np.array([norm_math(extract_math(p)) == a for p, a in zip(preds, MA)], float)
    t0 = time.time(); bm = AutoModelForCausalLM.from_pretrained(STUDENT_B, torch_dtype=torch.float16).to(DEV); BASE_B = math_correct(bm); del bm; gc.collect(); torch.cuda.empty_cache()
    print(f"[B] base {STUDENT_B} MATH-500 accuracy {BASE_B.mean():.3f} ({time.time()-t0:.0f}s)")
    for n in BUD_B:
        for var in ["unique", "mixed"]:
            pairs = campaign(MATH_ROWS, var, n, np.random.default_rng(SEED + 77 + n))
            model = train_lora_lowmem(STUDENT_B, pairs, 0, tokB); RB[(var, n)] = math_correct(model); del model; gc.collect(); torch.cuda.empty_cache()
            print(f"[B] {var:>7} n={n:5d}: MATH-500 {RB[(var, n)].mean():.3f} (gain {100*(RB[(var, n)].mean()-BASE_B.mean()):+.1f} pt)  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Predictions, figure, VERDICT REPORT — copy into the chat

# %%
# ===== Cell 9: prediction and verdict =====
GA_ = lambda v, n: float(np.mean([100 * (RA[(v, n, s)].mean() - BASE_A.mean()) for s in SEEDS_A if (v, n, s) in RA]))
nmax = max(BUD_A)
# [PAPER] Prediction log (Step 29, "Base-rate on-task prompts distil"): Fail (+1.5)
print(f"P29d tilted_c05 at {nmax} queries: gain {GA_('tilted_c05', nmax):+.1f} (pass if >= 8; Step 19 references: all-math +15.9, half padding +12.1)")
print(f"[context] p = {P_TASK:.4f}; on-task pairs {len(ON_PAIRS)}; base GSM8K {BASE_A.mean():.4f}")
print("=" * 90); print("STEP 29d VERDICT REPORT"); print("=" * 90)
for k, v in RA.items(): print(f"[A] {k}: {v.mean():.4f}")
print("=" * 90)
with open(f"{OUTDIR}/step29d_results.json", "w") as fh: json.dump(dict(A={str(k): float(v.mean()) for k, v in RA.items()}, base=float(BASE_A.mean()), p=P_TASK, n_on=len(ON_PAIRS)), fh, indent=1)
