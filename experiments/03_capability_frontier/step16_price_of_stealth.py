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
# | App. Table 13, col "All math" (Qwen2.5-0.5B, three seeds, 400 problems) | GSM8K gain at 100 / 300 / 900 queries | +9.8 ± 2.1 / +7.5 ± 0.5 / +8.6 ± 1.2 | Cell 6, `summary("unique", n)`; print "n=...: gains" |
# | App. Table 13, col "Half natural, mixed" | GSM8K gain at 100 / 300 / 900 queries | +9.8 ± 1.8 / +10.2 ± 2.1 / +10.1 ± 1.0 | Cell 6, `summary("mixed", n)` |
# | App. Table 13, col "Camouflaged (natural-looking math)" | GSM8K gain at 100 / 300 / 900 queries | +9.3 ± 0.7 / +8.7 ± 2.3 / +11.2 ± 1.9 | Cell 6, `summary("camouflage", n)` |
# | §6 "Two regimes"; App. G.1; Fig. 6b (elicitation, Qwen2.5-0.5B, three seeds) | gain from 100 math prompts, and no more from 900; padding costs nothing measurable | 9.8 points | Cell 6 (`G`); figure in Cell 8 |
# | Prediction log (Step 16), "All-math gains grow ≥ 3 points from 100 to 900 queries" | P16.1 | Fail (+9.8 to +8.6) | Cell 8 "P16.1" print |
# | Prediction log (Step 16), "Mixed gains ≤ all-math − 2 at 300 and 900" | P16.2 (paired mixed − unique) | Fail (+2.7, +1.5) | Cell 6 `paired_diff("mixed", "unique", n)`; Cell 8 "P16.2" print |
# | Prediction log (Step 16), "Camouflage gains ≤ 80% of all-math at 900" | P16.3 | Fail (130%) | Cell 8 "P16.3" print (ratio) |
# | Prediction log (Step 16), "MATH-500, 1.5B student: price of stealth" | P16.4 | Untestable (not run) | Cell 7 (Part H; `RUN_MATH = False`); MATH-500 was measured in Step 19 Part B instead |
#
# Not reported in the paper: the base Qwen2.5-0.5B accuracy on these 400 problems (App. Table 12's 34.4% is Step 14's, on 500 problems); seed SDs and item SEs of the paired differences; camouflage − unique paired differences; Part H (not run); `step16_figure.png`.

# %% [markdown]
# # Step 16 — The price of stealth where it can be non-zero
# **Kaggle:** GPU (T4/P100) + Internet + `HF_TOKEN` → **Run All**. Part G ≈ 1.5–2 h (27 LoRA fine-tunes + GSM8K evaluations). Part H (MATH, 1.5B student) is **off by default** (`RUN_MATH = True` adds ≈ 2 h). Reuses the Step 11 and Step 14 caches. `QUICK=True` first (≈ 10 min).
#
# Step 14 measured the price of mimicry at ≈ 820 MetaMathQA pairs, where GSM8K gains had already saturated, with one seed. Here:
# * **Below saturation:** budgets of 100, 300 and 900 queries of **GSM8K-type MetaMathQA** pairs (one per source problem).
# * **Equal query budgets, three variants:** `unique` (all campaign queries are math), `mixed` (half are natural WildChat prompts with their ChatGPT replies, as in within-account padding), `camouflage` (all math, but only the 20% of prompts that look most like WildChat users, i.e. the attacker makes the campaign queries themselves look natural).
# * **Three seeds** (data sampling and training order); every model is evaluated on the **same** GSM8K problems, so differences are paired.
# * **Part H (optional):** MATH-type MetaMathQA pairs, `Qwen/Qwen2.5-1.5B`, evaluation on MATH-500, budgets 500 and 2,000, `unique` vs `mixed`, two seeds.
#
# | ID | Locked prediction |
# |---|---|
# | **P16.1** | Non-saturated regime: `unique` gains grow by ≥ 3 points from 100 to 900 queries (seed means) |
# | **P16.2** | Price of stealth below saturation: at 300 and 900 queries, `mixed` gains ≤ `unique` − 2 points (seed means; paired) |
# | **P16.3** | Camouflage costs capability: at 900 queries, `camouflage` gains ≤ 80% of `unique` |
# | **P16.4** | (Part H) MATH-500: `unique` at 2,000 improves on the base model by ≥ 3 points, and `mixed` gains ≤ 70% of `unique` |

# %%
# ===== Cell 1: configuration =====
import math, json, time, os, gc, re, pickle
import numpy as np
QUICK = False
SYNTHETIC_TEST = False
RUN_MATH = False
SEED = 9091
OUTDIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
import torch
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); USE_GPU = DEV.type == "cuda"
if not USE_GPU: raise RuntimeError("No GPU visible. Kaggle: Settings -> Accelerator -> GPU, then restart.")
POOL_FRAC, CAMO_FRAC = 0.2, 0.2
STUDENT, STUDENT_MATH = "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"
if QUICK:
    BUDGETS, SEEDS, VARIANTS, N_EVAL, MAX_LEN, BS, GEN_BS, MAX_NEW = [40, 120], [0], ["unique", "mixed", "camouflage"], 24, 384, 4, 8, 96
    MATH_BUDGETS, MATH_SEEDS, N_EVAL_MATH, MAX_NEW_MATH = [60], [0], 16, 128
else:
    BUDGETS, SEEDS, VARIANTS, N_EVAL, MAX_LEN, BS, GEN_BS, MAX_NEW = [100, 300, 900], [0, 1, 2], ["unique", "mixed", "camouflage"], 400, 512, 8, 32, 320
    MATH_BUDGETS, MATH_SEEDS, N_EVAL_MATH, MAX_NEW_MATH = [500, 2000], [0, 1], 500, 512
CACHE11 = f"{OUTDIR}/step11_wildchat_E384_{'quick' if QUICK else 'full'}.npz"
CACHE14 = f"{OUTDIR}/step14_meta_{'quick' if QUICK else 'full'}.npz"
CACHE16 = f"{OUTDIR}/step16_mm_{'quick' if QUICK else 'full'}.pkl"
N_CONV = 30000 if QUICK else None; DIMS = [384]; D_MMU = None
print(f"GPU {torch.cuda.get_device_name(0)} | QUICK={QUICK} RUN_MATH={RUN_MATH}")

# %% [markdown]
# ## Part 0 — caches (WildChat embeddings and attacker-pool text pairs)

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
# ===== Cell 4: MetaMathQA by type (GSM / MATH), one (query, response) per source problem; natural-likeness for camouflage =====
if os.path.exists(CACHE16):
    with open(CACHE16, "rb") as fh: GSM_PAIRS, MATH_PAIRS, NAT = pickle.load(fh)
else:
    t0 = time.time()
    from datasets import load_dataset
    mmq = load_dataset("meta-math/MetaMathQA", split="train").select_columns(["query", "response", "original_question", "type"])
    groups = {"GSM": {}, "MATH": {}}
    for q, a, o, ty in zip(mmq["query"], mmq["response"], mmq["original_question"], mmq["type"]):
        if q and a and o and ty: groups["GSM" if ty.startswith("GSM") else "MATH"].setdefault(o, []).append((q.strip()[:1000], a.strip()[:2000]))
    rr = np.random.default_rng(SEED)
    def one_per_problem(gr):
        origs = list(gr); rr.shuffle(origs); return [gr[o][int(rr.integers(len(gr[o])))] for o in origs]
    GSM_PAIRS, MATH_PAIRS = one_per_problem(groups["GSM"]), one_per_problem(groups["MATH"])
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        os.system("pip install -q sentence-transformers"); from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda"); enc.max_seq_length = 256
    Q = enc.encode([q for q, _ in GSM_PAIRS], batch_size=512, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    ben = np.flatnonzero(~is_pool[ACCT]); smp = np.random.default_rng(3).choice(ben, min(50000, len(ben)), replace=False)
    B_ = torch.as_tensor(E16[smp].astype(np.float32), device=DEV); NAT = np.empty(len(Q), np.float32)
    for s0 in range(0, len(Q), 2048):
        NAT[s0:s0 + 2048] = (torch.as_tensor(Q[s0:s0 + 2048], device=DEV) @ B_.T).topk(10, dim=1).values.mean(1).cpu().numpy()
    del enc, B_; gc.collect(); torch.cuda.empty_cache()
    with open(CACHE16, "wb") as fh: pickle.dump((GSM_PAIRS, MATH_PAIRS, NAT), fh)
    print(f"GSM-type pairs {len(GSM_PAIRS)}, MATH-type pairs {len(MATH_PAIRS)} ({time.time()-t0:.0f}s)")
CAMO = np.flatnonzero(NAT >= np.quantile(NAT, 1 - CAMO_FRAC)); NATURAL = [POOL_TXT[i] for i in sorted(POOL_TXT) if POOL_TXT[i][0] and POOL_TXT[i][1]]
print(f"camouflage pool {len(CAMO)} GSM-type prompts (natural-likeness >= {np.quantile(NAT, 1 - CAMO_FRAC):.3f}, median {np.median(NAT):.3f}); natural pairs {len(NATURAL)}")

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


# %% [markdown]
# ## Part G — GSM8K below saturation

# %%
# ===== Cell 6: Part G — GSM8K below saturation, three variants, three seeds =====
tokz = AutoTokenizer.from_pretrained(STUDENT); tokz.pad_token = tokz.pad_token or tokz.eos_token
gsm = load_dataset("openai/gsm8k", "main", split="test").select(range(N_EVAL))
GQ = list(gsm["question"]); GOLD = [float(a.split("####")[-1].strip().replace(",", "")) for a in gsm["answer"]]
def gsm_correct(model):
    preds = [last_number(t) for t in generate(model, tokz, GQ, MAX_NEW)]
    return np.array([p is not None and abs(p - g) < 1e-4 for p, g in zip(preds, GOLD)], float)
def variant_pairs(var, n, rng):
    if var == "unique": return [GSM_PAIRS[i] for i in rng.choice(len(GSM_PAIRS), n, replace=False)]
    if var == "camouflage": return [GSM_PAIRS[i] for i in rng.choice(CAMO, n, replace=False)]
    if var == "mixed": return [GSM_PAIRS[i] for i in rng.choice(len(GSM_PAIRS), (n + 1) // 2, replace=False)] + [NATURAL[i] for i in rng.choice(len(NATURAL), n // 2, replace=False)]
t0 = time.time(); base_model = AutoModelForCausalLM.from_pretrained(STUDENT, torch_dtype=torch.float16).to(DEV)
BASE_C = gsm_correct(base_model); del base_model; gc.collect(); torch.cuda.empty_cache()
print(f"base GSM8K accuracy {BASE_C.mean():.3f} on {N_EVAL} problems ({time.time()-t0:.0f}s)")
CORR = {}
for n in BUDGETS:
    for var in VARIANTS:
        for sd in SEEDS:
            pairs = variant_pairs(var, n, np.random.default_rng(SEED + 1000 * sd + n))
            model = train_lora(STUDENT, pairs, sd, tokz); CORR[(var, n, sd)] = gsm_correct(model); del model; gc.collect(); torch.cuda.empty_cache()
            print(f"  {var:>10} n={n:4d} seed={sd}: GSM8K {CORR[(var, n, sd)].mean():.3f} (gain {100*(CORR[(var, n, sd)].mean()-BASE_C.mean()):+.1f} pt)  ({time.time()-t0:.0f}s)")
def summary(var, n):
    gains = [100 * (CORR[(var, n, s)].mean() - BASE_C.mean()) for s in SEEDS]; return float(np.mean(gains)), float(np.std(gains, ddof=1)) if len(gains) > 1 else float("nan")
def paired_diff(v1, v2, n):
    # per-seed paired difference in accuracy on identical problems; SE across seeds and problems
    d = [100 * (CORR[(v1, n, s)] - CORR[(v2, n, s)]) for s in SEEDS]; per_seed = [x.mean() for x in d]
    se_items = float(np.mean([x.std(ddof=1) / math.sqrt(len(x)) for x in d])); return float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float("nan"), se_items
G = {}
for n in BUDGETS:
    for var in VARIANTS: G[(var, n)] = summary(var, n)
    md, sdd, se = paired_diff("mixed", "unique", n); cd, sdc, sec = paired_diff("camouflage", "unique", n)
    G[("mixed-unique", n)] = (md, sdd, se); G[("camouflage-unique", n)] = (cd, sdc, sec)
    # [PAPER] App. Table 13 (all math / half natural / camouflaged): +9.8 ± 2.1 / +9.8 ± 1.8 / +9.3 ± 0.7 (100); +7.5 ± 0.5 / +10.2 ± 2.1 / +8.7 ± 2.3 (300);
    # [PAPER]   +8.6 ± 1.2 / +10.1 ± 1.0 / +11.2 ± 1.9 (900); Fig. 6b; §6 "Two regimes", App. G.1: 9.8 points from 100 prompts, no more from 900
    print(f"n={n:4d}: gains (mean±sd over seeds) " + " ".join(f"{v}={G[(v, n)][0]:+.1f}±{G[(v, n)][1]:.1f}" for v in VARIANTS)
          + f" | paired mixed−unique {md:+.1f} (seed sd {sdd:.1f}, item SE {se:.1f}); camouflage−unique {cd:+.1f} (seed sd {sdc:.1f}, item SE {sec:.1f})")

# %% [markdown]
# ## Part H — MATH-500 with a 1.5B student (optional)

# %%
# ===== Cell 7: Part H (optional) — MATH-500 with a 1.5B student =====
H = {}
# [PAPER] Prediction log (Step 16): P16.4 Untestable (not run): RUN_MATH was False; the MATH-500 numbers come from Step 19 Part B
if RUN_MATH:
    tokm = AutoTokenizer.from_pretrained(STUDENT_MATH); tokm.pad_token = tokm.pad_token or tokm.eos_token
    m5 = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(N_EVAL_MATH)); MQ, MA = list(m5["problem"]), [norm_math(a) for a in m5["answer"]]
    def math_correct(model):
        preds = generate(model, tokm, MQ, MAX_NEW_MATH)
        return np.array([norm_math(extract_math(p)) == a for p, a in zip(preds, MA)], float)
    t0 = time.time(); bm = AutoModelForCausalLM.from_pretrained(STUDENT_MATH, torch_dtype=torch.float16).to(DEV); MBASE = math_correct(bm); del bm; torch.cuda.empty_cache()
    print(f"MATH-500 base accuracy {MBASE.mean():.3f}")
    for n in MATH_BUDGETS:
        for var in ["unique", "mixed"]:
            for sd in MATH_SEEDS:
                r = np.random.default_rng(SEED + 500 * sd + n)
                pairs = [MATH_PAIRS[i] for i in r.choice(len(MATH_PAIRS), n if var == "unique" else (n + 1) // 2, replace=False)]
                if var == "mixed": pairs += [NATURAL[i] for i in r.choice(len(NATURAL), n // 2, replace=False)]
                model = train_lora(STUDENT_MATH, pairs, sd, tokm); c = math_correct(model); del model; gc.collect(); torch.cuda.empty_cache()
                H[(var, n, sd)] = float(100 * (c.mean() - MBASE.mean()))
                print(f"  MATH {var:>7} n={n:5d} seed={sd}: accuracy {c.mean():.3f} (gain {H[(var, n, sd)]:+.1f} pt)  ({time.time()-t0:.0f}s)")
else:
    print("Part H skipped (set RUN_MATH = True to run MATH-500 with Qwen2.5-1.5B)")

# %% [markdown]
# ## Predictions, figure, VERDICT REPORT — copy into the chat

# %%
# ===== Cell 8: predictions, figure, verdict =====
b0, b1 = BUDGETS[0], BUDGETS[-1]
# [PAPER] Prediction log (Step 16): P16.1 Fail (+9.8 to +8.6)
print(f"P16.1 unique gain {G[('unique', b0)][0]:+.1f} -> {G[('unique', b1)][0]:+.1f} from {b0} to {b1} queries (pass if increase >= 3)")
for n in BUDGETS[1:]:
    # [PAPER] Prediction log (Step 16): P16.2 Fail (+2.7, +1.5)
    print(f"P16.2 n={n}: mixed−unique {G[('mixed-unique', n)][0]:+.1f} pt (pass if <= -2)")
r_ = G[("camouflage", b1)][0] / G[("unique", b1)][0] if G[("unique", b1)][0] > 0 else float("nan")
# [PAPER] Prediction log (Step 16): P16.3 Fail (130%)
print(f"P16.3 n={b1}: camouflage gain {G[('camouflage', b1)][0]:+.1f} vs unique {G[('unique', b1)][0]:+.1f} -> ratio {r_:.2f} (pass if <= 0.8)")
if H:
    for n in MATH_BUDGETS:
        u = np.mean([H[('unique', n, s)] for s in MATH_SEEDS]); m = np.mean([H[('mixed', n, s)] for s in MATH_SEEDS])
        print(f"P16.4 MATH n={n}: unique {u:+.1f}, mixed {m:+.1f} (ratio {m/u if u > 0 else float('nan'):.2f}; pass at the largest budget if unique >= 3 and ratio <= 0.7)")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.5, 4))
for var, c_ in [("unique", "tab:red"), ("mixed", "tab:purple"), ("camouflage", "tab:green")]:
    m_ = [G[(var, n)][0] for n in BUDGETS]; s_ = [G[(var, n)][1] if not np.isnan(G[(var, n)][1]) else 0 for n in BUDGETS]
    ax.errorbar(BUDGETS, m_, yerr=s_, fmt="o-", color=c_, capsize=3, label=var)
ax.set_xscale("log"); ax.set_xticks(BUDGETS); ax.set_xticklabels(BUDGETS); ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("campaign queries (equal budget per variant)"); ax.set_ylabel("GSM8K gain over base (points, mean ± sd over seeds)")
ax.set_title("Price of stealth below saturation"); ax.legend(); plt.tight_layout(); plt.savefig(f"{OUTDIR}/step16_figure.png", dpi=130); plt.show()
print("=" * 90); print("STEP 16 VERDICT REPORT"); print("=" * 90)
print(f"[G] base GSM8K {BASE_C.mean():.4f} on {N_EVAL} problems; seeds {SEEDS}; budgets {BUDGETS}")
for k, v in G.items(): print(f"[G] {k}: mean {v[0]:+.2f} sd {v[1]:.2f}" + (f" itemSE {v[2]:.2f}" if len(v) > 2 else ""))
for (var, n, sd), c in CORR.items(): print(f"[G-raw] {var} n={n} seed={sd}: {c.mean():.4f}")
for k, v in H.items(): print(f"[H] {k}: gain {v:+.2f}")
print("=" * 90)
with open(f"{OUTDIR}/step16_results.json", "w") as fh: json.dump(dict(G={str(k): v for k, v in G.items()}, H={str(k): v for k, v in H.items()}, base=float(BASE_C.mean())), fh, indent=1)
