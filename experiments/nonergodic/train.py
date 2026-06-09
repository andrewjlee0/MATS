"""Train the tiny transformer on the non-ergodic Mess3 mixture, GPU-maxed,
saving everything for analysis into ./run/.

GPU utilization tactics (a 1-layer/128-dim model is small, so the lever is
throughput, not occupancy):
  - whole dataset lives on the GPU as one int tensor (no per-batch H2D copies)
  - very large batch size (set in config; falls back gracefully)
  - TF32 matmuls + cudnn autotune
  - bf16/fp16 autocast (mixed precision)
  - optional torch.compile (set COMPILE=1)
  - activation extraction batched on-GPU

Run:  python experiments/nonergodic/train.py
"""
import os, sys, json, math
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from configs.nonergodic_config import NonergodicConfig as Config
from src.nonergodic import (TinyTransformer, build_hmms, make_dataset, all_telescoped,
                            sample_sequences_gpu, all_telescoped_gpu)

OUT = os.path.join(_HERE, "run")
os.makedirs(OUT, exist_ok=True)
cfg = Config()
torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
USE_CUDA = DEV == "cuda"
if USE_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True      # TF32 matmuls
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True             # autotune kernels
    # prefer bf16 on Ampere+, else fp16
    AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)} | amp={AMP_DTYPE}")
else:
    AMP_DTYPE = None
    print("WARNING: CUDA not available — running on CPU. "
          "Install a CUDA torch build to use the GPU.")

# large batch is the main throughput lever for a tiny model; configurable
BATCH = getattr(cfg, "batch_size", 256)
COMPILE = os.environ.get("COMPILE", "0") == "1"

json.dump(cfg.to_dict(), open(os.path.join(OUT, "config.json"), "w"), indent=2)

# ── HMMs ──
T_mats, T_stack, pis, M_comp, M_block = build_hmms(cfg)
np.savez(os.path.join(OUT, "hmm.npz"),
         T_stack=np.array(T_stack), pis=np.array(pis),
         M_comp=np.array(M_comp), M_block=M_block,
         comp_params=np.array(cfg.comp_params), pi_prior=np.array(cfg.pi_prior))

# ── data (sampled on the GPU when available, else CPU numpy) ──
if USE_CUDA:
    g = torch.Generator(device=DEV).manual_seed(cfg.seed)
    train_comps_t = torch.randint(0, len(cfg.comp_params), (cfg.n_train,), generator=g, device=DEV)
    val_comps_t   = torch.randint(0, len(cfg.comp_params), (cfg.n_val,), generator=g, device=DEV)
    train_t = sample_sequences_gpu(cfg, T_stack, pis, train_comps_t, DEV, seed=cfg.seed)
    val_t   = sample_sequences_gpu(cfg, T_stack, pis, val_comps_t, DEV, seed=cfg.seed + 1)
    train_seqs = train_t.cpu().numpy(); train_comps = train_comps_t.cpu().numpy()
    val_seqs = val_t.cpu().numpy(); val_comps = val_comps_t.cpu().numpy()
else:
    train_seqs, train_comps = make_dataset(cfg, T_mats, pis, cfg.n_train, 0)
    val_seqs, val_comps = make_dataset(cfg, T_mats, pis, cfg.n_val, 10_000_000)
    train_t = torch.from_numpy(train_seqs).to(DEV)
    val_t = torch.from_numpy(val_seqs).to(DEV)

np.savez(os.path.join(OUT, "val_data.npz"), val_seqs=val_seqs, val_comps=val_comps,
         train_seqs=train_seqs[:2000], train_comps=train_comps[:2000])

# ── ground truth (GPU Bayesian filter when available) ──
print("Computing ground-truth telescoped beliefs...")
if USE_CUDA:
    tel, ntp, wts, loc = all_telescoped_gpu(cfg, T_stack, pis, M_comp, val_seqs, DEV)
else:
    tel, ntp, wts, loc = all_telescoped(cfg, T_stack, pis, M_comp, val_seqs)
np.savez(os.path.join(OUT, "ground_truth.npz"),
         telescoped=tel, optimal_ntp=ntp, posterior_weights=wts, local_beliefs=loc)

opt_loss, cnt = 0.0, 0
for i in range(len(val_seqs)):
    for t in range(cfg.seq_len - 1):
        opt_loss += -math.log(max(ntp[i, t, val_seqs[i, t+1]], 1e-12)); cnt += 1
opt_loss /= cnt
print(f"Bayes-optimal val loss: {opt_loss:.4f}")

# ── model ──
model = TinyTransformer(cfg).to(DEV)
if USE_CUDA:
    model = model.to(memory_format=torch.channels_last) if False else model
print(f"Model params: {sum(p.numel() for p in model.parameters())}, device={DEV}, batch={BATCH}")
if COMPILE:
    model = torch.compile(model)
    print("torch.compile enabled")

opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, fused=USE_CUDA)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
scaler = torch.amp.GradScaler("cuda", enabled=(USE_CUDA and AMP_DTYPE == torch.float16))

def amp_ctx():
    if USE_CUDA:
        return torch.autocast("cuda", dtype=AMP_DTYPE)
    import contextlib; return contextlib.nullcontext()

def gpu_batches(data, bs, shuffle=True):
    n = data.shape[0]
    idx = torch.randperm(n, device=data.device) if shuffle else torch.arange(n, device=data.device)
    for i in range(0, n, bs):
        yield data[idx[i:i+bs]]

tr_hist, va_hist = [], []
for ep in range(cfg.epochs):
    model.train(); tot = nb = 0
    for xb in gpu_batches(train_t, BATCH):
        with amp_ctx():
            lg = model(xb[:, :-1])
            loss = F.cross_entropy(lg.reshape(-1, cfg.vocab), xb[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        tot += loss.item(); nb += 1
    sched.step()
    model.eval(); vt = vn = 0
    with torch.no_grad():
        for xb in gpu_batches(val_t, BATCH, shuffle=False):
            with amp_ctx():
                lg = model(xb[:, :-1])
                vt += F.cross_entropy(lg.reshape(-1, cfg.vocab), xb[:, 1:].reshape(-1)).item()
            vn += 1
    tr_hist.append(tot/nb); va_hist.append(vt/vn)
    if (ep+1) % max(1, cfg.epochs//50) == 0 or ep == 0:
        print(f"epoch {ep+1:3d}  train {tr_hist[-1]:.4f}  val {va_hist[-1]:.4f}  (opt {opt_loss:.4f})")

# unwrap compiled module for saving a clean state_dict
to_save = getattr(model, "_orig_mod", model)
torch.save(to_save.state_dict(), os.path.join(OUT, "model.pt"))
np.savez(os.path.join(OUT, "train_history.npz"),
         train_loss=np.array(tr_hist), val_loss=np.array(va_hist), opt_loss=opt_loss)

# ── activations (batched on GPU; full precision for clean analysis) ──
print("Extracting activations...")
to_save.eval()
keys = ["embedding", "resid_post_attn", "resid_post_mlp", "resid_final", "logits"]
acc = {k: [] for k in keys}; probs_all, attn_all = [], []
with torch.no_grad():
    for xb in gpu_batches(val_t, BATCH, shuffle=False):
        out = to_save(xb, return_all=True)
        for k in keys: acc[k].append(out[k].float().cpu().numpy())
        probs_all.append(F.softmax(out["logits"].float(), dim=-1).cpu().numpy())
        attn_all.append(out["attn_weights"].float().cpu().numpy())
acts = {k: np.concatenate(acc[k], 0) for k in keys}
np.savez(os.path.join(OUT, "activations.npz"),
         model_probs=np.concatenate(probs_all, 0),
         attn_weights=np.concatenate(attn_all, 0), **acts)

sd = to_save.state_dict()
np.savez(os.path.join(OUT, "weights.npz"),
         W_U=sd["un.weight"].cpu().numpy(),
         lf_gamma=sd["lf.weight"].cpu().numpy(),
         lf_beta=sd["lf.bias"].cpu().numpy(),
         tok_emb=sd["te.weight"].cpu().numpy(),
         pos_emb=sd["pe.weight"].cpu().numpy())

print(f"\nSaved everything to: {OUT}")
print("Files:", ", ".join(sorted(os.listdir(OUT))))