"""Train the tiny transformer on the non-ergodic Mess3 mixture and save
everything needed for analysis into ./run/ (next to this script).

Run:  python experiments/nonergodic/train.py
Then open experiments/nonergodic/analyze.ipynb.

Saved files (./run/):
  config.json        all hyperparameters
  hmm.npz            per-component transition stacks, stationary dists,
                     emission matrices, block-diagonal emission
  model.pt           full model state_dict
  train_history.npz  train/val loss per epoch + Bayes-optimal val loss
  val_data.npz       val sequences + true component labels (+ train slice)
  ground_truth.npz   telescoped beliefs, optimal NTP, posterior weights,
                     normalized per-component beliefs (per position)
  activations.npz    embedding, post-attn, post-mlp, final-resid, logits,
                     model probs, attention weights (val set, all positions)
  weights.npz        unembedding, final-LN gamma/beta, token/pos embeddings
"""
import os, sys, json, math
import numpy as np
import torch
import torch.nn.functional as F

# make the repo root importable so `src` and `plots` resolve
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from configs.nonergodic_config import NonergodicConfig as Config
from src.nonergodic import TinyTransformer, build_hmms, make_dataset, all_telescoped

OUT = os.path.join(_HERE, "run")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
cfg = Config()
torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

json.dump(cfg.to_dict(), open(os.path.join(OUT, "config.json"), "w"), indent=2)

# HMMs
T_mats, T_stack, pis, M_comp, M_block = build_hmms(cfg)
np.savez(os.path.join(OUT, "hmm.npz"),
         T_stack=np.array(T_stack), pis=np.array(pis),
         M_comp=np.array(M_comp), M_block=M_block,
         comp_params=np.array(cfg.comp_params), pi_prior=np.array(cfg.pi_prior))

# data
train_seqs, train_comps = make_dataset(cfg, T_mats, pis, cfg.n_train, 0)
val_seqs, val_comps = make_dataset(cfg, T_mats, pis, cfg.n_val, 10_000_000)
np.savez(os.path.join(OUT, "val_data.npz"), val_seqs=val_seqs, val_comps=val_comps,
         train_seqs=train_seqs[:2000], train_comps=train_comps[:2000])

# ground truth
print("Computing ground-truth telescoped beliefs...")
tel, ntp, wts, loc = all_telescoped(cfg, T_stack, pis, M_comp, val_seqs)
np.savez(os.path.join(OUT, "ground_truth.npz"),
         telescoped=tel, optimal_ntp=ntp, posterior_weights=wts, local_beliefs=loc)

opt_loss, cnt = 0.0, 0
for i in range(len(val_seqs)):
    for t in range(cfg.seq_len - 1):
        opt_loss += -math.log(max(ntp[i, t, val_seqs[i, t+1]], 1e-12)); cnt += 1
opt_loss /= cnt
print(f"Bayes-optimal val loss: {opt_loss:.4f}")

# model + train
model = TinyTransformer(cfg).to(DEV)
print(f"Model params: {sum(p.numel() for p in model.parameters())}, device={DEV}")
opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

def batches(seqs, bs, shuffle=True):
    idx = np.arange(len(seqs))
    if shuffle: np.random.shuffle(idx)
    for i in range(0, len(seqs), bs):
        yield torch.from_numpy(seqs[idx[i:i+bs]]).to(DEV)

tr_hist, va_hist = [], []
for ep in range(cfg.epochs):
    model.train(); tot = nb = 0
    for xb in batches(train_seqs, cfg.batch_size):
        lg = model(xb[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, cfg.vocab), xb[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    sched.step()
    model.eval(); vt = vn = 0
    with torch.no_grad():
        for xb in batches(val_seqs, cfg.batch_size, shuffle=False):
            lg = model(xb[:, :-1])
            vt += F.cross_entropy(lg.reshape(-1, cfg.vocab), xb[:, 1:].reshape(-1)).item(); vn += 1
    tr_hist.append(tot/nb); va_hist.append(vt/vn)
    print(f"epoch {ep+1:2d}  train {tr_hist[-1]:.4f}  val {va_hist[-1]:.4f}  (opt {opt_loss:.4f})")

torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))
np.savez(os.path.join(OUT, "train_history.npz"),
         train_loss=np.array(tr_hist), val_loss=np.array(va_hist), opt_loss=opt_loss)

# exhaustive activations
print("Extracting activations...")
model.eval()
keys = ["embedding", "resid_post_attn", "resid_post_mlp", "resid_final", "logits"]
acc = {k: [] for k in keys}; probs_all, attn_all = [], []
with torch.no_grad():
    for xb in batches(val_seqs, cfg.batch_size, shuffle=False):
        out = model(xb, return_all=True)
        for k in keys: acc[k].append(out[k].cpu().numpy())
        probs_all.append(F.softmax(out["logits"], dim=-1).cpu().numpy())
        attn_all.append(out["attn_weights"].cpu().numpy())
acts = {k: np.concatenate(acc[k], 0) for k in keys}
np.savez(os.path.join(OUT, "activations.npz"),
         model_probs=np.concatenate(probs_all, 0),
         attn_weights=np.concatenate(attn_all, 0), **acts)

# raw weights
sd = model.state_dict()
np.savez(os.path.join(OUT, "weights.npz"),
         W_U=sd["un.weight"].cpu().numpy(),
         lf_gamma=sd["lf.weight"].cpu().numpy(),
         lf_beta=sd["lf.bias"].cpu().numpy(),
         tok_emb=sd["te.weight"].cpu().numpy(),
         pos_emb=sd["pe.weight"].cpu().numpy())

print(f"\nSaved everything to: {OUT}")
print("Files:", ", ".join(sorted(os.listdir(OUT))))
