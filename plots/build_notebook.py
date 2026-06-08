"""Builds plots/analyze.ipynb. Run once: python plots/build_notebook.py"""
import os, nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# Non-ergodic Mess3 — analysis

Loads what `experiments/nonergodic/train.py` saved to its `run/` directory and
plots. All probe fitting goes through `src.metrics.probes`; experiment-specific
analysis is in `src.nonergodic.analysis`; plotting is in `plots.nonergodic`.

Sections: training curve · belief decodability · telescope geometry (2D + 3D) ·
component-identity probe · effective readout (start→end) · direct unembedding
inspection · causal-test stub.
""")

code("""import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

# this notebook lives in plots/; repo root is one level up
REPO = os.path.abspath(os.path.join(os.getcwd(), ".."))
if REPO not in sys.path: sys.path.insert(0, REPO)

from src.nonergodic import analysis as A
from plots import nonergodic as P

RUN = os.path.join(REPO, "experiments", "nonergodic", "run")

cfg = json.load(open(os.path.join(RUN, "config.json")))
hmm = np.load(os.path.join(RUN, "hmm.npz"))
hist = np.load(os.path.join(RUN, "train_history.npz"))
val = np.load(os.path.join(RUN, "val_data.npz"))
gt = np.load(os.path.join(RUN, "ground_truth.npz"))
act = np.load(os.path.join(RUN, "activations.npz"))
wts = np.load(os.path.join(RUN, "weights.npz"))

K = len(cfg["comp_params"]); S = cfg["n_states"]; V = cfg["vocab"]; L = cfg["seq_len"]
d = cfg["d_model"]
M_block = hmm["M_block"]; comp_params = hmm["comp_params"]
print("components:", comp_params.tolist())
""")

md("## 1. Training curve")
code("""P.plot_training_curve(hist["train_loss"], hist["val_loss"], float(hist["opt_loss"]))
plt.tight_layout(); plt.show()""")

md("## 2. Telescoped-belief decodability  (via src.metrics.probes)")
code("""resid = act["resid_final"].reshape(-1, d)
B = gt["telescoped"].reshape(-1, K*S)
probe = A.decode_beliefs(resid, B, S, seed=0)
print("per-component test R²:", {c: round(r,3) for c,r in probe["per_block_r2"].items()})
P.plot_belief_r2(probe["per_block_r2"]); plt.tight_layout(); plt.show()""")

md("## 3. Telescope geometry")
code("""tel = gt["telescoped"].reshape(-1, K*S)
P.plot_telescope_2d(A.telescope_2d(tel, S), comp_params); plt.tight_layout(); plt.show()""")
code("""P.plot_telescope_3d(A.telescope_3d(tel, S), comp_params); plt.tight_layout(); plt.show()""")

md("## 4. Component-identity probe across positions  (via src.metrics.probes)")
code("""acc = A.component_probe_by_position(act["resid_final"], val["val_comps"], seed=0)
P.plot_component_probe(acc); plt.tight_layout(); plt.show()""")

md("""## 5. Effective readout (start→end)

Fit (shared probe code) decoded belief -> model probs, column-center to remove
the sum-to-1 gauge, compare to the true block emission. This is the start→end
argument: shows the implied map is emission-shaped, not that the unembedding
mechanism implements it.""")
code("""Pm = act["model_probs"].reshape(-1, V)
M_rec, M_rec_c = A.recover_readout(probe["decoded_all"], Pm)
print("cosine(recovered, true block emission), centered:",
      round(A.cosine(M_rec_c, A.col_center(M_block)), 3))
P.plot_readout_recovery(M_rec_c, A.col_center(M_block), S); plt.show()""")

md("""## 6. Direct unembedding inspection

`logits = resid_final @ W_U`. Pull `W_U` back into belief coordinates: how much
of it lives in the belief subspace, and does its belief-relevant action match
the (log) emission matrix.""")
code("""W_U = wts["W_U"].T
insp = A.inspect_unembedding(W_U, probe["W"], M_block)
print(f"captured energy in belief subspace: {insp['captured_energy']:.3f} "
      f"(random ~ {insp['random_baseline']:.3f}, clean = 1.0)")
print(f"reconstruction rel error          : {insp['recon_rel_err']:.3f}")
print(f"cosine(pullback, log-emission)    : {insp['cos_logM']:.3f}")
print(f"cosine(pullback, emission)        : {insp['cos_M']:.3f}")
P.plot_unembedding_inspection(insp["pullback_centered"], insp["logM_centered"], S); plt.show()""")

md("""## 7. (stub) Causal test — next step

Clean test of "the unembedding implements the emission matrix" is causal: patch
one component's belief subspace and check the output shifts by exactly
`delta_eta_n @ M_n` (weighted by `w_n`). Sidesteps the LayerNorm nonlinearity and
the log-scale issue because you measure output deltas directly. Use the saved
`weights.npz` (W_U, LN params) and `activations.npz`.
""")

nb["cells"] = cells
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze.ipynb")
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
