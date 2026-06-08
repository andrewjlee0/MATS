"""Analysis for the non-ergodic Mess3 experiment.

All linear-probe fitting is delegated to the existing ``src.metrics.probes``
module (OLS-with-bias). This file only adds experiment-specific logic on top:
component-block decoding, readout recovery, the direct unembedding inspection,
and telescope geometry. No reimplemented OLS here.
"""
import numpy as np
import torch

from src.metrics import probes


# ────────────────────────── small numeric helpers ──────────────────────────
def cosine(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def col_center(M):
    return M - M.mean(0, keepdims=True)


def _t(x):
    return torch.from_numpy(np.asarray(x, dtype=np.float64)).double()


# ────────────────────── belief probe (uses src.metrics.probes) ──────────────────────
def decode_beliefs(resid_flat, belief_flat, n_states, test_frac=0.2, seed=0):
    """Decode the telescoped belief from the residual using the shared probe code.

    Uses probes.fit_and_evaluate_multi_stacked for per-component-block R²
    (telescoped target is [block_0 | block_1 | ...]), and probes.fit_probe /
    probes.predict_probe for the probe weights and decoded belief.

    Returns dict: per_block_r2 {comp: R²}, W (d+1, K*S) numpy, decoded_all (N,K*S)
    numpy, train_idx, test_idx.
    """
    rng = np.random.default_rng(seed)
    n = len(resid_flat)
    perm = rng.permutation(n); cut = int(n * test_frac)
    te, tr = perm[:cut], perm[cut:]
    K = belief_flat.shape[1] // n_states

    Xtr, Xte = _t(resid_flat[tr]), _t(resid_flat[te])
    Ytr, Yte = _t(belief_flat[tr]), _t(belief_flat[te])

    per_block = probes.fit_and_evaluate_multi_stacked(
        Xtr, Xte, Ytr, Yte, n_states, list(range(K)), use_bias=True)
    W = probes.fit_probe(Xtr, Ytr, use_bias=True)
    decoded_all = probes.predict_probe(_t(resid_flat), W, use_bias=True).numpy()
    return {"per_block_r2": per_block, "W": W.numpy(),
            "decoded_all": decoded_all, "train_idx": tr, "test_idx": te}


# ────────────── component-identity probe (uses src.metrics.probes) ──────────────
def component_probe_by_position(resid_NL, comps, seed=0):
    """Linear discriminant (via shared probe code) for component identity,
    evaluated per context position. Assumes 2 components (labels 0/1)."""
    N, L, d = resid_NL.shape
    X = resid_NL.reshape(-1, d)
    y = np.repeat(comps, L)
    yy = np.where(y == 0, -1.0, 1.0)[:, None]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X)); cut = len(X) // 5
    tr = perm[cut:]
    W = probes.fit_probe(_t(X[tr]), _t(yy[tr]), use_bias=True)
    pred = (probes.predict_probe(_t(X), W, use_bias=True).numpy().ravel() > 0).astype(int).reshape(N, L)
    return np.array([(pred[:, t] == comps).mean() for t in range(L)])


# ────────────── effective readout: model probs ~ decoded belief ──────────────
def recover_readout(decoded_belief, model_probs):
    """Fit (via shared probe code) decoded belief -> model probabilities.
    Returns recovered readout (K*S, vocab) and its column-centered form."""
    W = probes.fit_probe(_t(decoded_belief), _t(model_probs), use_bias=True).numpy()
    M_rec = W[:-1]
    return M_rec, col_center(M_rec)


# ────────────── direct unembedding inspection (pure numpy) ──────────────
def inspect_unembedding(W_U, probe_W, M_block):
    """Does the unembedding carry the block-diagonal emission?

    W_U (d, vocab): logits = resid_final @ W_U
    probe_W (d+1, K*S): belief probe (bias row last)
    M_block (K*S, vocab): true block emission
    """
    Pmat = probe_W[:-1]
    Q, _ = np.linalg.qr(Pmat)
    WU_proj = Q @ (Q.T @ W_U)
    captured = np.linalg.norm(WU_proj) ** 2 / np.linalg.norm(W_U) ** 2
    X, *_ = np.linalg.lstsq(Pmat, W_U, rcond=None)
    recon_rel_err = np.linalg.norm(Pmat @ X - W_U) / np.linalg.norm(W_U)
    Xc = col_center(X)
    logM = col_center(np.log(np.clip(M_block, 1e-6, None)))
    Mc = col_center(M_block)
    return {
        "captured_energy": float(captured),
        "random_baseline": Pmat.shape[1] / Pmat.shape[0],
        "pullback": X,
        "pullback_centered": Xc,
        "recon_rel_err": float(recon_rel_err),
        "cos_logM": cosine(Xc, logM),
        "cos_M": cosine(Xc, Mc),
        "logM_centered": logM,
    }


# ────────────────────────── telescope geometry ──────────────────────────
SIMPLEX_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8660254]])
SIMPLEX_CENTER = SIMPLEX_VERTS.mean(0)


def telescope_2d(telescoped, n_states):
    """2D simplex coords scaled toward center by weight. list of (pts (M,2), n)."""
    K = telescoped.shape[1] // n_states
    out = []
    for n in range(K):
        blk = telescoped[:, n*n_states:(n+1)*n_states]
        w = blk.sum(1, keepdims=True)
        eta = np.divide(blk, w, out=np.zeros_like(blk), where=w > 1e-9)
        xy = eta @ SIMPLEX_VERTS
        pts = SIMPLEX_CENTER + (xy - SIMPLEX_CENTER) * w
        out.append((pts, n))
    return out


def telescope_3d(telescoped, n_states):
    """Raw 3D telescoped block per component: list of (pts (M,3), n)."""
    K = telescoped.shape[1] // n_states
    return [(telescoped[:, n*n_states:(n+1)*n_states], n) for n in range(K)]
