"""Plotting for the non-ergodic Mess3 experiment.

Every function takes already-computed arrays (from src.nonergodic.analysis)
and a matplotlib axis (or makes its own figure). No analysis logic here.
"""
import numpy as np
import matplotlib.pyplot as plt

# palette consistent with the rest of the project
COLORS = ["#534AB7", "#D85A30", "#1D9E75", "#BA7517"]


def plot_training_curve(train_loss, val_loss, opt_loss, ax=None):
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    ax.plot(train_loss, label="train")
    ax.plot(val_loss, label="val")
    ax.axhline(opt_loss, ls="--", c="gray", label=f"Bayes-opt {opt_loss:.4f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training vs Bayes-optimal"); ax.legend()
    return ax


def plot_belief_r2(per_block_r2, ax=None):
    """per_block_r2: {component_index: R²} from analysis.decode_beliefs."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    comps = sorted(per_block_r2)
    vals = [per_block_r2[c] for c in comps]
    ax.bar([f"comp {c}" for c in comps], vals, color=[COLORS[c] for c in comps])
    ax.set_ylabel("test R²"); ax.set_ylim(0, 1)
    ax.set_title(f"Belief decodability per component (mean {np.mean(vals):.3f})")
    return ax


def plot_telescope_2d(telescope_pts, comp_params, ax=None):
    """telescope_pts: list of (points (M,2), component_index) from analysis.telescope_2d."""
    ax = ax or plt.subplots(figsize=(6, 6))[1]
    for pts, n in telescope_pts:
        ax.scatter(pts[:, 0], pts[:, 1], s=2, c=COLORS[n], alpha=0.3,
                   label=f"comp {n} (α={comp_params[n][0]})")
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(markerscale=4); ax.set_title("Telescope geometry (2-simplex overlay)")
    return ax


def plot_telescope_3d(telescope_blocks, comp_params, fig=None):
    """telescope_blocks: list of (points (M,3), n) from analysis.telescope_3d."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    K = len(telescope_blocks)
    fig = fig or plt.figure(figsize=(5.5 * K, 5))
    for pts, n in telescope_blocks:
        ax = fig.add_subplot(1, K, n + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=COLORS[n], alpha=0.3)
        ax.set_title(f"comp {n} (α={comp_params[n][0]}, x={comp_params[n][1]})")
        ax.set_xlabel("s0"); ax.set_ylabel("s1"); ax.set_zlabel("s2")
    return fig


def plot_component_probe(acc_by_pos, ax=None):
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    ax.plot(range(len(acc_by_pos)), acc_by_pos, marker="o")
    ax.axhline(0.5, ls="--", c="gray", label="chance")
    ax.set_xlabel("context position"); ax.set_ylabel("component-id accuracy")
    ax.set_ylim(0.45, 1.0); ax.legend()
    ax.set_title("Component identification vs position")
    return ax


def _heatmap(ax, M, title, n_states, vmax=None):
    vmax = vmax or np.abs(M).max()
    ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title); ax.set_xlabel("token"); ax.set_ylabel("belief coord")
    K = M.shape[0] // n_states
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels([f"c{n}s{s}" for n in range(K) for s in range(n_states)])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8)


def plot_readout_recovery(M_rec_centered, M_block_centered, n_states, fig=None):
    fig = fig or plt.figure(figsize=(9, 4))
    a1, a2 = fig.subplots(1, 2)
    _heatmap(a1, M_rec_centered, "recovered (centered)", n_states, vmax=0.35)
    _heatmap(a2, M_block_centered, "true [M₁;M₂] (centered)", n_states, vmax=0.35)
    fig.tight_layout()
    return fig


def plot_unembedding_inspection(pullback_centered, logM_centered, n_states, fig=None):
    fig = fig or plt.figure(figsize=(9, 4))
    a1, a2 = fig.subplots(1, 2)
    _heatmap(a1, pullback_centered, "W_U in belief coords", n_states)
    _heatmap(a2, logM_centered, "log(M_block), centered", n_states)
    fig.tight_layout()
    return fig
