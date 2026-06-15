"""
Cone (normalized predictive vector) vs Prism (unnormalized) for the doc's
coin + even-process nonergodic composition (Introduction to Nonergodic Processes.md).

Coords are [coin, even-A, even-B] (R^3). We enumerate every binary sequence up to
length Lmax, run the block-diagonal belief filter, and collect:
  - PRISM point  = unnormalized vector v = eta0 @ T[x1] @ ... @ T[xL]      (length = Z = P(seq))
  - CONE  point  = v / Z                                                   (length = 1, on simplex)

The cone is exactly the radial projection (through the origin) of the prism onto the
unit simplex {x+y+z=1}.
"""
import itertools
import numpy as np

# --- Process definition (doc lines 176, 180) ---------------------------------
T0 = np.array([[2/3, 0,   0  ],
               [0,   0,   1/2],
               [0,   1,   0  ]])
T1 = np.array([[1/3, 0,   0  ],
               [0,   1/2, 0  ],
               [0,   0,   0  ]])
T = {0: T0, 1: T1}
eta0 = np.array([1/2, 1/3, 1/6])          # = (1/2*1, 1/2*2/3, 1/2*1/3)

# sanity: after "1" the unnormalized vector should be (1/6, 1/6, 0)  (doc line 188)
assert np.allclose(eta0 @ T1, [1/6, 1/6, 0])

Lmax = 11
rows = []                                  # (prism_xyz, cone_xyz, length)
for L in range(0, Lmax + 1):
    for s in itertools.product((0, 1), repeat=L):
        v = eta0.copy()
        for x in s:
            v = v @ T[x]
        Z = v.sum()
        if Z < 1e-12:                      # forbidden word -> zero vector, skip
            continue
        rows.append((v.copy(), v / Z, L))

# dedup on the prism point (rounded) to avoid massive overplotting
seen = {}
for v, c, L in rows:
    key = tuple(np.round(v, 7))
    if key not in seen:
        seen[key] = (v, c, L)
prism = np.array([r[0] for r in seen.values()])
cone  = np.array([r[1] for r in seen.values()])
lengths = np.array([r[2] for r in seen.values()])
print(f"{len(prism)} unique prism points; {len(np.unique(np.round(cone,6),axis=0))} unique cone points")

# ---------------------------------------------------------------------------
# Matplotlib static (for verification) + Plotly interactive HTML (for rotating)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

V = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)   # simplex vertices
rng = np.random.default_rng(0)
samp = rng.choice(len(prism), size=min(60, len(prism)), replace=False)

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")
ax.add_collection3d(Poly3DCollection([V], alpha=0.12, facecolor="tab:gray", edgecolor="gray"))
# rays origin -> prism -> cone (show the radial projection)
for i in samp:
    ax.plot([0, cone[i, 0]], [0, cone[i, 1]], [0, cone[i, 2]],
            color="lightgray", lw=0.5, zorder=1)
sc = ax.scatter(prism[:, 0], prism[:, 1], prism[:, 2], c=lengths, cmap="viridis",
                s=10, depthshade=False, label="PRISM (unnormalized)  length = Z = P(seq)")
ax.scatter(cone[:, 0], cone[:, 1], cone[:, 2], color="crimson", s=14,
           depthshade=False, label="CONE (normalized)  on simplex x+y+z=1")
ax.scatter([0], [0], [0], color="black", s=30)
ax.text(0, 0, 0, "  origin", fontsize=8)
for v, lab in zip(V, ["coin", "even-A", "even-B"]):
    ax.text(*v, "  " + lab, fontsize=9)
ax.set_xlabel("coin"); ax.set_ylabel("even-A"); ax.set_zlabel("even-B")
ax.set_title("Nonergodic belief geometry: coin + even process\n"
             "cone = prism radially projected onto the unit simplex (÷Z)")
ax.legend(loc="upper left", fontsize=8)
ax.view_init(elev=22, azim=35)
fig.colorbar(sc, ax=ax, shrink=0.5, label="sequence length ℓ")
fig.tight_layout()
fig.savefig("plots/cone_vs_prism.png", dpi=140)
print("wrote plots/cone_vs_prism.png")

try:
    import plotly.graph_objects as go
    fig2 = go.Figure()
    fig2.add_trace(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=[0], j=[1], k=[2],
                             opacity=0.12, color="gray", name="unit simplex", showscale=False))
    for i in samp:
        fig2.add_trace(go.Scatter3d(x=[0, cone[i, 0]], y=[0, cone[i, 1]], z=[0, cone[i, 2]],
                                    mode="lines", line=dict(color="lightgray", width=1),
                                    showlegend=False, hoverinfo="skip"))
    fig2.add_trace(go.Scatter3d(x=prism[:, 0], y=prism[:, 1], z=prism[:, 2], mode="markers",
                                marker=dict(size=3, color=lengths, colorscale="Viridis",
                                            colorbar=dict(title="ℓ")),
                                name="PRISM (unnormalized, length=Z)"))
    fig2.add_trace(go.Scatter3d(x=cone[:, 0], y=cone[:, 1], z=cone[:, 2], mode="markers",
                                marker=dict(size=4, color="crimson"),
                                name="CONE (normalized, on simplex)"))
    fig2.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text",
                                marker=dict(size=4, color="black"), text=["origin"],
                                textposition="top center", showlegend=False))
    fig2.update_layout(title="cone = prism ÷ Z  (radial projection onto unit simplex)",
                       scene=dict(xaxis_title="coin", yaxis_title="even-A", zaxis_title="even-B"),
                       legend=dict(x=0, y=1))
    fig2.write_html("plots/cone_vs_prism.html")
    print("wrote plots/cone_vs_prism.html")
except Exception as e:
    print("plotly skipped:", e)
