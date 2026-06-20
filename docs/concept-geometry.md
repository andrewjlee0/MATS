# The Geometry of Concepts in Activation Space — a working vocabulary

**Status:** living document. Started 2026-06-20, out of a MATS dinner conversation. Corrections,
counterexamples, and new rows in the example table are all welcome — edit freely.

**Audience:** interpretability researchers, especially the belief-state-geometry crowd. Assumes
light familiarity with HMMs / mixed-state presentations but tries to define terms as it goes.

---

## 0. Why this document exists

There is no consensus on what we mean when we say a concept is "a manifold," "a feature," "a
direction," or "a topological structure." The terms are used loosely and interchangeably, and the
question of whether some representation "is or isn't a manifold" often turns on *which object* the
*same word* is pointing at. This document tries to pin the vocabulary down precisely enough that
the well-known examples (belief-state fractals, days-of-the-week circles, modular arithmetic,
binary concepts) can all be described without contradiction.

The central claim: **"is concept X a manifold?" is not a well-posed question until you say
*which set*, *in which limit*, and *under which criterion* you mean.** Once you fix those three,
the question resolves.

---

## 1. Definitions, stated precisely

The single most common error is treating "manifold" as a synonym for "smooth blob" or "closed
shape." It is neither. There is **one organizing question** behind the entire classification:
*what does a small neighborhood of a point look like?* That local model — and how it is allowed to
vary from point to point — defines every class below.

| Class | Local model: a neighborhood of a point is homeomorphic to… |
|---|---|
| discrete set / **0-manifold** | a point ($\mathbb{R}^0$) |
| **manifold**, dim $n$ | all of $\mathbb{R}^n$ (same model at every point) |
| **manifold with boundary** | $\mathbb{R}^n$ **or** a half-space $\mathbb{H}^n=\{x_n\ge 0\}$ |
| **manifold with corners** | $\mathbb{R}^n$, half-space, **or** an orthant $[0,\infty)^k\times\mathbb{R}^{n-k}$ |
| **stratified space** | (a stratum) $\times$ (cone on a lower-dimensional link): the model *changes* between points but is constant *along* each stratum |
| **fractal / IFS attractor** | self-similar / branching at every scale — **no Euclidean model at any point** |

These classes nest: $0$-manifold $\subset$ manifold $\subset$ manifold-with-boundary $\subset$
manifold-with-corners $\subset$ stratified space, each strictly larger. Fractals sit *outside* the
whole tower. The detailed entries:

- **Topological manifold (dimension $n$):** a (Hausdorff, second-countable) space in which *every
  point has a neighborhood homeomorphic to $\mathbb{R}^n$*. The defining property is **locally
  Euclidean** ("locally linear," informally) — nothing about being closed, bounded, or finite.
  - $\mathbb{R}^n$, an infinite line, and the plane are all manifolds. **Manifolds need not be
    compact or bounded.**
  - A **circle** and a **sphere** are manifolds that happen to be compact.
  - The dimension $n$ is **well-defined** (invariance of domain, Brouwer): no space is both an
    $m$- and an $n$-manifold for $m\neq n$. This is what makes "the dimension of a representation"
    a meaningful quantity at all.

- **"Closed" is a trap word.** It means two unrelated things: (a) a topologically closed *set*,
  and (b) a **closed manifold** = *compact and without boundary* (circle, sphere, torus). A line
  is a manifold but not a *closed manifold*. Avoid the bare word "closed" in geometry discussions.

- **Manifold with boundary:** boundary points get charts onto a *half-space* $\{x_n \ge 0\}$
  instead of all of $\mathbb{R}^n$. A filled disk is the canonical example. The boundary
  $\partial M$ is itself an $(n-1)$-manifold *without* boundary (the boundary of a disk is a
  circle).

- **Manifold with corners:** corner points get charts onto an *orthant* $[0,\infty)^k \times
  \mathbb{R}^{n-k}$. A filled triangle (the 2-simplex $\Delta^2$) is the canonical example —
  interior points see $\mathbb{R}^2$, edges see a half-space ($k=1$), vertices see a quadrant
  ($k=2$). Probability simplices $\Delta^{n-1}$ are the case that matters here.

- **Smooth manifold:** a topological manifold with a compatible differentiable atlas, so tangent
  spaces (local *linearizations*) exist. "Locally linear" in the ML sense (LLE, tangent-plane
  approximation) is really *smooth*-manifold language.

- **Stratified space:** a space partitioned into manifold pieces ("strata") of *possibly different
  dimensions*, subject to two conditions that keep it well-behaved: the **frontier condition** (the
  closure of any stratum is a union of strata) and **local triviality along strata** (the local
  model is constant as you move within a stratum, and degenerates only when you cross to a
  lower-dimensional one). A polytope is the clean case: open top-cell ∪ open faces ∪ edges ∪
  vertices. An ordinary manifold is the degenerate case of a single stratum. This is the right
  framework for *mixed-dimension, piecewise-manifold* objects.

- **Fractal / IFS attractor:** by **Hutchinson's theorem**, any finite set of contractions
  $f_1,\dots,f_k$ on a complete metric space has a *unique* nonempty compact **attractor**
  $A=\bigcup_i f_i(A)$. Under the open-set condition its Hausdorff dimension is the *similarity
  dimension* $s$ solving $\sum_i r_i^{\,s}=1$ (Sierpinski triangle: $k=3$ maps, ratio $r=\tfrac12$
  $\Rightarrow s=\log 3/\log 2 \approx 1.585$).
  - **Why it is not a manifold:** it is **not locally Euclidean at any point** — every
    neighborhood keeps branching at all scales, so no chart onto $\mathbb{R}^n$ exists. The clean
    *certificate* is dimensional: a manifold has Hausdorff dimension equal to its (integer)
    topological dimension, whereas the attractor's Hausdorff dimension is non-integer (and exceeds
    its topological dimension). Non-integer dimension is therefore not a curiosity but a *proof* of
    non-manifoldness.
  - **Belief-state tie-in:** the Bayesian belief-update maps — one contraction per emission
    symbol — *are* exactly such an IFS on the simplex, and the reachable belief set is precisely
    their attractor. So "the belief geometry is a fractal" is a statement about the *process*
    (its update maps and their similarity dimension), not an artifact of finite sampling.

- **"Being a manifold" vs "lying on a manifold" — different questions.** A set can fail to be a
  manifold and still *sit inside* one, and that is exactly the case for belief geometry: the MSP
  attractor **is not** a manifold, but it **lies on** the 2-simplex $\Delta^2$ (equivalently, inside
  a 2-D affine subspace of the residual stream — this containment *is* the "linearly represented"
  result). The enclosing manifold cannot be made arbitrarily thin: any $k$-manifold containing a set
  of Hausdorff dimension $d_H$ requires $k\ge d_H$, so a fractal with $d_H\approx1.585$ **cannot lie
  on a curve** ($k=1$); the minimal enclosing manifold has dimension $2$. This is the §2
  occupied-vs-functional split in one line — *occupied set = fractal (not a manifold); enclosing /
  functional geometry = a manifold the fractal lives on.*

- **0-manifold:** a discrete set of points. Worth stating explicitly because it dissolves a
  common confusion: **"discrete" does not mean "non-manifold."** A finite set of clusters is a
  (disconnected) 0-manifold. What actually breaks manifold-ness is *mixed dimension* (→ stratified)
  or *non-integer dimension* (→ fractal), **not** discreteness and **not** disconnectedness.

### Quick corrections to common intuitions

| Intuition | Verdict |
|---|---|
| "Manifolds are closed/finite/bounded objects." | ✗ — they're *locally Euclidean*; can be infinite/non-compact. |
| "A boundary disqualifies something from being a manifold." | ✗ — it's a *manifold with boundary*. |
| "The corners of a simplex aren't locally linear." | ✓ — that makes it a *manifold with corners*. |
| "A circle and a line are locally linear everywhere." | ✓ — both are 1-manifolds. |
| "Discrete/disconnected clusters can't be a manifold." | ✗ — disjoint points/blobs form a (disconnected) 0-manifold. |
| "A fractal is just a weird manifold." | ✗ — not locally Euclidean anywhere (non-integer Hausdorff dimension is the certificate) ⇒ not a manifold at all. |

---

## 2. The reframe: three questions that make the question well-posed

"Is concept X a manifold?" only becomes answerable once you specify:

1. **Which set?**
   - **Generative ground-truth geometry** — the structure of the latent variable itself (e.g. the
     belief states the *process* actually produces).
   - **Occupied geometry** — the set of *activations the model actually fills* when run on data.
   - **Functional geometry** — the geometry the model's downstream computation *uses/respects*,
     which can extend smoothly into regions no data ever visits.

2. **Which limit?** Manifold-ness is a property of an *infinite* object, so it is **undecidable
   from a finite sample**: any finite point cloud lies on manifolds of *every* dimension from $0$
   (the points themselves) up to the ambient $N$, so "the dimension" is not even defined for finite
   data. The well-posed object is the **support of the representation measure** induced by the
   generating process — equivalently, the **attractor of the representation dynamics** — not the
   sample. (For an ergodic process the closure of the occupied set *is* this attractor.) Only once
   you name the process and pass to this limit does the question "fractal? circle? discrete?" have
   an answer.

3. **Which criterion?** Two genuinely different bars, and the gap between them is the whole point:
   - **Descriptive** — the geometry *fits the points*: there exists a (low-distortion, smooth)
     coordinate map from the occupied set onto $G$. This is cheap — you can fit a circle to almost
     anything.
   - **Causal** — the geometry is *used by the computation*: the model's readout **factors through
     coordinates on $G$** and behaves consistently with $G$'s structure (continuity along it, and
     **equivariance** under whatever group/metric $G$ carries), as verified by **intervention** —
     patch or steer along $G$ and check that predictions move the way $G$ predicts, including at
     interpolated points the data never visits.

   The causal criterion does more than rank the two — **it is what *picks out which geometry the
   system actually has*.** Descriptively, many geometries fit the same points; the intervention
   test decides which one is real *for the model*. In the days example the discrete points and the
   circle are *both* descriptively present, and only the causal question ("is the in-between used to
   compute day arithmetic?") tells you the circle is the functional geometry rather than an artifact
   of the layout. So **a geometry is not merely observed, it is determined by what the computation
   uses** — which is why §8 makes the intervention test the central open methodological item.

**A fully specified claim is therefore a triple** — *(which set, which limiting process, which
criterion)* — and "is concept $X$ a manifold?" is a function of that triple, not a property of $X$
alone. Most "is it a manifold?" confusions come from answering for different columns of question 1
(or silently mixing the descriptive and causal bars).

---

## 3. The taxonomy: one ladder, several orthogonal axes

Manifold-ness is not a single yes/no axis. A representation should be described along *all* of:

- **Topological type:** discrete (0-manifold) → manifold → manifold-with-boundary →
  manifold-with-corners → stratified space → fractal.
- **Dimension:** and whether it is *pure-dimensional* (a manifold) or *mixed-dimension* (→
  stratified).
- **Connectivity:** connected vs disconnected (clusters). *Disconnectedness does not break
  manifold-ness.*
- **Metric structure:** are distances meaningful, and which metric? (KL / Fisher information for
  belief states; cyclic distance for days.) Topology alone is blind to this.
- **Algebraic / symmetry structure:** is there a group acting? (Cyclic group $\mathbb{Z}/n$ for
  days and modular arithmetic.) This is often the *reason* the geometry has the shape it does.

A representation is a *point* in this multi-axis space, not a label on the topology axis alone.

---

## 4. Worked examples (the living core — add rows freely)

| Concept | Occupied set (data limit) | Functional geometry (causally used) | Algebraic structure | Manifold verdict |
|---|---|---|---|---|
| Days / months | discrete (7 / 12 pts) = **0-manifold** | circle $U(1)$ (rotation = "+k days") | cyclic group $\mathbb{Z}/n$ | occupied: 0-mfld; functional: 1-mfld |
| Modular addition (grokking) | $p$ discrete points | circle(s); addition = angle-add | $\mathbb{Z}/p \hookrightarrow U(1)$ | same shape as days |
| Belief states, single Mess3 | **fractal** (IFS attractor, Hausdorff dim $\approx 1.585$) | the simplex / affine subspace the belief-update acts on | semigroup of belief-update maps | occupied: **not a manifold**; functional: mfld-with-corners |
| Belief simplex $\Delta^{n-1}$ (the *space* of beliefs) | filled polytope | — | — | **manifold with corners** |
| Non-ergodic Mess3 mixture (the telescope) | **coupled fractal** across components | per-component affine subspaces | product/semigroup | occupied: not a manifold |
| Binary concept (e.g. `is_dog`) | 2 clusters / one direction | linear threshold; no meaningful interpolation | — | discrete / linear, not curved |
| Continuous magnitude (sentiment, number line, brightness) | a curve / interval | 1-manifold; interpolation *is* meaningful | ordered / affine | a genuine manifold |

The recurring lesson: **occupied geometry and functional geometry routinely differ.** Days are
discrete points (occupied) on a causally-real circle (functional). Belief states are a fractal
(occupied) embedded *linearly* in an affine subspace (functional). "Linearly decodable" and "is a
manifold" are **independent** properties — the Mess3 activations occupy a *fractal subset of a flat
subspace*.

> **Citations to slot in / verify:** circular day/month features — Engels et al., *"Not All Language
> Model Features Are Linear"* (2024); modular-addition circles — Nanda et al., *"Progress Measures for
> Grokking via Mechanistic Interpretability"*; belief-state geometry & linear embedding — Shai,
> Riechers et al., *"Transformers Represent Belief State Geometry in their Residual Stream."*

---

## 5. Catch-all terms: are they valid? Is there one umbrella?

This was the motivating question, so it gets its own section. Short answer: **the popular catch-all
terms are each valid for a *sub-family* but none covers everything we study, and that is itself the
finding worth reporting.**

- **"Manifold" (as a catch-all): invalid.** It excludes the two cases we care about most — the
  *corners* of the belief simplex (manifold-*with-corners*, not a manifold) and the *fractal*
  reachable set (not a manifold of any kind). Using "manifold" as the umbrella is the overclaim
  worth resisting.

- **"Stratified manifold / stratified space": the best umbrella for the *piecewise-manifold*
  family — but not for everything.** It correctly covers polytopes/simplices, unions of manifolds
  of mixed dimension, and discrete-clusters-on-a-circle. It **does not** cover fractals (non-integer
  Hausdorff dimension is not a stratified manifold), it is overkill for a plain manifold, and it is
  silent about the *group/metric* structure that explains *why* a representation has the shape it
  does. So: adopt "stratified space" as the umbrella for the non-fractal piecewise cases, but do not
  let it absorb the fractal case — keeping fractals as a *distinct* class is precisely what makes
  belief geometry a special, non-generic object.

- **"Topological structure" (as a catch-all): too narrow.** Topology captures connectivity, holes,
  and dimension, but is **blind to metric structure** (distances — KL/Fisher for beliefs, cyclic
  distance for days) and **blind to algebraic structure** (the *cyclic group* is the reason days
  form a circle rather than an interval; a circle vs an interval is an algebraic/geometric fact, not
  a purely topological one). Calling a representation "a topological structure" throws away the two
  layers that usually carry the meaning.

- **The honest umbrella: a "structured (low-dimensional) subset of activation space,"** described by
  the *multi-axis taxonomy* of §3 (topological type + dimension + connectivity + metric + symmetry)
  rather than by a single word. There is **no single existing term** that simultaneously covers
  {discrete set, manifold, manifold-with-corners, stratified space, fractal, embedded group}. If we
  want one, we have to either (a) coin it, or (b) — better — always report the *tuple of axes*. The
  closest honest phrase is **"the representational geometry of the concept,"** treated as a tuple,
  not a label.

  If a coined term is wanted, candidates worth bikeshedding: *"representational variety"* (borrowing
  "variety" from algebraic geometry, which already tolerates singularities), or simply
  *"concept geometry."* Flag: "variety" has a precise algebraic-geometry meaning (zero set of
  polynomials) and most of our objects are not varieties, so it would be a loose borrowing.

---

## 6. Concept vs. representation vs. feature

These three are used interchangeably and shouldn't be. Proposed split:

- **Concept** — a *functional / semantic* entity: a variable the system tracks (a latent state) or
  a category. Defined by its **role in the computation**, not by its shape. ("The day of the week,"
  "the current belief state," "whether the subject is a dog.")

- **Representation** — the *geometric object* in activation space that carries a concept. This is
  the thing the §3 taxonomy describes. One concept can have different representational geometries in
  different models or layers.

- **Feature** — **deliberately left overloaded, because the field has not converged.** Document the
  competing senses and pick one *locally* each time:
  1. a **direction / 1-D subspace** (linear representation hypothesis),
  2. an **SAE dictionary atom** (a learned, sparsely-activating direction),
  3. **any computed variable** / function of the input,
  4. Anthropic-style **interpretable direction** recovered by dictionary learning.

  Because senses 1–4 are mutually inconsistent, **"a feature can be almost anything" is a true
  statement about the literature, not a personal confusion.** Treat "feature" as needing a
  qualifier every time it's used.

**Corollary on "a concept is a topological structure":** closer to right is *a concept is carried
by a representation, and a representation has geometric structure that is simultaneously
topological, metric, and algebraic — which layer matters is dictated by the computation.* Topology
is one of three layers, not the whole thing.

---

## 7. Parameter space vs. representation space vs. behavior — which is fundamental, and what is gauge

A recurring objection: *parameter space is more fundamental than representation space, because many
different functions (parameter settings) can produce the same representations.* This is correct and
important, and it slots in directly above the "which set?" question of §2 as a question of **which
level of description** you are working at.

### The tower of many-to-one maps

$$\text{parameters } \theta\in\mathbb{R}^P \;\longrightarrow\; \text{representation geometry}
\;\longrightarrow\; \text{function / behavior (I/O map)}$$

Every arrow is **many-to-one**, so each level is more abstract and less redundant than the one
before. Parameter space is the most redundant description; behavior is the most invariant; the
representation geometry is the middle layer we can actually measure and where computation happens.
A claim about "the geometry" should say not only *which set* (§2.1) but *which level of this tower*,
and *modulo which group* (below).

### Why the first arrow collapses: gauge symmetry of parameters

Many distinct $\theta$ compute the *exact same function*, related by **parameter symmetries**:

- permutation of hidden units; ReLU **positive rescaling** ($W\to cW$ in, $\tfrac1c$ out); tanh
  **sign flips**;
- the **$GL$ freedom inside attention**: $W_Q,W_K$ enter only through $W_Q^{\!\top}W_K$ and
  $W_O,W_V$ only through $W_OW_V$, so $W_Q\to AW_Q,\,W_K\to A^{-\top}W_K$ (and analogously for
  $W_O,W_V$) is invisible to the function;
- softmax **shift** invariance; **LayerNorm** scale invariance.

These form a group $G$ acting on $\Theta$; the honest function space is the quotient $\Theta/G$
(the gauge-theory view of networks). *Beyond* exact symmetry, genuinely different circuits can
compute the same function — degeneracy that is not a symmetry at all.

### The second arrow has gauge freedom too

Representation geometry is itself only defined **up to the group the downstream readout cannot
see**. A linear probe/readout can absorb any invertible linear map, so beliefs are defined only up
to the affine/$GL$ group — which is exactly why we say they are "**linearly decodable**" rather than
"located at coordinates $x$." Two consequences:

- **Raw Euclidean distances in activation space are gauge-dependent.** A $GL$ change of basis warps
  them. So the **metric axis of §3 is only meaningful relative to a readout** — use the
  readout-induced metric (or Fisher information), not the ambient dot product.
- The *topological/affine/algebraic* structure survives the gauge group; specific coordinates and
  raw distances do not. Report the gauge-invariant structure.

### So is parameter space "more fundamental"? First disambiguate "fundamental"

The word hides two different notions, and the claim is true under one and false under the other:

- **Fundamental = generative substrate** — causally upstream; what is actually stored; what training
  optimizes; what governs learning and generalization.
- **Fundamental = invariant essence** — what is preserved / canonical / "the real thing," with the
  redundant description quotiented away.

**A note on the premise.** The argument usually offered — *"different functions produce the same
representations"* — is, read literally, a statement that the representation is the **canonical
quotient** you keep after modding out the differences between functions. That is a gauge-style
argument, and it supports **representation-primacy in the *essence* sense** — the opposite of the
conclusion it is used for. It supports *parameter*-primacy only via the *separate* substrate
argument. So "many functions → same representation, therefore parameters are more fundamental" does
not follow from its own premise; the two senses must be kept apart. With that done:

- **Substrate sense — parameters win.** The map $\theta\mapsto$ function is not merely symmetric but
  **singular**: its Jacobian degenerates on the symmetry orbits *and beyond them*, so the preimage of
  a function is a positive-dimensional, often singular variety rather than a discrete set of points.
  **Watanabe's singular learning theory** makes this the central object — the **real log canonical
  threshold (RLCT) / learning coefficient** replaces parameter-counting in the asymptotics of Bayesian
  generalization, and the *geometry of these singularities* controls what is learned, when (the basis
  of **developmental interpretability** — staged learning, phase transitions, the Timaeus program).
  At this level the representation is downstream and partly *underdetermined* by behavior, and
  parameter-space geometry is genuinely primary.
- **Essence sense — the representation wins, when the task pins it.** Computational mechanics says
  optimal prediction $\Rightarrow$ belief states $\Rightarrow$ the mixed-state-presentation geometry;
  so *any* network that predicts well must carry that geometry (up to the gauge group), whatever its
  parameters. There the geometry is an **invariant of the problem**, and the parameters are the
  redundant description.

### The tension, stated cleanly

Two true statements coexist:

1. **Degeneracy from below:** parameters $\to$ representation is many-to-one (different functions,
   same representation).
2. **Canonicity from above:** task $\to$ representation geometry is (nearly) one-to-one (the same
   geometry is *forced* by the prediction problem).

So "fundamental" is **not a single ranking**. It depends on what you hold fixed: the *weights*
(then parameter space is primary and representations are derived) or the *task* (then the
representation geometry is the canonical invariant and the weights are the redundant coordinates).
This document's own lean: parameter-space (singular) geometry is the right level for *how and when*
structure forms and generalizes; the task-pinned representation geometry is the right level for
*what concept is represented*. The dinner claim wins the first question, not the second.

### Don't conflate the two symmetries

- **Intrinsic / algebraic symmetry** (§3): the cyclic group of days, part of *the concept* —
  structure to be **discovered**.
- **Gauge symmetry** (this section): permutation / rescaling / $GL$ — redundancy of *the
  description* — structure to be **quotiented out**.

They point in opposite methodological directions and should never be merged.

---

## 8. Open questions / TODO

- Make the causal criterion (§2.3) operational for our setting: patch an *interpolated, off-attractor*
  belief state into the residual stream and test whether predictions move along the simplex
  coherently. This is the concrete test of "functional geometry" for belief states.
- Settle whether to coin an umbrella term or commit to reporting the §3 axis-tuple. (Current lean:
  report the tuple; resist coinage.)
- Verify/complete the citations in §4.
- Add rows: positional/rotary structure, color, syntax trees (genuinely a CW-/simplicial-complex
  case?), and any concept where occupied and functional geometry are claimed to *coincide*.
- Decide whether "stratified space" or "concept geometry (as a tuple)" is the term we put forward to
  the wider community.
- (§7) Test the **canonicity-from-above** claim directly: is the recovered belief geometry invariant
  across seeds/architectures *up to the gauge group*? A positive result is a universality claim;
  measure it with a gauge-invariant comparison (readout-induced metric, or a CKA-style invariant),
  not raw activation distances.
- (§7) Pin down citations for the parameter/representation/behavior section: singular learning theory
  (Watanabe; RLCT / learning coefficient), developmental interpretability / phase transitions
  (Timaeus), permutation symmetry / mode connectivity ("git re-basin", Ainsworth et al.),
  representation similarity up to transform (CKA, Kornblith et al.), and computational mechanics
  (Crutchfield) for the task-pins-geometry direction.
