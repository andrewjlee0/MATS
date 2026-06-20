# The Geometry of Concepts in Activation Space — a working vocabulary

**Status:** living document. Started 2026-06-20 (Andrew + collaborators, out of a MATS dinner
discussion). Corrections, counterexamples, and new rows in the example table are all welcome —
edit freely.

**Audience:** interpretability researchers, especially the belief-state-geometry crowd. Assumes
light familiarity with HMMs / mixed-state presentations but tries to define terms as it goes.

---

## 0. Why this document exists

There is no consensus on what we mean when we say a concept is "a manifold," "a feature," "a
direction," or "a topological structure." The terms are used loosely and interchangeably, and
debates about whether some representation "is or isn't a manifold" often turn out to be people
pointing at *different objects* with the *same word*. This document tries to pin the vocabulary
down precisely enough that the well-known examples (belief-state fractals, days-of-the-week
circles, modular arithmetic, binary concepts) can all be described without contradiction.

The central claim: **"is concept X a manifold?" is not a well-posed question until you say
*which set*, *in which limit*, and *under which criterion* you mean.** Once you fix those three,
the disagreements dissolve.

---

## 1. Definitions, stated precisely

The single most common error is treating "manifold" as a synonym for "smooth blob" or "closed
shape." It is neither. The precise ladder:

- **Topological manifold (dimension $n$):** a (Hausdorff, second-countable) space in which *every
  point has a neighborhood homeomorphic to $\mathbb{R}^n$*. The defining property is **locally
  Euclidean** ("locally linear," informally) — nothing about being closed, bounded, or finite.
  - $\mathbb{R}^n$, an infinite line, and the plane are all manifolds. **Manifolds need not be
    compact or bounded.**
  - A **circle** and a **sphere** are manifolds that happen to be compact.

- **"Closed" is a trap word.** It means two unrelated things: (a) a topologically closed *set*,
  and (b) a **closed manifold** = *compact and without boundary* (circle, sphere, torus). A line
  is a manifold but not a *closed manifold*. Avoid the bare word "closed" in geometry discussions.

- **Manifold with boundary:** boundary points get charts onto a *half-space* $\{x_n \ge 0\}$
  instead of all of $\mathbb{R}^n$. A filled disk is the canonical example.

- **Manifold with corners:** corner points get charts onto an *orthant* $[0,\infty)^n$. A filled
  triangle (the 2-simplex) is the canonical example — edges are half-space points, vertices are
  corner points.

- **Smooth manifold:** a topological manifold with a compatible differentiable atlas, so tangent
  spaces (local *linearizations*) exist. "Locally linear" in the ML sense (LLE, tangent-plane
  approximation) is really *smooth*-manifold language.

- **Stratified space:** a space decomposed into a union of manifolds ("strata") of *possibly
  different dimensions*, glued along their boundaries (e.g. a polytope = 2-faces ∪ edges ∪
  vertices). The right framework for *mixed-dimension, piecewise-manifold* objects.

- **Fractal / IFS attractor:** a self-similar set, typically of *non-integer Hausdorff
  dimension*, that is **not locally Euclidean anywhere** and therefore **not a manifold of any
  kind**. The attractor of an Iterated Function System (a finite set of contraction maps) is the
  prototypical example.

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
| "A fractal is just a weird manifold." | ✗ — non-integer dimension ⇒ not a manifold at all. |

---

## 2. The reframe: three questions that make the question well-posed

"Is concept X a manifold?" only becomes answerable once you specify:

1. **Which set?**
   - **Generative ground-truth geometry** — the structure of the latent variable itself (e.g. the
     belief states the *process* actually produces).
   - **Occupied geometry** — the set of *activations the model actually fills* when run on data.
   - **Functional geometry** — the geometry the model's downstream computation *uses/respects*,
     which can extend smoothly into regions no data ever visits.

2. **Which limit?** *Any finite point cloud is simultaneously "a discrete set of points" and "a
   sample of a manifold."* All real data is finite, so manifold-ness is **undecidable from data
   alone**. You must specify the infinite-data / generative limit: does the reachable set converge
   to a fractal? a circle? does it stay discrete?

3. **Which criterion?**
   - **Descriptive** — you *can fit* the geometry (you can fit a circle to almost anything).
   - **Causal** — the model's computation *respects* the geometry: interventions/interpolations
     along it move predictions coherently. This is the criterion that actually matters, and the one
     that connects the math to mechanistic interpretability.

**Most "is it a manifold?" disagreements are people answering for different columns of question 1.**

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

The recurring lesson: **occupied geometry and functional geometry routinely disagree.** Days are
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
  reachable set (not a manifold of any kind). Using "manifold" as the umbrella is exactly the
  overclaim that started the dinner argument.

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

## 7. Open questions / TODO

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
