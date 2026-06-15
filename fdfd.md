Read it. Crucially, that doc never says "cone" or "prism" — those are Kyle's geometric nicknames
  (from the chat / his other notes) for two objects the doc does define. Once you see they're the
  same vector with one difference, it clicks. And I had them backwards in my earlier message — let
  me fix that too.

  Both are the same stacked vector. They differ by one division.

  The doc's central object is the block-stacked predictive vector — one block per component, each
  block = (component weight) × (within-component belief). It shows up in two forms:

  Prism = the unnormalized version (doc eq. at line 70):
  $$\eta^{(\emptyset)} T^{(x_1)}\cdots T^{(x_\ell)} = \big(\pi_1,\eta_1^{(\emptyset)}T_1\cdots,\
  \ldots\big) = \big(\underbrace{\pi_n Q_n}_{\text{block magnitude}},\eta_n\big)n$$
  Block $n$'s length is $\pi_n Q_n$ (prior × component-likelihood). The **whole vector's length is 
  $Z = Q(x{1:\ell})$** — the probability of the sequence so far. As the sequence grows, $Z$ shrinks
  toward 0 (long strings are improbable), so the prism physically shrinks over time.

  Cone = the normalized version (doc eq. at line 94):
  $$\eta^{(x_{1:\ell})} = \big(w_1\eta_1,\ \ldots,\ w_N\eta_N\big),\qquad w_n = \frac{\pi_n
  Q_n}{Z}$$
  This is just the prism divided by $Z$. Now the total length is always 1 — it's a genuine
  probability distribution over (component, within-state). Block $n$'s magnitude is the posterior 
  weight $w_n$.

  That's the whole distinction: prism ÷ Z = cone. Nothing else differs. (The names are presumably
  geometric: stacking the within-component simplices weighted by $w_n$ traces a cone; keeping the
  extra $Z$ "likelihood" axis extrudes that cone into a prism. I'm guessing at the naming; the math
  above is the solid part.)

  Your "telescope" is this same stacked object — the cone is its normalized form.

  Why that one division is exactly the probs test

  The model needs the conditional next-token prob $P(x_{\ell+1}\mid x_{1:\ell})$. In block form
  it's emission · (normalized vector):

  - From the cone $(w_n\eta_n)$: $P(\text{next}) = \sum_n w_n,\eta_n T_n\mathbf 1$. One matrix
  multiply. Linear. ✓ ("cone maps linearly to probs.")
  - From the prism $(\pi_n Q_n\eta_n)$: the same multiply gives $\sum_n \pi_n Q_n,\eta_n T_n\mathbf
  1 = Z\cdot P(\text{next})$ — the joint, not the conditional. To recover the conditional you must
  divide by $Z$ = the sum of the prism's own coordinates. Dividing by a sum of your own 
  coordinates is not a linear operation. ✗ ("prism doesn't even map linearly to probs.")

  So the prism fails the probs test for one reason only: it still has the normalizer baked in, and
  pulling it out is nonlinear. The cone already paid that cost.

  If you want it fully concrete, I verified it on the doc's own "1,0" example: after "1" the prism
  is $(\tfrac16,\tfrac16,0)$, and reading "next = 0" off it linearly gives the joint
  $P(10)=\tfrac{7}{36}$, whereas the cone $(\tfrac12,\tfrac12,0)$ gives the conditional $P(0\mid
  1)=\tfrac{7}{12}$ directly. The gap is the factor $Z=\tfrac13$.

  Correcting myself

  Two messages ago I guessed "cone = unnormalized forward vector." That was backwards — per Kyle's
  usage (which you just gave me, and which the linear-to-probs test confirms), cone = normalized,
  prism = unnormalized. Sorry for adding to the fog.

  Is this the piece that was missing — what the two objects actually are — or is the part still
  bugging you the separate question of why log-probs (not probs) is the test Kyle says "actually 
  matters"? Those are two different confusions and I want to hit the right one.