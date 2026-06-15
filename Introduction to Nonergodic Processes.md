---
author: Kyle Ray
date: 2026-02-09
tags:
  - theory
  - guide
  - kyle-ray
  - factored-representations
  - nonergodic
---

## Overview

A nonergodic process is one where the long-run behavior depends on initial conditions. Unlike ergodic processes, which eventually "forget" their starting point, nonergodic processes carry the influence of their initial state forever.

In our framework, nonergodic processes arise as a "grab bag" of several ergodic components. One of these ergodic component is drawn from the bag at the start, and that component generates all subsequent tokens. The observer never sees the selection directly — they must infer it from the token stream.

This document is meant to be an easy to read entry point that covers the basics of how to approach such nonergodic compositions in our framework. This includes working out a simple example, and talking over the high-level expectations that should carry over form previous work.

## Intuition: Hidden Coin

Imagine someone has two coins — one fair and one biased toward heads. They secretly pick one coin and start flipping it. You see only the sequence of outcomes: H, T, H, H, T, H, H, ...

At first, you're uncertain which coin is generating the sequence. But as flips accumulate, the frequency of heads should reveal which coin was likely chosen. This is a nonergodic process.

- The "which coin" variable is set once and never changes
- Different initial selections produce different long-run statistics
- An observer gradually resolves uncertainty about the active component

## GHMM Recap

A generalized Hidden Markov Model (GHMM) is defined by the tuple:

$$\mathcal{M} = \left(\mathcal{X},\; \mathcal{S},\; \eta^{(\emptyset)},\; \left(T^{(x)}\right)_{x \in \mathcal{X}}\right)$$

where $\mathcal{X}$ is the token alphabet, $\mathcal{S} = \mathbb{R}^d$ is the latent space, $\eta^{(\emptyset)}$ is the initial state vector, and $T^{(x)}$ is the symbol-labeled transition matrix for token $x$.

The probability of a sequence is computed by chaining the transition matrices:

$$Q_\mathcal{M}(x_{1:\ell}) = \eta^{(\emptyset)} \, T^{(x_1)} \cdots T^{(x_\ell)} \, \mathbf{1}$$

## The Direct Sum Construction

Given $N$ component GHMMs $\mathcal{M}_1, \ldots, \mathcal{M}_N$ the **nonergodic composition** forms a new GHMM whose symbol-labeled transition matrices are the direct sum of the component matrices:

$$T^{(x)} = \bigoplus_{n=1}^{N} T_n^{(x)} = \begin{pmatrix} T_1^{(x)} & & 0 \\ & \ddots & \\ 0 & & T_N^{(x)} \end{pmatrix}$$

The block-diagonal structure is the key property: since the off-diagonal blocks are zero, a state in block $n$ can never transition to block $m \neq n$. The process is permanently confined to whichever block it starts in.

Because of the block-diagonal structure, the resulting sequence distribution is a mixture:
$$Q_\mathcal{M}(x_{1:\ell}) = \sum_{n=1}^{N} \pi_n \, Q_{\mathcal{M}_n}(x_{1:\ell})$$
With the $\pi_n$ reflecting the mixture weights over the components. Here $\pi_n \geq 0$ and $\sum_n \pi_n = 1$.

The latent space of the composed process is the direct sum of the component spaces:

$$\mathcal{S} = \bigoplus_{n=1}^{N} \mathcal{S}_n$$

with $\sum_n (d_n - 1) + (N - 1)$ effective degrees of freedom: $d_n - 1$ within each component (normalization removes one dimension per block) and $N - 1$ for the mixing weights. This grows linearly in both $N$ and the $d_n$. The initial state for a given realization can be sampled according to a distribution over $\mathcal{S}$ that encodes both the weights $\pi_n$ over components and the initial distribution $\eta_n^{(\emptyset)}$ within each component. The elements of this distribution can also be thought of as the elements of the initial predictive vector for a Bayes-optimal observer:

$$\eta^{(\emptyset)} = \left(\pi_1 \, \eta_1^{(\emptyset)},\; \pi_2 \, \eta_2^{(\emptyset)},\; \ldots,\; \pi_N \, \eta_N^{(\emptyset)}\right)$$

### Belief Dynamics

The **predictive vector** after observing context $x_{1:\ell}$ is the normalized product:

$$\eta^{(x_{1:\ell})} := \frac{\eta^{(\emptyset)} \, T^{(x_1)} \cdots T^{(x_\ell)}}{\eta^{(\emptyset)} \, T^{(x_1)} \cdots T^{(x_\ell)} \, \mathbf{1}}$$

Because $T^{(x)}$ is block-diagonal,  the unnormalized predictive vector separates by block:

$$\eta^{(\emptyset)} T^{(x_1)} \cdots T^{(x_\ell)} = \left(\pi_1 \, \eta_1^{(\emptyset)} T_1^{(x_1)} \cdots T_1^{(x_\ell)},\; \ldots,\; \pi_N \, \eta_N^{(\emptyset)} T_N^{(x_1)} \cdots T_N^{(x_\ell)}\right)$$

The normalization $Z(1:\ell)$ can also be decomposed into per component blocks

$$Z(1:\ell) = \sum_{m=1}^{N} \pi_m \, Q_{\mathcal{M}_m}(x_{1:\ell})$$

Every block shares this same denominator. The $n$-th block of the normalized vector is:

$$\frac{\pi_n \, \eta_n^{(\emptyset)} T_n^{(x_1)} \cdots T_n^{(x_\ell)}}{Z(1:\ell)}$$

This can be factored by multiplying and dividing by $Q_{\mathcal{M}_n}(x_{1:\ell})$:

$$= \frac{\pi_n \, Q_{\mathcal{M}_n}(x_{1:\ell})}{Z(1:\ell)} \;\cdot\; \frac{\eta_n^{(\emptyset)} T_n^{(x_1)} \cdots T_n^{(x_\ell)}}{Q_{\mathcal{M}_n}(x_{1:\ell})} \;=\; w_n(1:\ell) \;\cdot\; \eta_n^{(x_{1:\ell})}$$

The first factor is a scalar — the posterior component weight:

$$w_n(1:\ell) = \frac{\pi_n \, Q_{\mathcal{M}_n}(x_{1:\ell})}{Z(\ell)}$$

The second factor is the locally normalized predictive vector within component $n$:

$$\eta_n^{(x_{1:\ell})} = \frac{\eta_n^{(\emptyset)} \, T_n^{(x_1)} \cdots T_n^{(x_\ell)}}{Q_{\mathcal{M}_n}(x_{1:\ell})}$$

Collecting all blocks, the predictive vector decomposes as:

$$\eta^{(x_{1:\ell})} = \left(w_1(1:\ell) \, \eta_1^{(x_{1:\ell})},\; \ldots,\; w_N(\ell) \, \eta_N^{(x_{1:\ell})}\right)$$

The next-token probability under component $n$ is read off from its local predictive vector:

$$P_n(x_{\ell+1} \mid x_{1:\ell}) = \eta_n^{(x_{1:\ell})} \, T_n^{(x_{\ell+1})} \, \mathbf{1}_n$$

The weight update follows: after observing one more token $x_{\ell+1}$,

$$w_n(1:\ell+1) \propto w_n(1:\ell) \cdot P_n(x_{\ell+1} \mid x_{1:\ell})$$

Components that better predict the observed data receive increasing weight over time.

A note on notation: the weights $w_n(1:\ell)$ are deterministic functions of the full observed sequence $x_{1:\ell}$, not just its length. We write $w_n(\ell)$ form here on out because the context sequence is always understood, and the lighter notation keeps formulas readable — especially alongside $\eta_n^{(x_{1:\ell})}$, which already carries the full context explicitly.

> [!summary]
>
> Belief states updating in a nonergodic composition has two layers:
>
> - **Which component?** The weights $w_n$ track uncertainty about the identity of the active process. This uncertainty resolves over time as evidence accumulates.
> - **Where within the component?** Each $\eta_n^{(x_{1:\ell})}$ tracks the belief state *within* component $n$, evolving according to that component's own dynamics.
>
> Early in a sequence, both layers are active — the observer is simultaneously unsure which process is running *and* what state it's in. As the sequence grows, the component identity can potentially converge (the weights concentrate), after which the problem reduces to standard state estimation within the winning components.

<!-- -->

> [!note] Generative and Inference Priors
>
> The "mixing weights" appearing in the initial state vector above can play two distinct roles that are worth separating explicitly.
>
> Let's call one the **generative prior** $\pi_n$ , which is the actual probability  for selecting component $n$ and is a fixed property of the data source.
>
> The **inference prior** $w_n(0)$ is an observer's initial belief about which component is active at the start of a new sequence.
>
> The observers initial predictive vector encodes their inference prior:
>
> $$\eta^{(\emptyset)} = \left(w_1(0) \, \eta_1^{(\emptyset)},\; \ldots,\; w_N(0) \, \eta_N^{(\emptyset)}\right)$$
>
> A Bayes-optimal observer sets $w_n(0) = \pi_n$, minimizing expected log-loss. But the two can differ: a new observer might, for example, always use a uniform prior $w_n(0) = 1/N$. If the generative prior is not uniform, the observer and must *learn* the true generative prior from training data to use as its inference prior. In the examples that follow, we use uniform generator priors and assume optimal predictors.

## Examples: Biased Coin and the Even Process

> [!example] The Even Process
>
> The Even Process is a 2-state HMM over the binary alphabet $\mathcal{X} = \{0, 1\}$, parameterized by $p \in (0, 1)$:
>
> - **State A**: emit 1, stay in A (probability $p$); emit 0, go to B (probability $1 - p$)
> - **State B**: emit 0, go to A (probability 1)
>
> Zeros always appear in runs of even length: after emitting a 0 from state A (entering B), the process *must* emit another 0 to return to A.
>
> With $p = 1/2$, the symbol-labeled transition matrices are:
>
> $$T_{\text{even}}^{(1)} = \begin{pmatrix} 1/2 & 0 \\ 0 & 0 \end{pmatrix}, \qquad T_{\text{even}}^{(0)} = \begin{pmatrix} 0 & 1/2 \\ 1 & 0 \end{pmatrix}$$
>
> The net transition matrix $T = T^{(0)} + T^{(1)}$ has stationary distribution $\eta^{\emptyset}_{\text{even}} = (2/3,\; 1/3)$, giving marginal token probabilities $P(0) = 2/3$ and $P(1) = 1/3$.

<!-- -->

> [!example] Forbidden Words
>
> A *forbidden word* is a finite string that has zero probability of appearing anywhere in a stationary sequence. For the Even Process, the minimal forbidden word is **"101"** — a single zero flanked by two ones. The transition matrix product confirms this:
>
> $$T_{\text{even}}^{(1)} \, T_{\text{even}}^{(0)} \, T_{\text{even}}^{(1)} = \begin{pmatrix} 1/2 & 0 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} 0 & 1/2 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 1/2 & 0 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$$
>
> The zero matrix means "101" has probability zero from any starting state. More generally, any odd-length run of zeros between ones is forbidden: "101", "10001", "1000001", and so on.

<!-- -->

> [!example] Matching Marginals with a Biased Coin
>
> Now define a memoryless (i.i.d.) process — a biased coin — with $P_{\text{coin}}(0) = 2/3$ and $P_{\text{coin}}(1) = 1/3$. As a GHMM, this is a 1-state process:
>
> $$T_{\text{coin}}^{(0)} = (2/3), \qquad T_{\text{coin}}^{(1)} = (1/3)$$
>
> The coin has the same marginal token distribution as the Even Process, so any single token provides zero information about which component is active.

<!-- -->

> [!example] The Composed Process
>
> The nonergodic composition has a 3-dimensional latent space $\mathbb{R}^1 \oplus \mathbb{R}^2 = \mathbb{R}^3$, but only 2 effective degrees of freedom: the coin's 1-state space is fully determined($d_1-1=0$) , the Even Process contributes 1 ( $d_2 - 1 = 1$), and the mixing weight adds 1 ($N - 1 = 1$). The coin occupies the first coordinate and the Even Process the remaining two. The $3 \times 3$ block-diagonal transition matrices are:
>
> $$T^{(0)} = \begin{pmatrix} 2/3 & 0 & 0 \\ 0 & 0 & 1/2 \\ 0 & 1 & 0 \end{pmatrix}, \qquad T^{(1)} = \begin{pmatrix} 1/3 & 0 & 0 \\ 0 & 1/2 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$
>
> With equal inference priors $w_{\text{coin}}(0) = w_{\text{even}}(0) = 1/2$ and the Even Process starting in its stationary distribution, the initial state vector is:
>
> $$\eta^{(\emptyset)} = \left(\tfrac{1}{2} \cdot 1,\;\; \tfrac{1}{2} \cdot \tfrac{2}{3},\;\; \tfrac{1}{2} \cdot \tfrac{1}{3}\right) = \left(\tfrac{1}{2},\; \tfrac{1}{3},\; \tfrac{1}{6}\right)$$

<!-- -->

> [!example] Worked Belief Updates: The Sequence "1, 0, 1"
>
> **After observing "1":**
>
> $$\eta^{(\emptyset)} T^{(1)} = \left(\tfrac{1}{2},\; \tfrac{1}{3},\; \tfrac{1}{6}\right) \begin{pmatrix} 1/3 & 0 & 0 \\ 0 & 1/2 & 0 \\ 0 & 0 & 0 \end{pmatrix} = \left(\tfrac{1}{6},\; \tfrac{1}{6},\; 0\right)$$
>
> Normalizing yields $\eta^{(1)} = (1/2,\; 1/2,\; 0)$.
>
> Component weights: $w_{\text{coin}} = 1/2$, $w_{\text{even}} = 1/2$.
>
> Both components predict "1" with probability $1/3$, so the weights are unchanged. The Even Process is now known to be in state A (only A can emit 1).
>
> **After observing "10":**
>
> $$\eta^{(1)} T^{(0)} = \left(\tfrac{1}{2},\; \tfrac{1}{2},\; 0\right) \begin{pmatrix} 2/3 & 0 & 0 \\ 0 & 0 & 1/2 \\ 0 & 1 & 0 \end{pmatrix} = \left(\tfrac{1}{3},\; 0,\; \tfrac{1}{4}\right)$$
>
> Normalizing yields $\eta^{(10)} = (4/7,\; 0,\; 3/7)$.
>
> Component weights: $w_{\text{coin}} = 4/7 \approx 0.571$, $w_{\text{even}} = 3/7 \approx 0.429$.
>
> The coin predicts "0" with probability $2/3$; the Even Process in state A predicts it with only $1/2$. The coin's better prediction earns it weight. The Even Process is now in state B (third coordinate only).
>
> **After observing "101":**
>
> $$\eta^{(10)} T^{(1)} = \left(\tfrac{4}{7},\; 0,\; \tfrac{3}{7}\right) \begin{pmatrix} 1/3 & 0 & 0 \\ 0 & 1/2 & 0 \\ 0 & 0 & 0 \end{pmatrix} = \left(\tfrac{4}{21},\; 0,\; 0\right)$$
>
> Component weights: $w_{\text{coin}} = 1$, $w_{\text{even}} = 0$.
>
> The Even Process is eliminated. State B can only emit 0, so seeing "1" is impossible — the forbidden word "101" collapses the posterior to certainty.

<!-- -->

> [!example] An alternate timeline: "1, 0, 0"
>
> What if the third token had been "0" instead of "1"? The first two steps are identical: after "10" the weights are $w_{\text{coin}} = 4/7$, $w_{\text{even}} = 3/7$, and the Even Process is in state B.
>
> **After observing "100":**
>
> $$\eta^{(10)} T^{(0)} = \left(\tfrac{4}{7},\; 0,\; \tfrac{3}{7}\right) \begin{pmatrix} 2/3 & 0 & 0 \\ 0 & 0 & 1/2 \\ 0 & 1 & 0 \end{pmatrix} = \left(\tfrac{8}{21},\; \tfrac{3}{7},\; 0\right)$$
>
> Normalizing yields $\eta^{(100)} = (8/17,\; 9/17,\; 0)$.
>
> Component weights: $w_{\text{coin}} = 8/17 \approx 0.471$, $w_{\text{even}} = 9/17 \approx 0.529$.
>
> The weight swings back toward the Even Process: from state B it predicted "0" with certainty ($P = 1$), beating the coin's $2/3$.

<!-- -->

> [!example] What This Example Shows
>
> The third token is the fork: "1" triggers the forbidden word and produces instant certainty; "0" aligns with the Even Process's paired-zero structure and shifts the balance in its favor.
>
> - **Matched marginals neutralize individual tokens.** Single observations may not carry discriminative information; only sequential context helps.
> - **Shared words are evidence.**  Different components' probabilities diverge based on internal dynamics when conditioned on observations.
> - **Forbidden words are conclusive evidence.** A single occurrence eliminates a component entirely, driving its posterior weight to exactly zero.

## Contrast with Multipartite Processes

Nonergodic (direct sum) and multipartite (tensor product) processes represent two fundamentally different kinds of composition. In the notation of the *Transformers learn factored representations* paper:

| Property           | Nonergodic ($\bigoplus$)                   | Multipartite ($\bigotimes$)        |
| ------------------ | ------------------------------------------ | ---------------------------------- |
| Matrix structure   | Block-diagonal                             | Kronecker product                  |
| Components         | One active at a time                       | All active simultaneously          |
| Effective DOF      | $\sum_n (d_n - 1) + (N-1)$                 | $\prod_n d_n - 1$                  |
| Observer's task    | Identify which component and track only it | Track all components in parallel   |
| Token construction | Shared/Overlapping alphabet                | Cartesian product of sub-alphabets |

In multipartite processes, all factors run in parallel and each contributes a sub-token combined into the observed output. In nonergodic processes, one component is selected and runs alone.

A transformer encountering nonergodic data must solve a *model selection* problem (which process am I observing?) in addition to the usual *state estimation* problem (what state is the process in?). This two-layered inference structure is a distinctive signature of nonergodic compositions.

## Factored Components

What happens when some or all of the ergodic components are themselves *factored* — multipartite processes whose transition operators are tensor products of simpler sub-processes? This produces a nested composition: a direct sum of tensor products.

### The Construction

Suppose component $\mathcal{M}_n$ in the nonergodic composition is itself a multipartite process with $K_n$ conditionally independent factors. Then its transition operators take the tensor-product form:

$$T_n^{(x)} = \bigotimes_{k=1}^{K_n} T_{n,k}^{(x)}$$

where $T_{n,k}^{(x)}$ is the transition operator for the $k$-th factor within component $n$, acting on a local latent space $\mathcal{S}_{n,k}$ of dimension $d_{n,k}$.

The full nonergodic composition then has transition matrices that are block-diagonal, with each block being a tensor product:

$$T^{(x)} = \bigoplus_{n=1}^{N} \bigotimes_{k=1}^{K_n} T_{n,k}^{(x)}$$

The latent space is:

$$\mathcal{S} = \bigoplus_{n=1}^{N} \bigotimes_{k=1}^{K_n} \mathcal{S}_{n,k}$$

with $\sum_{n=1}^{N} (\prod_{k=1}^{K_n} d_{n,k} - 1) + (N-1)$ effective degrees of freedom.

### Belief Dynamics with Factored Components

The predictive vector retains the block structure from the nonergodic composition:

$$\eta^{(x_{1:\ell})} = \left(w_1(\ell) \, \eta_1^{(x_{1:\ell})},\; \ldots,\; w_N(\ell) \, \eta_N^{(x_{1:\ell})}\right)$$

When component $n$ has conditionally independent factors, its local predictive vector stays on the product-state submanifold:

$$\eta_n^{(x_{1:\ell})} = \bigotimes_{k=1}^{K_n} \eta_{n,k}^{(x_{1:\ell})}$$

Conditional independence ensures this product structure is preserved under belief updates. Applying the Factored World Hypothesis (FWH) within each block, the local representation further decomposes into orthogonal subspaces:

$$\eta_{n,\text{FWH}}^{(x_{1:\ell})} = \bigoplus_{k=1}^{K_n} \tilde{\eta}_{n,k}^{(x_{1:\ell})}$$

where $\tilde{\eta}_{n,k}$ is the embedded local predictive vector for the $k$-th factor of component $n$, projected onto the subspace orthogonal to $\mathbf{1}_{n,k}$.

The overall factored representation combines both levels:

$$\eta_{\text{FWH}}^{(x_{1:\ell})} = \bigoplus_{n=1}^{N} w_n(\ell) \bigoplus_{k=1}^{K_n} \tilde{\eta}_{n,k}^{(x_{1:\ell})}$$

This is a direct sum (over components) of direct sums (over factors within each component).

#### Three Levels of Inference

The observer faces a three-level inference problem:

- **Component identification**: Track the posterior weights $w_n(\ell)$ to determine which component is active.
- **Factor tracking**: Within the active component, track each factor's state in its own orthogonal subspace.
- **Cross-factor correlations**: If factors are not perfectly conditionally independent, the factored representation is lossy and the observer may need to expand into a higher-dimensional joint representation.

### When Only Some Components Are Factored

Components need not share the same internal structure:

| Component       | Structure                     | Joint dim                   | Factored dim                |
| --------------- | ----------------------------- | --------------------------- | --------------------------- |
| $\mathcal{M}_1$ | Single 3-state HMM            | 2                           | 2 (no factoring)            |
| $\mathcal{M}_2$ | Product of two 3-state HMMs   | 8                           | $2 \times 2 = 4$            |
| $\mathcal{M}_3$ | Product of three 3-state HMMs | 26                          | $3 \times 2 = 6$            |
| **Total**       |                               | **38** (36 + 2 for weights) | **14** (12 + 2 for weights) |

### Connection to the Factored World Hypothesis

The Factored World Hypothesis predicts that transformers have an inductive bias toward factored representations. In this context:

- **Within each block**, the transformer should discover factored structure and represent each factor in a separate orthogonal subspace. The block-diagonal structure prevents different components' representations from interfering.
- **When factoring is lossy** (residual cross-factor correlations), the same efficiency-vs-fidelity tradeoff arises.
- **Subspace reuse across components** is an open question. Since only one component is active at a time, factored subspaces within different components could in principle overlap — unlike in multipartite processes where all subspaces must be simultaneously orthogonal.
