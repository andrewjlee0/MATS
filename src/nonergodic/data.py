"""GPU-vectorized HMM sampling and Bayesian filtering for the non-ergodic
Mess3 experiment. Mirrors data.py but runs the per-position loops as batched
tensor ops on the GPU (all sequences advanced in parallel each step).

Use these in train.py to keep the whole pre-training phase on the GPU.
The numpy versions in data.py are kept for compatibility with other code.
"""
import numpy as np
import torch


def _to_dev(T_stack, pis, cfg, device):
    K = len(cfg.comp_params)
    T = torch.stack([torch.as_tensor(T_stack[k], dtype=torch.float32, device=device)
                     for k in range(K)])               # (K, V, S, S)
    pi = torch.stack([torch.as_tensor(pis[k], dtype=torch.float32, device=device)
                      for k in range(K)])               # (K, S)
    return T, pi


def sample_sequences_gpu(cfg, T_stack, pis, comps, device, seed=0):
    """Vectorized HMM sampling on the GPU.

    comps: (N,) int64 component index per sequence (tensor or array).
    Returns (N, seq_len) int64 token tensor on `device`. Loops over positions
    (sequential dependency) but samples all N sequences in parallel each step.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    comps = torch.as_tensor(comps, dtype=torch.long, device=device)
    N, L, S, V = len(comps), cfg.seq_len, cfg.n_states, cfg.vocab

    T, pi = _to_dev(T_stack, pis, cfg, device)           # (K,V,S,S), (K,S)
    Tc = T[comps]                                        # (N, V, S, S) per-seq
    ar = torch.arange(N, device=device)

    state = torch.multinomial(pi[comps], 1, generator=g).squeeze(1)   # (N,)
    tokens = torch.empty(N, L, dtype=torch.long, device=device)

    for t in range(L):
        # P(token, next_state | state): gather current-state slice -> (N, V, S)
        row = Tc[ar[:, None], torch.arange(V, device=device)[None, :], state]  # (N,V,S)
        tok_p = row.sum(-1)                              # (N, V)  P(token | state)
        tok_p = tok_p / tok_p.sum(-1, keepdim=True)
        z = torch.multinomial(tok_p, 1, generator=g).squeeze(1)       # (N,)
        tokens[:, t] = z
        nsp = Tc[ar, z, state]                           # (N, S)  next-state dist
        nsp = nsp / nsp.sum(-1, keepdim=True)
        state = torch.multinomial(nsp, 1, generator=g).squeeze(1)
    return tokens


def all_telescoped_gpu(cfg, T_stack, pis, M_comp, seqs, device):
    """GPU Bayesian filter over all components for a batch of sequences.

    seqs: (N, L) int64 (tensor or array).  Returns numpy arrays matching
    data.all_telescoped: telescoped (N,L,K*S), ntp (N,L,V), weights (N,L,K),
    local (N,L,K,S). Advances all sequences in parallel per position.
    """
    seqs = torch.as_tensor(seqs, dtype=torch.long, device=device)
    N, L, S, V = seqs.shape[0], cfg.seq_len, cfg.n_states, cfg.vocab
    K = len(cfg.comp_params)

    T, pi = _to_dev(T_stack, pis, cfg, device)           # (K,V,S,S), (K,S)
    M = torch.stack([torch.as_tensor(M_comp[k], dtype=torch.float32, device=device)
                     for k in range(K)])                 # (K, S, V)
    prior = torch.tensor(cfg.pi_prior, dtype=torch.float32, device=device)  # (K,)

    # unnormalized forward vector a[k] per sequence: (N, K, S)
    a = pi[None].expand(N, K, S) * prior[None, :, None]
    a = a.clone()

    tel = torch.empty(N, L, K*S, device=device)
    ntp = torch.empty(N, L, V, device=device)
    wts = torch.empty(N, L, K, device=device)
    loc = torch.empty(N, L, K, S, device=device)

    for t in range(L):
        x = seqs[:, t]                                   # (N,)
        # a[n,k,:] = a[n,k,:] @ T[k, x_n]   (per-seq token-specific transition)
        Tx = T[:, x]                                     # (K, N, S, S)
        Tx = Tx.permute(1, 0, 2, 3)                      # (N, K, S, S)
        a = torch.einsum('nks,nksd->nkd', a, Tx)         # (N, K, S)
        Z = a.reshape(N, -1).sum(-1, keepdim=True)       # (N, 1) global norm
        blk = a / Z[:, None]                             # (N, K, S) telescoped blocks
        w = blk.sum(-1)                                  # (N, K) weights
        eta = blk / w.clamp_min(1e-12)[..., None]        # (N, K, S) local beliefs
        # optimal NTP = sum_k blk[n,k] @ M[k]
        p = torch.einsum('nks,ksv->nv', blk, M)          # (N, V)
        p = p / p.sum(-1, keepdim=True)

        tel[:, t] = blk.reshape(N, K*S)
        ntp[:, t] = p
        wts[:, t] = w
        loc[:, t] = eta

    return (tel.cpu().numpy(), ntp.cpu().numpy(),
            wts.cpu().numpy(), loc.cpu().numpy())