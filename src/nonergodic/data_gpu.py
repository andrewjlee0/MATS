"""GPU-vectorized HMM sampling and Bayesian filtering for the non-ergodic
Mess3 experiment. All sequences advanced in parallel each position.

The filter renormalizes the forward vector every step (and runs in float64)
so it cannot underflow to zero/nan over long sequences.
"""
import numpy as np
import torch


def _to_dev(T_stack, pis, cfg, device, dtype=torch.float32):
    K = len(cfg.comp_params)
    T = torch.stack([torch.as_tensor(T_stack[k], dtype=dtype, device=device)
                     for k in range(K)])
    pi = torch.stack([torch.as_tensor(pis[k], dtype=dtype, device=device)
                      for k in range(K)])
    return T, pi


def sample_sequences_gpu(cfg, T_stack, pis, comps, device, seed=0):
    """Vectorized HMM sampling on the GPU. comps: (N,) int component indices.
    Returns (N, seq_len) int64 token tensor on `device`."""
    g = torch.Generator(device=device).manual_seed(seed)
    comps = torch.as_tensor(comps, dtype=torch.long, device=device)
    N, L, S, V = len(comps), cfg.seq_len, cfg.n_states, cfg.vocab
    T, pi = _to_dev(T_stack, pis, cfg, device)
    Tc = T[comps]                                   # (N, V, S, S)
    ar = torch.arange(N, device=device)
    state = torch.multinomial(pi[comps], 1, generator=g).squeeze(1)
    tokens = torch.empty(N, L, dtype=torch.long, device=device)
    for t in range(L):
        row = Tc[ar, :, state, :]                   # (N, V, S)
        tok_p = row.sum(-1)                         # (N, V)
        tok_p = tok_p / tok_p.sum(-1, keepdim=True)
        z = torch.multinomial(tok_p, 1, generator=g).squeeze(1)
        tokens[:, t] = z
        nsp = Tc[ar, z, state, :]                   # (N, S)
        nsp = nsp / nsp.sum(-1, keepdim=True)
        state = torch.multinomial(nsp, 1, generator=g).squeeze(1)
    return tokens


def all_telescoped_gpu(cfg, T_stack, pis, M_comp, seqs, device):
    """GPU Bayesian filter over all components. Returns numpy arrays matching
    data.all_telescoped: telescoped (N,L,K*S), ntp (N,L,V), weights (N,L,K),
    local (N,L,K,S). float64 + per-step renormalization to avoid underflow."""
    dt = torch.float64
    seqs = torch.as_tensor(seqs, dtype=torch.long, device=device)
    N, L, S, V = seqs.shape[0], cfg.seq_len, cfg.n_states, cfg.vocab
    K = len(cfg.comp_params)

    T, pi = _to_dev(T_stack, pis, cfg, device, dtype=dt)
    M = torch.stack([torch.as_tensor(M_comp[k], dtype=dt, device=device)
                     for k in range(K)])            # (K, S, V)
    prior = torch.tensor(cfg.pi_prior, dtype=dt, device=device)

    a = (pi[None].expand(N, K, S) * prior[None, :, None]).clone()   # (N, K, S)

    tel = torch.empty(N, L, K*S, dtype=dt, device=device)
    ntp = torch.empty(N, L, V, dtype=dt, device=device)
    wts = torch.empty(N, L, K, dtype=dt, device=device)
    loc = torch.empty(N, L, K, S, dtype=dt, device=device)

    for t in range(L):
        x = seqs[:, t]
        Tx = T[:, x].permute(1, 0, 2, 3)            # (N, K, S, S)
        a = torch.einsum('nks,nksd->nkd', a, Tx)    # (N, K, S)
        Z = a.reshape(N, -1).sum(-1, keepdim=True).clamp_min(1e-300)
        a = a / Z[:, None]                          # carry NORMALIZED vector forward
        blk = a                                     # globally normalized (sum=1)
        w = blk.sum(-1)
        eta = blk / w.clamp_min(1e-30)[..., None]
        p = torch.einsum('nks,ksv->nv', blk, M)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-300)
        tel[:, t] = blk.reshape(N, K*S)
        ntp[:, t] = p
        wts[:, t] = w
        loc[:, t] = eta

    return (tel.float().cpu().numpy(), ntp.float().cpu().numpy(),
            wts.float().cpu().numpy(), loc.float().cpu().numpy())