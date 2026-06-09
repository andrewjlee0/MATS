"""HMM construction, dataset sampling, and ground-truth telescoped beliefs
for the non-ergodic Mess3 mixture."""
import numpy as np

from src.hmm.definitions import mess3_matrices
from src.hmm.core import stationary_distribution, sample_hmm_sequence, emission_matrix


def build_hmms(cfg):
    """Per-component transition stacks, stationary dists, emission matrices,
    and the block-diagonal emission [M_1 ; ... ; M_K]."""
    T_mats = [mess3_matrices(a, x) for (a, x) in cfg.comp_params]
    T_stack = [np.stack(Tm) for Tm in T_mats]
    pis = [stationary_distribution(Tm) for Tm in T_mats]
    M_comp = [emission_matrix(Tm) for Tm in T_mats]
    M_block = np.vstack(M_comp)
    return T_mats, T_stack, pis, M_comp, M_block


def make_dataset(cfg, T_mats, pis, n_seqs, seed0):
    """Each sequence is generated entirely by one component, chosen by pi_prior."""
    rng = np.random.default_rng(seed0)
    seqs = np.zeros((n_seqs, cfg.seq_len), dtype=np.int64)
    comps = np.zeros(n_seqs, dtype=np.int64)
    for i in range(n_seqs):
        c = rng.choice(len(cfg.comp_params), p=cfg.pi_prior)
        comps[i] = c
        seqs[i] = sample_hmm_sequence(T_mats[c], pis[c], cfg.seq_len, seed=seed0 + i + 1)
    return seqs, comps


def telescoped_beliefs(cfg, T_stack, pis, M_comp, tokens):
    """For one token sequence, per position: telescoped blocks (L, K*S),
    optimal NTP (L, vocab), posterior weights (L, K), local beliefs (L, K, S)."""
    K, S, L = len(cfg.comp_params), cfg.n_states, cfg.seq_len
    a = [pis[n].copy() * cfg.pi_prior[n] for n in range(K)]
    tel = np.zeros((L, K * S)); ntp = np.zeros((L, cfg.vocab))
    weights = np.zeros((L, K)); local = np.zeros((L, K, S))
    for t in range(L):
        for n in range(K):
            a[n] = a[n] @ T_stack[n][tokens[t]]
        Z = sum(a[n].sum() for n in range(K))
        p = np.zeros(cfg.vocab)
        for n in range(K):
            blk = a[n] / Z
            tel[t, n*S:(n+1)*S] = blk
            w = blk.sum()
            weights[t, n] = w
            local[t, n] = blk / w if w > 1e-12 else blk
            p += blk @ M_comp[n]
        ntp[t] = p / p.sum()
    return tel, ntp, weights, local


def all_telescoped(cfg, T_stack, pis, M_comp, seqs):
    K, S, L = len(cfg.comp_params), cfg.n_states, cfg.seq_len
    N = len(seqs)
    tel = np.zeros((N, L, K*S)); ntp = np.zeros((N, L, cfg.vocab))
    wts = np.zeros((N, L, K)); loc = np.zeros((N, L, K, S))
    for i, s in enumerate(seqs):
        tel[i], ntp[i], wts[i], loc[i] = telescoped_beliefs(cfg, T_stack, pis, M_comp, s)
    return tel, ntp, wts, loc