"""Shared in-context representation across vocabularies.

Tests whether an LLM forms a SHARED belief-state subspace for the same HMM when its
emissions are expressed in two different token vocabularies, interleaved in one context.

Design (state-continuity version): we sample ONE continuous emission-symbol sequence from
the HMM, so the belief trajectory is continuous and identical at every position. "Vocabulary"
is purely a rendering choice: within alternating blocks, the same emission symbols are
rendered with vocab-A letters or vocab-B letters. A model that recognizes the two streams as
the same underlying process should map both vocabs into one belief subspace -> a probe trained
on vocab-A positions decodes vocab-B positions (cross-vocab transfer).

Conditions:
  same      : one continuous HMM, alternating vocab-A / vocab-B blocks (primary).
  diff      : two DIFFERENT params (same family) interleaved, one per vocab (control 1 — the
              null: cross-vocab transfer should be low / track ground-truth belief similarity).
  separate  : same HMM in two SEPARATE contexts (one per vocab); probe trained on context A,
              tested on context B (control 2 — does sharing require co-presence in context?).

Confound handling: cross-vocab transfer is measured at every layer (token-identity differences
dominate only early layers); a transient window after each vocab switch is dropped.

Outputs: shared_vocab_{model}.csv
"""
import argparse, gc, sys, os
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS, REPRESENTATIVES
from src.hmm import stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs
from src.metrics.probes import fit_probe, predict_probe, compute_r2
from src.model_utils import load_model, tokens_to_prompt, match_positions, get_tok_ids, extract_activations_chunked

# Second vocabulary: disjoint single-letter tokens. VERIFY these tokenize to single tokens
# (one token per letter, space-prefixed) for your model — same check the main pipeline does.
VOCAB_B_POOL = ["K", "M", "W", "Z", "J"]


def _render_prompt(sym, vocab, names_a, names_b):
    """Render emission symbols to a space-separated letter string, choosing vocab per position."""
    letters = [(names_a if v == 0 else names_b)[s] for s, v in zip(sym, vocab)]
    return " ".join(letters)


def _block_vocab(n, block_len, rng=None):
    """Alternating vocab assignment 0/1 in blocks of block_len."""
    return (np.arange(n) // block_len) % 2


def _transient_mask(vocab_late, switch_cut):
    """False for positions within switch_cut tokens after a vocab switch."""
    keep = np.ones(len(vocab_late), dtype=bool)
    switches = np.where(np.diff(vocab_late) != 0)[0] + 1
    for sidx in switches:
        keep[sidx:sidx + switch_cut] = False
    return keep


def _probe_transfer(X, belief, vocab, idx_keep, layers_X, device, seed):
    """Fit per-vocab probes; return within/cross R2 per layer.
    X: dict layer->(n,d) np; belief:(n,m); vocab:(n,); idx_keep: bool mask."""
    rows = []
    n = len(belief)
    idx = np.arange(n)[idx_keep]
    vA = idx[vocab[idx] == 0]
    vB = idx[vocab[idx] == 1]
    if len(vA) < 20 or len(vB) < 20:
        return rows
    aTr, aTe = train_test_split(vA, train_size=0.5, random_state=seed)
    bTr, bTe = train_test_split(vB, train_size=0.5, random_state=seed)
    Yt = lambda ix: torch.tensor(belief[ix], device=device, dtype=torch.float32)
    for l in layers_X:
        Xl = X[l]
        Xt = lambda ix: torch.tensor(Xl[ix], device=device, dtype=torch.float32)
        # train on A
        Wa = fit_probe(Xt(aTr), Yt(aTr), use_bias=True)
        r2_AA = compute_r2(Yt(aTe), predict_probe(Xt(aTe), Wa, use_bias=True))   # within A
        r2_AB = compute_r2(Yt(bTe), predict_probe(Xt(bTe), Wa, use_bias=True))   # cross A->B
        # train on B
        Wb = fit_probe(Xt(bTr), Yt(bTr), use_bias=True)
        r2_BB = compute_r2(Yt(bTe), predict_probe(Xt(bTe), Wb, use_bias=True))   # within B
        r2_BA = compute_r2(Yt(aTe), predict_probe(Xt(aTe), Wb, use_bias=True))   # cross B->A
        for tv, ev, kind, r2 in [('A','A','within',r2_AA), ('A','B','cross',r2_AB),
                                 ('B','B','within',r2_BB), ('B','A','cross',r2_BA)]:
            rows.append({"layer": l, "train_vocab": tv, "eval_vocab": ev,
                         "kind": kind, "R2": float(r2)})
    return rows


def _run_context(wrapper, tokenizer, sym, vocab, names_a, names_b, tok_ids_all,
                 n_tok, probe_start, chunk_size, layers, device):
    """Render -> tokenize -> match -> extract activations at late positions.
    Returns (acts_late dict, vocab_late, n_late_start_index) with positions index>=probe_start."""
    prompt = _render_prompt(sym, vocab, names_a, names_b)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
    pos_indices, tok_at_pos = match_positions(input_ids, tok_ids_all)
    n_matched = min(len(sym), len(pos_indices))
    # recover emission symbol and vocab from the matched token index (tok_ids_all = A.. then B..)
    sym_rec = (tok_at_pos[:n_matched] % n_tok).astype(np.int64)
    voc_rec = (tok_at_pos[:n_matched] // n_tok).astype(np.int64)
    late = slice(probe_start, n_matched)
    late_pos = pos_indices[late]
    acts, _ = extract_activations_chunked(wrapper, input_ids, layers, late_pos, chunk_size, device)
    acts_np = {l: acts[l].cpu().numpy() for l in layers}
    del acts; gc.collect(); torch.cuda.empty_cache()
    return acts_np, voc_rec[late], n_matched


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--block_len", type=int, default=1000)
    P.add_argument("--switch_cut", type=int, default=50)     # drop transient after each switch
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=None)
    P.add_argument("--device", default="cuda")
    args = P.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    wrapper, tokenizer = load_model(args.model, device)
    layers = list(range(wrapper.n_layers))
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")

    all_rows = []
    families = args.families or list(HMMS.keys())
    for hmm_name in families:
        cfg = HMMS.get(hmm_name)
        if not cfg: continue
        n_tok = cfg["n_tokens"]
        names_a = cfg["token_names"]
        names_b = VOCAB_B_POOL[:n_tok]
        tok_ids_all = get_tok_ids(tokenizer, list(names_a) + list(names_b))  # A symbols 0..n-1, B symbols n..2n-1
        rep = REPRESENTATIVES.get(hmm_name)
        if rep is None: continue
        label = cfg["label_fn"](rep)
        T = cfg["fn"](*rep); T_stack = np.stack(T); pi = stationary_distribution(T)
        # a DIFFERENT param of the same family for the 'diff' control (same belief dim)
        others = [p for p in cfg["params"] if cfg["label_fn"](p) != label]
        rep2 = others[0] if others else rep
        label2 = cfg["label_fn"](rep2)
        T2 = cfg["fn"](*rep2); T2_stack = np.stack(T2); pi2 = stationary_distribution(T2)

        print(f"\n===== {hmm_name}  (A={names_a}, B={names_b}; same={label}, diff={label2}) =====")
        pbar = tqdm(total=args.n_seeds * 3, desc=hmm_name)
        for seed in range(args.n_seeds):
            vocab = _block_vocab(args.seq_len, args.block_len)

            # ---------- SAME: one continuous process, alternating vocab ----------
            tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)
            beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)
            acts, voc_late, nm = _run_context(wrapper, tokenizer, tokens, vocab, names_a, names_b,
                                              tok_ids_all, n_tok, args.probe_start, args.chunk_size, layers, device)
            bel_late = beliefs[args.probe_start:nm]
            keep = _transient_mask(voc_late, args.switch_cut)
            for r in _probe_transfer(acts, bel_late, voc_late, keep, layers, device, seed):
                r.update({"hmm": hmm_name, "condition": "same", "param": label, "seed": seed})
                all_rows.append(r)
            del acts; gc.collect(); torch.cuda.empty_cache(); pbar.update(1)

            # ---------- DIFF: two different params, one per vocab (control 1) ----------
            tok1 = sample_hmm_sequence(T,  pi,  args.seq_len, seed=seed)
            tok2 = sample_hmm_sequence(T2, pi2, args.seq_len, seed=seed + 50000)
            bel1 = full_bayesian_beliefs(tok1.astype(np.int64), T_stack, pi)
            bel2 = full_bayesian_beliefs(tok2.astype(np.int64), T2_stack, pi2)
            # interleave consecutive chunks: A blocks from stream1, B blocks from stream2
            sym_d = np.empty(args.seq_len, dtype=np.int64)
            bel_d = np.empty((args.seq_len, len(pi)))
            p1 = p2 = 0
            for start in range(0, args.seq_len, args.block_len):
                end = min(start + args.block_len, args.seq_len)
                L = end - start
                if vocab[start] == 0:
                    sym_d[start:end] = tok1[p1:p1+L]; bel_d[start:end] = bel1[p1:p1+L]; p1 += L
                else:
                    sym_d[start:end] = tok2[p2:p2+L]; bel_d[start:end] = bel2[p2:p2+L]; p2 += L
            acts, voc_late, nm = _run_context(wrapper, tokenizer, sym_d, vocab, names_a, names_b,
                                              tok_ids_all, n_tok, args.probe_start, args.chunk_size, layers, device)
            bel_late = bel_d[args.probe_start:nm]
            keep = _transient_mask(voc_late, args.switch_cut)
            for r in _probe_transfer(acts, bel_late, voc_late, keep, layers, device, seed):
                r.update({"hmm": hmm_name, "condition": "diff", "param": f"{label}|{label2}", "seed": seed})
                all_rows.append(r)
            del acts; gc.collect(); torch.cuda.empty_cache(); pbar.update(1)

            # ---------- SEPARATE: same process, two separate single-vocab contexts (control 2) ----------
            # same emission tokens rendered fully in vocab A (context A) and fully in vocab B (context B)
            tokens_s = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)
            beliefs_s = full_bayesian_beliefs(tokens_s.astype(np.int64), T_stack, pi)
            zerosA = np.zeros(args.seq_len, dtype=np.int64)   # all vocab A
            onesB  = np.ones(args.seq_len, dtype=np.int64)    # all vocab B
            actsA, _, nmA = _run_context(wrapper, tokenizer, tokens_s, zerosA, names_a, names_b,
                                         tok_ids_all, n_tok, args.probe_start, args.chunk_size, layers, device)
            actsB, _, nmB = _run_context(wrapper, tokenizer, tokens_s, onesB, names_a, names_b,
                                         tok_ids_all, n_tok, args.probe_start, args.chunk_size, layers, device)
            nshared = min(nmA, nmB)
            belA = beliefs_s[args.probe_start:nmA]; belB = beliefs_s[args.probe_start:nmB]
            for l in layers:
                XA = actsA[l]; XB = actsB[l]
                nA, nB = len(XA), len(XB)
                aTr, aTe = train_test_split(np.arange(nA), train_size=0.5, random_state=seed)
                bTr, bTe = train_test_split(np.arange(nB), train_size=0.5, random_state=seed)
                Yt = lambda Y, ix: torch.tensor(Y[ix], device=device, dtype=torch.float32)
                Xt = lambda X, ix: torch.tensor(X[ix], device=device, dtype=torch.float32)
                Wa = fit_probe(Xt(XA, aTr), Yt(belA, aTr), use_bias=True)
                r2_AA = compute_r2(Yt(belA, aTe), predict_probe(Xt(XA, aTe), Wa, use_bias=True))
                r2_AB = compute_r2(Yt(belB, bTe), predict_probe(Xt(XB, bTe), Wa, use_bias=True))
                for tv, ev, kind, r2 in [('A','A','within',r2_AA), ('A','B','cross',r2_AB)]:
                    all_rows.append({"hmm": hmm_name, "condition": "separate", "param": label,
                                     "seed": seed, "layer": l, "train_vocab": tv,
                                     "eval_vocab": ev, "kind": kind, "R2": float(r2)})
            del actsA, actsB; gc.collect(); torch.cuda.empty_cache(); pbar.update(1)
        pbar.close()
        pd.DataFrame(all_rows).to_csv(os.path.join(args.output_dir, f"shared_vocab_{ms}.csv"), index=False)
    print(f"Done. {len(all_rows)} rows.")


if __name__ == "__main__":
    main()
