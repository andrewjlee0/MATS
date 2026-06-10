# MATS / belief-geometry repo

Andrew's MATS work with Adam Shai & Paul Riechers (Simplex). This repo also hosts the SPAR NeurIPS belief-geometry paper code. The global scientist-operating-standard (`~/.claude/CLAUDE.md`) applies here in full — first principles, confounds, controls, intellectual honesty, no silent shortcuts.

## What this repo is
- **MATS thread:** belief-state geometry of small transformers trained on **non-ergodic mixtures of Mess3 HMMs**. Key object is the **coupled-IFS "telescope"** geometry: tuple of unnormalized per-component belief vectors `(w_1·η_1, …, w_C·η_C)` (direction = belief within a component, magnitude = component likelihood). Currently: train a tiny (1-layer/1-head/1-MLP) transformer, decode the telescoped belief, test whether the emission matrix is recoverable / causally used.
- **SPAR paper thread:** "LLMs Develop Belief State Geometry In-Context" (NeurIPS 2026, submitted). Probing pretrained LLMs (main model **Qwen3.5-9B**) fed in-context HMM sequences (families **Mess3, Arch, Wing, Strata**) for linearly-decodable belief geometry.

## Layout & conventions
- `configs/` (e.g. `nonergodic_config.py` `NonergodicConfig`, `hmm_configs.py`), `src/` (`hmm.py`, `metrics/probes.py`, `nonergodic/` subpackage: `model.py`/`data.py`/`data_gpu.py`/`analysis.py`), `experiments/<question>/` train+run scripts, `plots/` (all plotting incl. `analyze.ipynb`).
- Experiment subfolders import via `sys.path.insert(0, "../..")`.
- **Train/analyze split with exhaustive saves**: `train.py` saves everything to `run/` (`.npz`/`.pt`/`.json`); analysis/plots read from disk. Save per-layer AND per-seed rows (never pre-average away the replicate dimension — CIs need it).
- **Reuse existing code** (`src/metrics/probes.py`) rather than reimplementing.
- GPU-maximized (whole dataset on GPU, bf16, TF32, fused Adam). RunPod, venv (not conda), `--break-system-packages`. Git via `gh auth`/token.

## Hard rules for experiments here
- Use the **full** data/sequence length/seed count he specifies; don't quietly shrink anything.
- **Keep all layers** (`range(n_layers)`), not subsets.
- Probe OLS **with a bias term** (a real past bug was the bias-less pinv solution).
- Watch the known footguns: leading-space tokenization (align KL by position), full-vocab KL fp32 OOM on big models, Gemma logit softcapping for tuned-lens correctness, float underflow in long belief filters (renormalize forward vector, use float64).
- R² is **not comparable as a level across differently-encoded predictors** — lean on an independent selector/modulator, not a raw level.

## Plotting
Use the `figure` skill (seaborn-only, interactive HTML, 95% CI, consult on the averaging unit). Don't deviate from his figure house style.
