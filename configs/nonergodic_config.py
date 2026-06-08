"""Config for the non-ergodic Mess3 mixture experiment (sits alongside hmm_configs.py)."""


class NonergodicConfig:
    # two VERY different Mess3 components
    comp_params = [(0.05, 0.05), (0.90, 0.15)]   # (alpha, x) per component
    pi_prior    = [0.5, 0.5]                      # generative prior over components
    vocab       = 3
    n_states    = 3
    seq_len     = 16
    n_train     = 40000
    n_val       = 4000
    # model
    d_model     = 64
    d_mlp       = 256
    n_heads     = 1
    # train
    epochs      = 25
    batch_size  = 256
    lr          = 1e-3
    seed        = 0

    @classmethod
    def to_dict(cls):
        return {k: getattr(cls, k) for k in dir(cls)
                if not k.startswith("_") and k != "to_dict"}
