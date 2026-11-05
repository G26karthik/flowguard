"""Paths of the PDEBench-derived training data, shared by training and evaluation scripts."""
RAW = "data/raw/2D_DarcyFlow_beta1.0_Train.hdf5"


def steady_targets_path(face, bc):
    """Steady unit-source solutions of all 10k fields for one discretisation (scripts/resolve_dataset.py, resolve_hf.py).
    Evaluation must compare a model with targets of its own discretisation."""
    base = "data/cache/2D_DarcyFlow_beta1.0_Train_steady"
    return f"{base}.npy" if (face, bc) == ("arith", "ghost") else f"{base}_{face}_{bc}.npy"
