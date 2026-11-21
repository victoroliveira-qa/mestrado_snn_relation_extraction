import warnings
import numpy as np
from sklearn.model_selection import train_test_split


def split_ids(y, cfg, seed):
    """Return row IDs. Split original rows before generating pairs (no pair leakage)."""
    ids = np.arange(len(y))
    fractions = [float(cfg[k]) for k in ("train", "validation", "test")]
    if abs(sum(fractions) - 1) > 1e-7 or min(fractions) <= 0:
        raise ValueError("Train/validation/test fractions must be positive and sum to one")
    stratify_mode = cfg.get("stratify", "auto")
    if stratify_mode not in ("auto", "strict", "none"):
        raise ValueError("stratify must be auto, strict, or none")
    def do_split(indices, test_size, stage_seed, candidate_labels):
        s = candidate_labels if stratify_mode != "none" else None
        try:
            return train_test_split(indices, test_size=test_size, random_state=stage_seed, stratify=s)
        except ValueError as exc:
            if stratify_mode == "strict":
                raise ValueError("Stratified split failed. Check class support and requested fractions") from exc
            if stratify_mode == "auto":
                warnings.warn(f"Stratification unavailable; using random split: {exc}")
            return train_test_split(indices, test_size=test_size, random_state=stage_seed)
    train, remaining = do_split(ids, 1-fractions[0], seed, y)
    relative_test = fractions[2] / (fractions[1]+fractions[2])
    val, test = do_split(remaining, relative_test, seed, y[remaining])
    if len(set(train)&set(val)) or len(set(train)&set(test)) or len(set(val)&set(test)):
        raise AssertionError("Split overlap")
    return {"train":train,"validation":val,"test":test}
