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


def split_official_ids(y, source_splits, cfg, seed):
    """Preserve official train/test partitions and sample validation within train.

    train and test labels may overlap; no test sample is used for training,
    early stopping or prototype creation. Class support is checked by main.
    """
    source_splits = np.asarray(source_splits).astype(str)
    y = np.asarray(y)
    if len(y) != len(source_splits):
        raise ValueError('Label vector and official-split vector have different lengths')
    invalid = sorted(set(source_splits) - {'train', 'test'})
    if invalid:
        raise ValueError(f'Unexpected official split labels: {invalid}')
    train_pool = np.where(source_splits == 'train')[0]
    test = np.where(source_splits == 'test')[0]
    if len(train_pool) < 3 or len(test) == 0:
        raise ValueError('Official CORE split requires train and test instances')
    # Validation fraction applies ONLY to the official train partition.
    validation_fraction = float(cfg.get('validation_fraction', .10))
    if not 0 < validation_fraction < 1:
        raise ValueError('split.validation_fraction must be between 0 and 1')
    mode = cfg.get('stratify', 'auto')
    if mode not in ('auto', 'strict', 'none'):
        raise ValueError('stratify must be auto, strict, or none')
    try:
        train, val = train_test_split(train_pool, test_size=validation_fraction,
                                      random_state=seed, stratify=y[train_pool] if mode != 'none' else None)
    except ValueError as exc:
        if mode == 'strict':
            raise ValueError('CORE official-train validation stratification failed') from exc
        if mode == 'auto':
            warnings.warn(f'CORE validation stratification unavailable; using random validation: {exc}')
        train, val = train_test_split(train_pool, test_size=validation_fraction, random_state=seed)
    return {'train':np.asarray(train), 'validation':np.asarray(val), 'test':np.asarray(test)}
