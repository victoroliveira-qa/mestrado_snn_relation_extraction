import numpy as np
import tensorflow as tf


class PairSequence(tf.keras.utils.Sequence):
    """Balanced same/different relation pairs sampled within a single split."""
    def __init__(self, X, y, batch_size, steps, seed, positive_fraction=0.5):
        super().__init__()
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y)
        self.batch_size, self.steps = int(batch_size), int(steps)
        self.seed = int(seed)
        self.positive_fraction = float(positive_fraction)
        self.classes = np.unique(self.y)
        self.by_class = {c:np.flatnonzero(self.y==c) for c in self.classes}
        self.positive_classes = [c for c in self.classes if len(self.by_class[c])>=2]
        if len(self.classes)<2 or not self.positive_classes:
            raise ValueError("Pair training requires >=2 relation classes and >=2 samples in at least one class in this split")
        if not (0 < self.positive_fraction < 1):
            raise ValueError("positive_fraction must be between 0 and 1")
        self.n_positive = max(1, min(self.batch_size-1, round(self.batch_size*self.positive_fraction)))

    def __len__(self):
        return self.steps

    def __getitem__(self, batch_index):
        # Stateless batches ensure validation remains identical across epochs.
        rng = np.random.default_rng(self.seed + int(batch_index))
        left, right, targets = [], [], []
        for _ in range(self.n_positive):
            c = rng.choice(self.positive_classes)
            a,b = rng.choice(self.by_class[c], size=2, replace=False)
            left.append(self.X[a]);right.append(self.X[b]);targets.append(1.)
        for _ in range(self.batch_size-self.n_positive):
            a,b = rng.choice(self.classes,size=2,replace=False)
            left.append(self.X[rng.choice(self.by_class[a])]);right.append(self.X[rng.choice(self.by_class[b])]);targets.append(0.)
        order = rng.permutation(len(targets))
        return (np.asarray(left, dtype=np.float32)[order], np.asarray(right,dtype=np.float32)[order]), np.asarray(targets,dtype=np.float32)[order]
