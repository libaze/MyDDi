import numpy as np
import random


class KFold:
    def __init__(self, n_splits=5, shuffle=True, val_ratio=0.2):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.val_ratio = val_ratio

    def split(self, data):
        n_samples = len(data)
        indices = np.arange(n_samples)

        if self.shuffle:
            random.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]

            # 剩余部分作为训练集和验证集的候选
            remaining_indices = np.concatenate([indices[:start], indices[stop:]])

            # 计算验证集大小
            n_val = int(len(remaining_indices) * self.val_ratio)

            # 划分训练集和验证集
            if self.shuffle:
                np.random.shuffle(remaining_indices)
            val_indices = remaining_indices[:n_val]
            train_indices = remaining_indices[n_val:]

            yield train_indices, val_indices, test_indices
            current = stop