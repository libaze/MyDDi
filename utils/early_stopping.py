import torch
import os

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience  # 允许的连续不提升的epoch数
        self.verbose = verbose    # 是否打印日志
        self.delta = delta        # 认为有提升的最小变化量
        self.counter = 0          # 计数器
        self.best_score = None    # 最佳分数
        self.early_stop = False   # 是否触发早停

    def __call__(self, val_metric, model, checkpoint_path):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, checkpoint_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, checkpoint_path)
            self.counter = 0

    def save_checkpoint(self, model, checkpoint_path):
        if self.verbose:
            print(f'Validation metric improved to {self.best_score:.6f}. Saving model...')
        torch.save(model.state_dict(), checkpoint_path)