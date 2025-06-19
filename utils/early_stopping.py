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

    def __call__(self, val_metric, checkpoint, checkpoint_path):
        '''
            :param val_metric:
            :param checkpoint:
                {
                    'model_state_dict': model.state_dict(),
                    'predictor_state_dict': predictor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_metric': best_val_auc,
                    'current_lr': optimizer.param_groups[0]['lr']
                }
            :param checkpoint_path:
            :return:
        '''

        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(checkpoint, checkpoint_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'  ⏳ EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("  🛑 Early stopping triggered")
        else:
            self.best_score = score
            self.save_checkpoint(checkpoint, checkpoint_path)
            self.counter = 0

    def save_checkpoint(self, save_dict, checkpoint_path):
        if self.verbose:
            print(f'  🎯 New best model saved (Validation metric: {self.best_score:.4f})')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(save_dict, checkpoint_path)