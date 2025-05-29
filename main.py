import argparse
import os
import random
import numpy as np
import torch
import yaml
from finetune import finetune


def set_all_seeds(seed=42):
    print('seed:', seed)
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch（CPU/GPU）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    # CUDA优化设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Drug Interaction Prediction')
    parser.add_argument('-m', '--mode', type=str, default='finetune')
    parser.add_argument('-c', '--config', type=str, default='./config/finetune/train.yaml',
                        help='path to configuration file')
    args = parser.parse_args(args=[])
    config = load_config(args.config)
    print(config)
    set_all_seeds(seed=config['training']['seed'])
    if args.mode == 'finetune':
        finetune(config)
    else:
        pass



if __name__ == '__main__':
    main()