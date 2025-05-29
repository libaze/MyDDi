import os.path

import torch
from torch import nn

from utils.early_stopping import EarlyStopping
from utils.k_fold import KFold
from torchvision import transforms
from torch.utils.data import DataLoader
from model.ddi_model import DDINet
from data.dataset import load_data, DrugDataset
from utils.optimizer import get_optimizer
from utils.train_test import train_one_epoch, evaluate
from data_process.generate_graph import gen_hetero_graph


def finetune(config):
    # 加载数据
    ddis, drkg, drkgID2ImagePath, entities, drkg_relations = load_data(config['dataset']['path'], config['dataset']['type'])
    print(f"Loaded {len(drkgID2ImagePath)} drug images mapping")
    log_config = config['logging']
    # 初始化 KFold
    kf = KFold(
        n_splits=config['training']['fold_k']['n_splits'],
        shuffle=config['training']['fold_k']['shuffle'],
        val_ratio=config['training']['fold_k']['val_ratio'],
    )

    basic_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    device = torch.device(config['training']['device'])

    for idx, (train_indices, val_indices, test_indices) in enumerate(kf.split(ddis)):
        # 数据集
        label_type = config['dataset']['type']
        train_dataset = DrugDataset(dataset_config=config['dataset'], data=ddis[train_indices],
                                    id2path=drkgID2ImagePath, transforms=basic_transforms)
        val_dataset = DrugDataset(dataset_config=config['dataset'], data=ddis[val_indices], id2path=drkgID2ImagePath,
                                  transforms=basic_transforms)
        test_dataset = DrugDataset(dataset_config=config['dataset'], data=ddis[test_indices], id2path=drkgID2ImagePath,
                                   transforms=basic_transforms)

        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True,
                                  num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False,
                                 num_workers=8)

        # 模型
        if config['model']['type'] != 'only_mol':
            hetero_graph, ID2H_ID = gen_hetero_graph(drkg, drkg_relations, entities)
            model = DDINet(config['model'], hetero_graph.to(device), ID2H_ID)
        else:
            model = DDINet(config['model'])
        model.to(device)
        # 损失函数
        if label_type == 'multi_class':
            loss_func = nn.CrossEntropyLoss()
        elif label_type == 'binary_class':
            loss_func = nn.BCEWithLogitsLoss()
        elif label_type == 'multi_label':
            loss_func = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported type: {label_type}")
        # 优化器
        optimizer_config = config['training']['optimizer']
        optimizer = get_optimizer(optimizer_config['type'])(
            model.parameters(),
            lr=optimizer_config.get('lr', 1e-3),
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )
        early_stopping = EarlyStopping(patience=config['training']['early_stopping'], verbose=True)
        checkpoint_folder_path = os.path.join(config['training']['checkpoint'], config['dataset']['name'], label_type, f'fold_{idx + 1}')
        if not os.path.exists(checkpoint_folder_path):
            os.makedirs(checkpoint_folder_path, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_folder_path, "best_model.pth")
        # 训练
        for epoch in range(config['training']['epochs']):
            # 训练一个epoch
            train_one_epoch(model, loss_func, optimizer, train_loader, idx + 1, epoch, label_type, device, log_config)

            # 在验证集上评估
            val_metrics = evaluate(model, val_loader, idx + 1, epoch, label_type, device, log_config)

            # 根据验证集性能保存最佳模型
            if label_type == 'binary_class':
                key_metric = val_metrics['f1']
            elif label_type == 'multi_class':
                key_metric = val_metrics['f1']
            elif label_type == 'multi_label':
                key_metric = val_metrics['f1']

            # 使用早停
            early_stopping(key_metric, model, checkpoint_path)

            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # 训练结束后加载最佳模型
        model.load_state_dict(torch.load(checkpoint_path))

        # 在测试集上最终评估
        test_metrics = evaluate(model, test_loader, idx + 1, epoch, label_type, device, log_config)

