import os.path

import numpy as np
import torch
import dgl
from dgl.dataloading import NeighborSampler, DataLoader
from dgl.dataloading.negative_sampler import GlobalUniform
from tqdm import tqdm
from data_process.dataset import DRKGDGLDataset
import torch.nn.functional as F
from model.kg_model import DRKGModel
from model.predictor import ScorePredictor
from utils.early_stopping import EarlyStopping
from utils.optimizer import get_optimizer, get_scheduler
from utils.train_test import pretrain_kg_eval


def pretrain_kg(config):
    device = torch.device(config['training']['device'])
    dataset = DRKGDGLDataset(config['dataset']['path'])

    g = dataset[0]
    g = g.to(device)

    rel_names = g.etypes
    node_names = g.ntypes
    print('rel_names: ', rel_names)
    print('node_names: ', node_names)

    # 划分训练/验证集
    train_retio = 1 - config["training"]["val_ratio"]
    train_eids = {etype: g.edges(etype=etype, form='eid')[:int(train_retio * len(g.edges(etype=etype, form='eid')))]
                  for etype in g.etypes}
    val_eids = {etype: g.edges(etype=etype, form='eid')[int(train_retio * len(g.edges(etype=etype, form='eid'))): int(
        train_retio * len(g.edges(etype=etype, form='eid'))) + 1000]
                for etype in g.etypes}

    # 采样器
    sampler = NeighborSampler([config["training"]["batch_size"]] * config["model"]['kg_model']['num_layers'])
    train_sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler,
        negative_sampler=GlobalUniform(config["training"]['negative_sample_ratio']))
    val_sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler,
        negative_sampler=GlobalUniform(config["training"]['negative_sample_ratio']))

    # 数据加载器
    train_dataloader = DataLoader(
        g, train_eids, train_sampler,
        device=device, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=False,
        num_workers=config["training"]["num_workers"]
    )
    val_dataloader = DataLoader(
        g, val_eids, val_sampler,
        device=device, batch_size=config["training"]["batch_size"], shuffle=False, drop_last=False,
        num_workers=config["training"]["num_workers"]
    )

    model = DRKGModel(**config["model"]['kg_model'], rel_names=rel_names).to(device)
    predictor = ScorePredictor().to(device)
    # 优化器
    optimizer_config = config["training"]["optimizer"]
    optimizer = get_optimizer(optimizer_config["type"])(
        model.parameters(),
        lr=optimizer_config.get("lr", 1e-3),
        weight_decay=optimizer_config.get("weight_decay", 0.0),
    )

    scheduler_config = config["training"]["scheduler"]
    scheduler = get_scheduler(optimizer, scheduler_config)

    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping"], verbose=True
    )

    # 初始化最佳指标
    best_val_auc = 0

    for epoch in range(1, config['training']['epochs'] + 1):
        # ==================== 训练阶段 ====================
        model.train()
        predictor.train()

        epoch_loss = []

        # 训练进度条
        train_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}/{config["training"]["epochs"]}')

        for step, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(train_bar):
            # 1. 前向传播
            input_features = blocks[0].srcdata['features']
            h = model(blocks, input_features)

            # 2. 计算正负样本得分
            pos_score = predictor(positive_graph, h)
            neg_score = predictor(negative_graph, h)

            # 3. 异构图的损失计算
            loss = 0
            num_rels = 0

            for rel_type in pos_score.keys():
                if pos_score[rel_type].numel() == 0 or neg_score[rel_type].numel() == 0:
                    continue
                # 为每种关系类型计算损失
                pos_labels = torch.ones_like(pos_score[rel_type])
                neg_labels = torch.zeros_like(neg_score[rel_type])

                rel_loss = F.binary_cross_entropy_with_logits(
                    torch.cat([pos_score[rel_type], neg_score[rel_type]]),
                    torch.cat([pos_labels, neg_labels])
                )
                loss += rel_loss
                num_rels += 1

            loss = loss / num_rels  # 平均所有关系类型的损失

            # 4. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录指标
            epoch_loss.append(loss.item())

            # 更新进度条信息
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            # ==================== 验证阶段 ====================
            if (step + 1) % config['training']['eval_every'] == 0:
                val_auc = pretrain_kg_eval(model, predictor, val_dataloader)
                model.train()
                scheduler.step()
                # ==================== 日志记录 ====================
                print(f'\nEpoch {epoch}:')
                print(f'  Val AUC: {val_auc:.4f}')

                # ==================== 模型保存与早停 ====================
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'predictor_state_dict': predictor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_metric': best_val_auc,
                    'current_lr': optimizer.param_groups[0]['lr']
                }
                early_stopping(val_auc, checkpoint, os.path.join(config['training']['checkpoint'], 'pretrain', 'pretrain_kg_model.pth'))

                if early_stopping.early_stop:
                    break


        # 计算训练指标
        train_loss = np.mean(epoch_loss)

        print(f'\nEpoch {epoch}: Train Mean Loss: {train_loss:.4f}')

    # 训练结束提示
    print(f'\nTraining completed. Best validation AUC: {best_val_auc:.4f}')
