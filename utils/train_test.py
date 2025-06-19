import os

import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
from utils.logger import print_log
from utils.metrics import multi_class_metrics, binary_class_metrics, multi_label_metrics


def load_checkpoint(resume_path, model, predictor, optimizer, scheduler):
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        predictor.load_state_dict(checkpoint['predictor_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 从下一轮开始
        best_val_auc = checkpoint['best_metric']
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return start_epoch, best_val_auc
    return 0, 0.0  # 如果没有检查点，从头开始


def train_one_epoch(model, loss_func, optimizer, scheduler, train_loader, fold_idx, epoch, label_type, device):
    print("当前学习率:", optimizer.param_groups[0]['lr'])
    step = 0
    model.train()
    with tqdm(total=len(train_loader)) as pbar:
        for batch_idx, (drug1, drug2, labels, drug1_id, drug2_id) in enumerate(train_loader):
            drug1, drug2, labels = drug1.to(device), drug2.to(device), labels.to(device)
            if label_type == 'binary_class':
                y_pred = model(drug1, drug2, drug1_id, drug2_id).flatten()
                y_pred = torch.sigmoid(y_pred)
                loss = loss_func(y_pred, labels.float())
                train_metrics = binary_class_metrics(y_pred.cpu(), labels.cpu())
                print_log('train', fold_idx, epoch, step, loss, train_metrics, pbar)
            elif label_type == 'multi_class':
                y_pred = model(drug1, drug2, drug1_id, drug2_id)
                loss = loss_func(y_pred, labels)
                y_pred = y_pred.argmax(dim=-1).detach().cpu().numpy()
                train_metrics = multi_class_metrics(y_pred, labels.cpu(), num_classes=model.config['num_classes'])
                print_log('train', fold_idx, epoch, step, loss, train_metrics, pbar)
            elif label_type == 'multi_label':
                y_pred = model(drug1, drug2, drug1_id, drug2_id)
                y_pred = torch.sigmoid(y_pred)
                loss = loss_func(y_pred, labels)
                y_pred = torch.sigmoid(y_pred).detach().cpu().numpy()
                train_metrics = multi_label_metrics(y_pred, labels)
                print_log('train', fold_idx, epoch, step, loss, train_metrics, pbar)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            pbar.update(1)
    scheduler.step()


def mol_eval(model, test_loader, fold_idx, epoch, label_type, device):
    # 验证集评估
    model.eval()
    all_preds = []
    all_labels = []

    with tqdm(total=len(test_loader)) as pbar:
        with torch.no_grad():
            for batch_idx, (drug1, drug2, labels, drug1_id, drug2_id) in enumerate(test_loader):
                drug1, drug2, labels = drug1.to(device), drug2.to(device), labels.to(device)

                if label_type == 'binary_class':
                    y_pred = model(drug1, drug2, drug1_id, drug2_id).flatten()
                    y_pred = torch.sigmoid(y_pred)
                    all_preds.append(y_pred.cpu())
                    all_labels.append(labels.cpu())

                elif label_type == 'multi_class':
                    y_pred = model(drug1, drug2, drug1_id, drug2_id).flatten()
                    y_pred = y_pred.argmax(dim=-1)
                    all_preds.append(y_pred.cpu())
                    all_labels.append(labels.cpu())

                elif label_type == 'multi_label':
                    y_pred = model(drug1, drug2, drug1_id, drug2_id).flatten()
                    y_pred = torch.sigmoid(y_pred)
                    all_preds.append(y_pred.cpu())
                    all_labels.append(labels.cpu())

                pbar.update(1)

            # 计算评估指标
            if label_type == 'binary_class':
                y_pred = torch.cat(all_preds)
                y_true = torch.cat(all_labels)
                eval_metrics = binary_class_metrics(y_pred, y_true)

            elif label_type == 'multi_class':
                y_pred = torch.cat(all_preds).numpy()
                y_true = torch.cat(all_labels).numpy()
                eval_metrics = multi_class_metrics(y_pred, y_true, num_classes=model.config['num_classes'])

            elif label_type == 'multi_label':
                y_pred = torch.cat(all_preds).numpy()
                y_true = torch.cat(all_labels).numpy()
                eval_metrics = multi_label_metrics(y_pred, y_true)

            # 打印评估结果
            print_log('eval', fold_idx, epoch, None, None, eval_metrics, pbar)

            return eval_metrics



def pretrain_kg_eval(model, predictor, val_loader):
    model.eval()
    predictor.eval()

    val_scores = []
    val_labels = []

    with torch.no_grad():
        for input_nodes, positive_graph, negative_graph, blocks in tqdm(val_loader, desc='Validation'):
            # 1. 前向传播
            input_features = blocks[0].srcdata['features']
            h = model(blocks, input_features)

            # 2. 计算正负样本得分
            pos_score = predictor(positive_graph, h)
            neg_score = predictor(negative_graph, h)

            # 3. 异构图的所有关系类型的分数和标签
            for rel_type in pos_score.keys():
                val_scores.append(torch.sigmoid(pos_score[rel_type]).cpu())
                val_labels.append(torch.ones_like(pos_score[rel_type]).cpu())

                val_scores.append(torch.sigmoid(neg_score[rel_type]).cpu())
                val_labels.append(torch.zeros_like(neg_score[rel_type]).cpu())

    # 计算验证指标
    val_scores = np.concatenate(val_scores)
    val_labels = np.concatenate(val_labels)
    val_auc = roc_auc_score(val_labels, val_scores)
    return val_auc



