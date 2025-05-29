import torch
from tqdm import tqdm
import numpy as np
from utils.logger import print_log
from utils.metrics import multi_class_metrics, binary_class_metrics, multi_label_metrics


def train_one_epoch(model, loss_func, optimizer, train_loader, fold_idx, epoch, label_type, device, log_config):
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


def evaluate(model, test_loader, fold_idx, epoch, label_type, device, log_config):
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


