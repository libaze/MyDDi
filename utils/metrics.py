from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, average_precision_score
import numpy as np
from sklearn.preprocessing import label_binarize


def multi_class_metrics(pred, labels, average='macro', num_classes=None):
    """
    计算多分类问题的评估指标

    Args:
        pred: 模型预测值，可以是类别概率矩阵(n_samples, n_classes)或直接类别标签(n_samples)
        labels: 真实标签数组(n_samples)
        average: 多类指标的平均方法('macro', 'micro', 'weighted')
        num_classes: 可选参数，指定类别数量(默认从数据推断)

    Returns:
        包含各项指标(百分比0-100)的字典
    """
    # 处理预测值(概率矩阵或直接类别)
    if pred.ndim > 1:
        # 从概率矩阵获取预测类别(取每行最大值的索引)
        pred_labels = np.argmax(pred, axis=1)
        # 类别数取参数值或概率矩阵的列数
        num_classes = num_classes or pred.shape[1]
    else:
        # 已经是类别标签的情况
        pred_labels = pred
        # 类别数取参数值或标签中的唯一值数量
        num_classes = num_classes or len(np.unique(labels))

    # 将真实标签二值化(one-hot编码)
    y_true = label_binarize(labels, classes=range(num_classes))
    # 处理预测分数：概率矩阵直接使用，类别标签则二值化
    y_score = pred if pred.ndim > 1 else label_binarize(pred_labels, classes=range(num_classes))

    # 计算各项指标
    metrics = {
        'acc': accuracy_score(labels, pred_labels),  # 准确率
        'f1': f1_score(labels, pred_labels, average=average, zero_division=0),  # F1分数
        'precision': precision_score(labels, pred_labels, average=average, zero_division=0),  # 精确率
        'recall': recall_score(labels, pred_labels, average=average, zero_division=0),  # 召回率
        'auc': roc_auc_score(y_true, y_score, multi_class='ovr', average=average),  # ROC曲线下面积
        'aupr': average_precision_score(y_true, y_score, average=average)  # 平均精确率
    }

    # 将所有指标转换为百分比形式
    return {**{k: v * 100 for k, v in metrics.items()}}


def binary_class_metrics(pred, labels, threshold=0.5):
    """
    计算二分类问题的评估指标

    Args:
        pred: 模型预测值(概率或得分)
        labels: 真实标签数组
        threshold: 将概率转换为二分类预测的阈值(默认0.5)

    Returns:
        包含各项指标(百分比0-100)和阈值的字典
    """
    # 确保预测值是numpy数组(处理PyTorch张量情况)
    proba = pred.detach().numpy() if not isinstance(pred, np.ndarray) else pred

    # 根据阈值将概率转换为二分类预测(0或1)
    pred_labels = (proba >= threshold).astype(int)

    # 计算各项指标
    metrics = {
        'acc': accuracy_score(labels, pred_labels),  # 准确率
        'f1': f1_score(labels, pred_labels, zero_division=0),  # F1分数
        'precision': precision_score(labels, pred_labels, zero_division=0),  # 精确率
        'recall': recall_score(labels, pred_labels, zero_division=0),  # 召回率
        'auc': roc_auc_score(labels, proba),  # ROC曲线下面积
        'aupr': average_precision_score(labels, proba)  # 平均精确率
    }
    # 将指标转换为百分比并添加阈值
    return {**{k: v * 100 for k, v in metrics.items()}, 'threshold': threshold}


def multi_label_metrics(pred, labels, threshold=0.5, average='macro'):
    """
    计算多标签分类的多个评估指标

    Args:
        pred：预测概率数组 (shape: [n_samples, n_classes])
        labels：真标签数组 (shape: [n_samples, n_classes])
        threshold：将概率转换为二进制预测的阈值
        average：多类/多标签度量的平均方法('macro', 'micro', 'samples', 'weighted')

    Returns:
        包含各项指标(百分比0-100)和阈值的字典
    """
    # 根据阈值将概率矩阵转换为二分类预测(0或1)
    pred_labels = (pred >= threshold).astype(int)

    # 计算各项指标
    metrics = {
        'acc': accuracy_score(labels, pred_labels),  # 准确率
        'f1': f1_score(labels, pred_labels, average=average, zero_division=0),  # F1分数
        'precision': precision_score(labels, pred_labels, average=average, zero_division=0),  # 精确率
        'recall': recall_score(labels, pred_labels, average=average, zero_division=0),  # 召回率
        'auc': roc_auc_score(labels, pred, average=average),  # ROC曲线下面积
        'aupr': average_precision_score(labels, pred, average=average)  # 平均精确率
    }

    # 将指标转换为百分比并添加阈值
    return {**{k: v * 100 for k, v in metrics.items()},
            'threshold': threshold}