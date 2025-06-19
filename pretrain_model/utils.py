import os
import pickle
import dgl
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_link_prediction_graph(graph, test_ratio=0.3, ):
    """
    通用异构图形链接预测数据划分
    参数：
        graph: DGL异构图
        test_ratio: 测试集比例
        val_ratio: 验证集比例
    返回：
        train_graph, val_graph, test_graph,
        neg_samples_dict (包含各关系类型的负样本)
    """
    # 若图数据存在，则加载图数据
    if os.path.exists('pretrain_graph_data.dgl') and os.path.exists('neg_samples.pkl'):
        loaded_graphs, label_dict = dgl.load_graphs('pretrain_graph_data.dgl')
        train_graph = loaded_graphs[0]
        test_graph = loaded_graphs[1]
        with open('neg_samples.pkl', 'rb') as f:
            neg_samples_dict = pickle.load(f)
        return train_graph, test_graph, neg_samples_dict

    # 1. 为每个关系类型生成划分
    neg_samples_dict = {}
    train_edges = {}
    test_edges = {}

    for idx, canonical_etype in enumerate(graph.canonical_etypes):
        print(f'[{idx + 1}/{len(graph.canonical_etypes)}]\tCanonical etype:', canonical_etype)
        src_type, rel_type, dst_type = canonical_etype
        src, dst = graph.edges(etype=canonical_etype)
        edges = torch.stack([src, dst], dim=1)

        # 生成负样本（确保不存在于图中）
        num_neg = len(edges)
        neg_src = torch.randint(0, graph.num_nodes(src_type), (num_neg,))
        neg_dst = torch.randint(0, graph.num_nodes(dst_type), (num_neg,))

        # 过滤掉真实存在的边
        for i in tqdm(range(num_neg), desc=f'{str(canonical_etype)}负采样...'):
            while graph.has_edges_between(neg_src[i], neg_dst[i], etype=canonical_etype):
                neg_src[i] = torch.randint(0, graph.num_nodes(src_type), (1,))
                neg_dst[i] = torch.randint(0, graph.num_nodes(dst_type), (1,))
        neg_samples = torch.stack([neg_src, neg_dst], dim=1)

        # 划分正样本
        train_pos, test_pos = train_test_split(
            edges, test_size=test_ratio, random_state=42
        )

        # 划分负样本（相同比例）
        train_neg, test_neg = train_test_split(
            neg_samples, test_size=test_ratio, random_state=42
        )

        # 存储划分结果
        train_edges[canonical_etype] = (train_pos[:, 0], train_pos[:, 1])
        test_edges[canonical_etype] = (test_pos[:, 0], test_pos[:, 1])
        neg_samples_dict[canonical_etype] = {
            'train': train_neg,
            'test': test_neg
        }

    # 2. 构建子图
    train_graph = dgl.heterograph(train_edges)
    test_graph = dgl.heterograph(test_edges)

    # 3. 保留原始特征
    for ntype in graph.ntypes:
        for key in graph.nodes[ntype].data:
            train_graph.nodes[ntype].data[key] = graph.nodes[ntype].data[key]
            test_graph.nodes[ntype].data[key] = graph.nodes[ntype].data[key]

    # 保存图数据
    dgl.save_graphs('pretrain_graph_data.dgl', [train_graph, test_graph])
    # 单独保存负样本字典
    with open('neg_samples.pkl', 'wb') as f:
        pickle.dump(neg_samples_dict, f)

    return train_graph, test_graph, neg_samples_dict















