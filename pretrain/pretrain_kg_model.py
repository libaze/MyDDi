import dgl
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from data_process.generate_graph import gen_hetero_graph
from sklearn.metrics import roc_auc_score
from model.kg_model import HeteroRGCN


# # 定义训练函数
# def pretrain_one_epoch(model, graph, optimizer, device):
#     model.train()
#     optimizer.zero_grad()
#
#     # 获取训练边
#     train_edges = {}
#     for rel_type in relation_types:
#         src, dst = graph.edges(etype=rel_type)
#         train_edges[rel_type] = (src, dst)
#
#     # 计算损失
#     loss = 0
#     for rel_type in relation_types:
#         pos_src, pos_dst = train_edges[rel_type]
#         neg_dst = torch.randint(0, graph.number_of_nodes(entity_types[-1]), (pos_src.shape[0],))
#
#         # 正样本预测
#         pos_pred = model.predict(graph, ((entity_types[0], relation_types[0], entity_types[0]), pos_src, pos_dst))
#         # 负样本预测
#         neg_pred = model.predict(graph, ((entity_types[0], relation_types[0], entity_types[0]), pos_src, neg_dst))
#
#         # 二元交叉熵损失
#         pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
#         neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
#         loss += pos_loss + neg_loss
#
#     # 反向传播
#     loss.backward()
#     optimizer.step()
#
#     return loss.item()
#
#
# # 定义评估函数
# def evaluate(model, graph, edges, device):
#     model.eval()
#
#     with torch.no_grad():
#         preds = []
#         labels = []
#
#         for rel_type in relation_types:
#             src, dst = graph.edges(etype=rel_type)
#             neg_dst = torch.randint(0, graph.number_of_nodes(entity_types[-1]), (src.shape[0],))
#
#             # 正样本
#             pos_pred = model.predict(graph, ((entity_types[0], relation_types[0], entity_types[0]), src, dst))
#             preds.append(pos_pred.cpu().numpy())
#             labels.append(np.ones(len(pos_pred)))
#
#             # 负样本
#             neg_pred = model.predict(graph, ((entity_types[0], relation_types[0], entity_types[0]), src, neg_dst))
#             preds.append(neg_pred.cpu().numpy())
#             labels.append(np.zeros(len(neg_pred)))
#
#         preds = np.concatenate(preds)
#         labels = np.concatenate(labels)
#
#         auc = roc_auc_score(labels, preds)
#         return auc
#
#
# # 预测新链接的函数
# def predict_new_link(model, graph, head, head_type, tail, tail_type, relation_type):
#     model.eval()
#     with torch.no_grad():
#         # 获取节点ID
#         head_id = torch.tensor([head]).to(device)
#         tail_id = torch.tensor([tail]).to(device)
#
#         # 预测
#         pred = model.predict(graph, ((head_type, relation_type, tail_type), head_id, tail_id))
#         return pred.item()


import dgl
import torch
import numpy as np
from sklearn.model_selection import train_test_split


def split_link_prediction_graph(graph, test_ratio=0.2, val_ratio=0.1):
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
    # 1. 为每个关系类型生成划分
    neg_samples_dict = {}
    train_edges = {}
    val_edges = {}
    test_edges = {}

    for canonical_etype in graph.canonical_etypes:
        src_type, rel_type, dst_type = canonical_etype
        src, dst = graph.edges(etype=canonical_etype)
        edges = torch.stack([src, dst], dim=1)

        # 生成负样本（确保不存在于图中）
        num_neg = len(edges)
        neg_src = torch.randint(0, graph.num_nodes(src_type), (num_neg,))
        neg_dst = torch.randint(0, graph.num_nodes(dst_type), (num_neg,))

        # 过滤掉真实存在的边
        for i in range(num_neg):
            while graph.has_edges_between(neg_src[i], neg_dst[i], etype=canonical_etype):
                neg_src[i] = torch.randint(0, graph.num_nodes(src_type), (1,))
                neg_dst[i] = torch.randint(0, graph.num_nodes(dst_type), (1,))
        neg_samples = torch.stack([neg_src, neg_dst], dim=1)

        # 划分正样本
        train_pos, temp_pos = train_test_split(
            edges, test_size=test_ratio + val_ratio, random_state=42
        )
        val_pos, test_pos = train_test_split(
            temp_pos, test_size=test_ratio / (test_ratio + val_ratio), random_state=42
        )

        # 划分负样本（相同比例）
        train_neg, temp_neg = train_test_split(
            neg_samples, test_size=test_ratio + val_ratio, random_state=42
        )
        val_neg, test_neg = train_test_split(
            temp_neg, test_size=test_ratio / (test_ratio + val_ratio), random_state=42
        )

        # 存储划分结果
        train_edges[canonical_etype] = train_pos
        val_edges[canonical_etype] = val_pos
        test_edges[canonical_etype] = test_pos
        neg_samples_dict[canonical_etype] = {
            'train': train_neg,
            'val': val_neg,
            'test': test_neg
        }

    # 2. 构建子图
    train_graph = dgl.heterograph(train_edges)
    val_graph = dgl.heterograph(val_edges)
    test_graph = dgl.heterograph(test_edges)

    # 3. 保留原始特征
    for ntype in graph.ntypes:
        for key in graph.nodes[ntype].data:
            train_graph.nodes[ntype].data[key] = graph.nodes[ntype].data[key]
            val_graph.nodes[ntype].data[key] = graph.nodes[ntype].data[key]
            test_graph.nodes[ntype].data[key] = graph.nodes[ntype].data[key]

    return train_graph, val_graph, test_graph, neg_samples_dict


def pretrain_kg():
    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 使用显式列规范加载数据
    entities = pd.read_csv('../data/finetune/DRKGWithDrugBank/entities_add_h_id.csv').values
    drkg_relations = pd.read_csv('../data/finetune/DRKGWithDrugBank/drkg_relations.csv').values
    drkg = pd.read_csv('../data/finetune/DRKGWithDrugBank/drkg.csv').values
    ID2H_ID = dict(zip(entities[:, 1], entities[:, 2]))

    # 打印基本信息
    print(f"Entities: {entities.shape[0]:,} rows")
    print(f"Relations: {drkg_relations.shape[0]:,} types")
    print(f"Edges: {drkg.shape[0]:,} connections")

    # 加载数据集
    graph = gen_hetero_graph(drkg, drkg_relations, entities, ID2H_ID)

    # 打印图摘要
    print("\nGraph summary:")
    print(f"Node types: {graph.ntypes}")
    print(f"Edge types: {graph.etypes}")
    print(f"Total nodes: {graph.num_nodes():,}")
    print(f"Total edges: {graph.num_edges():,}")

    # 获取实体和关系类型
    entity_types = graph.ntypes
    # relation_types = graph.etypes
    print(f"实体类型: {entity_types}")
    print(f"关系类型: {relation_types}")

    # 设置设备
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化模型
    # graph = graph.to(device)
    # model = HeteroRGCN(graph, in_size=64, hidden_size=128, out_size=1, num_layers=2).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 训练循环
    # num_epochs = 50
    # best_val_auc = 0

    # for epoch in range(num_epochs):
    #     loss = pretrain_one_epoch(model, graph, optimizer, device)
    #     train_auc = evaluate(model, graph, train_mask, device)
    #     val_auc = evaluate(model, graph, val_mask, device)
    #
    #     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
    #
    #     if val_auc > best_val_auc:
    #         best_val_auc = val_auc
    #         torch.save(model.state_dict(), 'best_model.pth')
    #
    # # 加载最佳模型
    # model.load_state_dict(torch.load('best_model.pth'))
    #
    # # 在测试集上评估
    # test_auc = evaluate(model, graph, test_mask, device)
    # print(f"测试集AUC: {test_auc:.4f}")
    #
    #
    #
    # # 示例预测
    # score = predict_new_link(model, graph, 0, '_N', 100, '_N', '_E')
    # print(f"链接预测得分: {score:.4f}")


if __name__ == '__main__':
    pretrain_kg()