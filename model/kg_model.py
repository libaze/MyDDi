import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.data import FB15k237Dataset
from dgl.nn.pytorch import HeteroGraphConv, GraphConv, SAGEConv
from sklearn.metrics import roc_auc_score
import torch.nn.init as init


# 定义异构图的RGCN模型
class HeteroRGCN(nn.Module):
    def __init__(self, graph, in_size, hidden_size, out_size, num_layers=2):
        super(HeteroRGCN, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_layers = num_layers

        # 可学习的节点嵌入
        self.embeddings = nn.ParameterDict({
            ntype: nn.Parameter(torch.Tensor(graph.number_of_nodes(ntype), in_size))
            for ntype in graph.ntypes
        })

        # 关系特定的转换权重
        self.rel_weights = nn.ModuleDict({
            rel_type: nn.Linear(in_size, hidden_size)
            for rel_type in graph.etypes
        })

        # 分层的RGCN卷积
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroGraphConv({
                rel_type: GraphConv(
                    hidden_size if i > 0 else in_size,
                    hidden_size,
                )
                for rel_type in graph.etypes
            }, aggregate='mean')  # 使用均值聚合更稳定

            self.convs.append(conv)

        # 分类器（添加dropout和层归一化）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, out_size)
        )

        # 初始化所有参数
        self.reset_parameters()

    def reset_parameters(self):
        # 节点嵌入初始化
        for param in self.embeddings.values():
            init.kaiming_uniform_(param, nonlinearity='relu')

        # 关系权重初始化
        for module in self.rel_weights.values():
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)

        # 卷积层初始化
        for conv in self.convs:
            for rel_type in relation_types:
                init.xavier_uniform_(conv.mods[rel_type].weight)
                if conv.mods[rel_type].bias is not None:
                    init.zeros_(conv.mods[rel_type].bias)

        # 分类器初始化
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, graph, node_ids=None):
        # 获取节点嵌入
        h = {
            ntype: self.embeddings[ntype][
                torch.arange(graph.number_of_nodes(ntype)).to(graph.device)
                if node_ids is None else node_ids[ntype]
            ]
            for ntype in graph.ntypes
        }

        # 应用关系特定的转换
        h = {
            ntype: self.rel_weights[ntype](h[ntype])
            if ntype in self.rel_weights else h[ntype]
            for ntype in h
        }

        # 通过RGCN层（添加残差连接）
        for i, conv in enumerate(self.convs):
            out = conv(graph, h)
            out = {k: F.leaky_relu(v, negative_slope=0.2) for k, v in out.items()}

            # 从第二层开始添加残差连接
            if i > 0:
                out = {k: out[k] + h[k][:, :self.hidden_size] if k in h else out[k] for k in out}

            h = out

        return h

    def predict(self, graph, edges):
        # 获取节点表示
        h = self.forward(graph)

        # 获取源节点和目标节点的嵌入
        src_type, _, dst_type = edges[0]
        src_ids = edges[1].to(graph.device)
        dst_ids = edges[2].to(graph.device)

        src_emb = h[src_type][src_ids]
        dst_emb = h[dst_type][dst_ids]

        # 拼接并使用分类器
        combined = torch.cat([src_emb, dst_emb], dim=1)
        return torch.sigmoid(self.classifier(combined))

    def regularization_loss(self):
        """计算L2正则化损失"""
        reg_loss = 0.0
        for param in self.parameters():
            reg_loss += torch.norm(param, p=2)
        return reg_loss


# if __name__ == '__main__':
#     # 使用显式列规范加载数据
#     entities = pd.read_csv('../data/finetune/DRKGWithDrugBank/entities_add_h_id.csv').values
#     drkg_relations = pd.read_csv('../data/finetune/DRKGWithDrugBank/drkg_relations.csv').values
#     drkg = pd.read_csv('../data/finetune/DRKGWithDrugBank/drkg.csv').values
#
#     # 打印基本信息
#     print(f"Entities: {entities.shape[0]:,} rows")
#     print(f"Relations: {drkg_relations.shape[0]:,} types")
#     print(f"Edges: {drkg.shape[0]:,} connections")
#
#     # 生成图
#     hetero_graph, rel2id = gen_hetero_graph(drkg, drkg_relations, entities)
#
#     kg_model = DRKGModel(hetero_graph, 256, 512, 'GraphConv', 0.5)
#
#     # h = kg_model(hetero_graph)
#     # print(h.keys())


