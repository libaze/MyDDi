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
            for rel_type in relation_types
        })

        # 分层的RGCN卷积
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroGraphConv({
                rel_type: GraphConv(
                    hidden_size if i > 0 else in_size,
                    hidden_size,
                )
                for rel_type in relation_types
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


# 定义训练函数
def train(model, graph, optimizer, device):
    model.train()
    optimizer.zero_grad()

    # 获取训练边
    train_edges = {}
    for rel_type in relation_types:
        src, dst = graph.edges(etype=rel_type)
        train_edges[rel_type] = (src, dst)

    # 计算损失
    loss = 0
    for rel_type in relation_types:
        pos_src, pos_dst = train_edges[rel_type]
        neg_dst = torch.randint(0, graph.number_of_nodes(entity_types[-1]), (pos_src.shape[0],))

        # 正样本预测
        pos_pred = model.predict(graph, ((entity_types[0], relation_types[0], entity_types[0]), pos_src, pos_dst))
        # 负样本预测
        neg_pred = model.predict(graph, ((entity_types[0], relation_types[0], entity_types[0]), pos_src, neg_dst))

        # 二元交叉熵损失
        pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        loss += pos_loss + neg_loss

    # 反向传播
    loss.backward()
    optimizer.step()

    return loss.item()


# 定义评估函数
def evaluate(model, graph, edges, device):
    model.eval()

    with torch.no_grad():
        preds = []
        labels = []

        for rel_type in relation_types:
            src, dst = graph.edges(etype=rel_type)
            neg_dst = torch.randint(0, graph.number_of_nodes(entity_types[-1]), (src.shape[0],))

            # 正样本
            pos_pred = model.predict(graph, ((entity_types[0], relation_types[0], entity_types[0]), src, dst))
            preds.append(pos_pred.cpu().numpy())
            labels.append(np.ones(len(pos_pred)))

            # 负样本
            neg_pred = model.predict(graph, ((entity_types[0], relation_types[0], entity_types[0]), src, neg_dst))
            preds.append(neg_pred.cpu().numpy())
            labels.append(np.zeros(len(neg_pred)))

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        auc = roc_auc_score(labels, preds)
        return auc



# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 加载数据集
dataset = FB15k237Dataset()
graph = dataset[0]

# 获取实体和关系类型
entity_types = graph.ntypes
relation_types = graph.etypes
print(f"实体类型: {entity_types}")
print(f"关系类型: {relation_types}")

# 划分训练/验证/测试集
train_mask = graph.edata['train_mask']
val_mask = graph.edata['val_mask']
test_mask = graph.edata['test_mask']

# 设置设备
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 初始化模型
graph = graph.to(device)
model = HeteroRGCN(graph, in_size=64, hidden_size=128, out_size=1, num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练循环
num_epochs = 50
best_val_auc = 0

for epoch in range(num_epochs):
    loss = train(model, graph, optimizer, device)
    train_auc = evaluate(model, graph, train_mask, device)
    val_auc = evaluate(model, graph, val_mask, device)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), 'best_model.pth')

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 在测试集上评估
test_auc = evaluate(model, graph, test_mask, device)
print(f"测试集AUC: {test_auc:.4f}")


# 预测新链接的函数
def predict_new_link(model, graph, head, head_type, tail, tail_type, relation_type):
    model.eval()
    with torch.no_grad():
        # 获取节点ID
        head_id = torch.tensor([head]).to(device)
        tail_id = torch.tensor([tail]).to(device)

        # 预测
        pred = model.predict(graph, ((head_type, relation_type, tail_type), head_id, tail_id))
        return pred.item()

# 示例预测
# 假设我们想预测实体0(类型为'entity')和实体100(类型为'entity')之间是否存在关系'relation_type'
score = predict_new_link(model, graph, 0, '_N', 100, '_N', '_E')
print(f"链接预测得分: {score:.4f}")