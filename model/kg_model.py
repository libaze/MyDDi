import dgl
from dgl.nn.pytorch import HeteroGraphConv, GraphConv, SAGEConv
from torch import nn
import torch.nn.functional as F


class DRKGModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, rel_names, num_layers=3, g_type='SAGEConv', num_heads=4, dropout=0.5, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node = 'Pharmacologic Class'
        self.dropout = dropout

        self.norms = nn.ModuleDict({
            **{f'layer_{i}': nn.LayerNorm(hidden_dim) for i in range(num_layers)}
        })

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(HeteroGraphConv({
                rel: SAGEConv(in_dim, hidden_dim, "mean")
                for rel in rel_names
            }))

        self.linears = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.linears.append(nn.Linear(in_dim, hidden_dim))

    def forward(self, blocks, x):
        hidden_x = x
        for layer_idx, (layer, block, linear) in enumerate(zip(self.layers, blocks, self.linears)):
            hidden_x = layer(block, hidden_x)
            x[self.node] = linear(x[self.node])

            hidden_x = {k: self.norms[f'layer_{layer_idx}'](v) for k, v in hidden_x.items()}
            x[self.node] = self.norms[f'layer_{layer_idx}'](x[self.node])

            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training)
                           for k, v in hidden_x.items()}
                x[self.node] = F.dropout(F.relu(x[self.node]), p=self.dropout, training=self.training)

        hidden_x = x | hidden_x
        hidden_x[self.node] = hidden_x[self.node][: blocks[-1].dstdata[dgl.NID][self.node].shape[0], :]
        return hidden_x