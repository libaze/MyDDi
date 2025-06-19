import os
import random
from collections import defaultdict
import dgl
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import InMemoryDataset, Dataset as PYGDataset
from torch_geometric.data import Data, Batch, HeteroData
from rdkit import Chem
from dgl.data import DGLDataset
import pandas as pd
from torchvision import transforms
import numpy as np
from tqdm import tqdm

_BIDIRECTED_RELATION = [
    # 边类型	                            	                实体类型	    说明
    'DRUGBANK::ddi-interactor-in::Compound:Compound',  # 药物-药物	药物相互作用（对称）
    'Hetionet::CrC::Compound:Compound',  # 药物-药物	药物相似性（对称）
    'Hetionet::DrD::Disease:Disease',  # 疾病-疾病	疾病相似性（对称）
    'GNBR::H::Gene:Gene',  # 基因-基因	相同蛋白/复合物（对称）
    'INTACT::ASSOCIATION::Gene:Gene',  # 基因-基因	关联（方向性模糊）
    'INTACT::PHYSICAL ASSOCIATION::Gene:Gene',  # 基因-基因	物理关联（对称）
    'STRING::BINDING::Gene:Gene',  # 基因-基因	蛋白结合（对称）
    'bioarx::HumGenHumGen:Gene:Gene',  # 基因-基因	蛋白-蛋白相互作用（对称）

    # 方向性模糊
    # 'GNBR::B::Compound:Gene',                               # 药物-基因	绑定（方向性可能不重要）
    # 'GNBR::B::Gene:Gene',                                   # 基因-基因	绑定（对称）
    # 'GNBR::I::Gene:Gene',                                   # 基因-基因	信号通路（方向性模糊）
    # 'INTACT::COLOCALIZATION::Gene:Gene',                    # 基因-基因	共定位（对称）
]


# 负样本采样函数
def generate_negative_samples(ddis_df, drug_smiles_df, label_col='Label'):
    # 将现有样本标记为1
    pos_df = ddis_df.copy()
    pos_df[label_col] = 1

    # 生成候选负样本
    all_drugs = list(set(drug_smiles_df['DRKG_ID'].unique()))
    existing_pairs = set(zip(ddis_df['ID1'], ddis_df['ID2']))

    # 随机采样与正样本数量相同的负样本
    neg_samples = []
    while len(neg_samples) < len(pos_df):
        drug1, drug2 = random.sample(all_drugs, 2)
        if (drug1, drug2) not in existing_pairs and (drug2, drug1) not in existing_pairs:
            neg_samples.append([drug1, drug2, 0])  # 标签设为0

    # 创建负样本DataFrame
    neg_df = pd.DataFrame(neg_samples, columns=['ID1', 'ID2', label_col])

    # 合并正负样本
    balanced_df = pd.concat([pos_df, neg_df], ignore_index=True)
    return balanced_df.sample(frac=1, random_state=42)  # 打乱顺序


def load_data(data_path, type='binary_class'):
    # 原始数据加载
    ddis_df = pd.read_csv(os.path.join(data_path, 'ddis.csv'))
    drug_smiles_df = pd.read_csv(os.path.join(data_path, 'drug_smiles.csv'))
    entities_df = pd.read_csv(os.path.join(data_path, 'entities_add_h_id.csv'))
    drkg_df = pd.read_csv(os.path.join(data_path, 'drkg.csv'))
    drkg_relations_df = pd.read_csv(os.path.join(data_path, 'drkg_relations.csv'))

    # 根据不同类型处理数据
    if type == 'binary_class':
        balanced_ddis = generate_negative_samples(ddis_df, drug_smiles_df)
    elif type == 'multi_label':
        balanced_ddis = generate_negative_samples(ddis_df, drug_smiles_df)
    elif type == 'multi_class':
        ddis_df['Label'] -= 1
        balanced_ddis = ddis_df

    # 转换为list格式
    ddis = balanced_ddis.values
    drug_smiles = drug_smiles_df.values.tolist()
    drkgID2ImagePath = {DRKG_ID: os.path.join('data/drug_images/drugbank_images', '{}.png'.format(Drug_ID))
                        for Drug_ID, DRKG_ID in drug_smiles_df.values[:, :2]}
    entities = entities_df.values
    drkg = drkg_df.values
    drkg_relations = drkg_relations_df.values

    return ddis, drkg, drkgID2ImagePath, entities, drkg_relations


class DrugDataset(Dataset):
    def __init__(self, dataset_config, data, id2path, transforms=None):
        super().__init__()
        self.config = dataset_config
        self.data = data
        self.id2path = id2path
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        drug1_id, drug2_id, rel = self.data[idx]
        drug1_image = self.read_image(self.id2path[drug1_id])
        drug2_image = self.read_image(self.id2path[drug2_id])
        if self.transforms:
            drug1_image = self.transforms(drug1_image)
            drug2_image = self.transforms(drug2_image)
        return drug1_image, drug2_image, rel, drug1_id, drug2_id

    def read_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        return img


class DRKGDGLDataset(DGLDataset):
    def __init__(self, data_path, feature_size=400, pretrained=False):
        self.graph = None
        self.feature_size = feature_size
        self.pretrained = pretrained
        super().__init__(name="drkg", raw_dir=data_path)

    def process(self):
        entities = pd.read_csv(os.path.join(self.raw_dir, 'entities_add_h_id.csv')).values
        drkg = pd.read_csv(os.path.join(self.raw_dir, 'drkg.csv')).values
        drkg_relations = pd.read_csv(os.path.join(self.raw_dir, 'drkg_relations.csv')).values
        ID2H_ID = dict(zip(entities[:, 1], entities[:, 2]))
        ID2Entity = dict(zip(entities[:, 1], entities[:, 0]))
        ID2DrkgRelation = dict(zip(drkg_relations[:, 1], drkg_relations[:, 0]))

        # 预计算实体类型
        entity_types = {
            entity_id: entity.split('::')[0]
            for entity_id, entity in ID2Entity.items()
        }

        # 使用defaultdict构建图形数据
        graph_data = defaultdict(lambda: ([], []))

        for head_id, tail_id, rel_id in tqdm(drkg, desc="Building graph"):
            h_type = entity_types[head_id]
            t_type = entity_types[tail_id]
            rel_type = ID2DrkgRelation[rel_id]

            # 原始边
            key = (h_type, rel_type, t_type)
            graph_data[key][0].append(ID2H_ID[head_id])
            graph_data[key][1].append(ID2H_ID[tail_id])

            # 反向边
            if rel_type in _BIDIRECTED_RELATION:
                graph_data[key[::-1]][0].append(ID2H_ID[tail_id])
                graph_data[key[::-1]][1].append(ID2H_ID[head_id])

        # 创建异构图
        self.graph = dgl.heterograph(graph_data)

        if self.pretrained:
            pass
        else:
            # 为每种节点类型添加随机初始化的特征
            for ntype in self.graph.ntypes:
                num_nodes = self.graph.number_of_nodes(ntype)
                # 随机初始化特征
                features = torch.randn(num_nodes, 400)  # 使用正态分布初始化
                # Xavier初始化
                # features = torch.nn.init.xavier_uniform_(torch.empty(num_nodes, 400))
                self.graph.nodes[ntype].data['features'] = features  # 将特征存入图中



    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

    def save(self):
        # 保存处理后的图数据
        dgl.save_graphs(os.path.join(self.save_path, 'graph.dgl'), [self.graph])

    def load(self):
        # 加载处理后的数据
        self.graph = dgl.load_graphs(os.path.join(self.save_path, 'graph.dgl'))[0][0]

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_path, 'graph.dgl'))


class DRKGPYGDataset(PYGDataset):
    def __init__(self, root, feature_size=128, transform=None, pre_transform=None, pre_filter=None):
        self.feature_size = feature_size
        print(f'feature size: {feature_size}')
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        return ['ddis.csv', 'drkg.csv', 'drkg_relations.csv', 'drug_smiles.csv', 'entities_add_h_id.csv']

    @property
    def processed_file_names(self):
        return ['drkg_hetero_data.pt']

    def download(self):
        pass

    def process(self):
        entities = pd.read_csv(os.path.join(self.raw_dir, 'entities_add_h_id.csv')).values
        drkg = pd.read_csv(os.path.join(self.raw_dir, 'drkg.csv')).values
        drkg_relations = pd.read_csv(os.path.join(self.raw_dir, 'drkg_relations.csv')).values
        ID2H_ID = dict(zip(entities[:, 1], entities[:, 2]))
        ID2Entity = dict(zip(entities[:, 1], entities[:, 0]))
        ID2DrkgRelation = dict(zip(drkg_relations[:, 1], drkg_relations[:, 0]))

        # 预计算实体类型
        ID2EentityTypes = {
            entity_id: entity.split('::')[0].replace(' ', '_')
            for entity_id, entity in ID2Entity.items()
        }
        entity_types = sorted(list({entity.split('::')[0].replace(' ', '_') for entity_id, entity in ID2Entity.items()}))
        RelationType2EdgeIndex = defaultdict(lambda: [])
        for drkg_id1, drkg_id2, rel_id in drkg:
            RelationType2EdgeIndex[
                (ID2EentityTypes[drkg_id1], f'rel_{rel_id}', ID2EentityTypes[drkg_id2])].append(
                [ID2H_ID[drkg_id1], ID2H_ID[drkg_id2]])
            if ID2DrkgRelation[rel_id] in _BIDIRECTED_RELATION:
                RelationType2EdgeIndex[
                    (ID2EentityTypes[drkg_id1], f'rel_{rel_id}', ID2EentityTypes[drkg_id2])].append(
                    [ID2H_ID[drkg_id2], ID2H_ID[drkg_id1]])

        data = HeteroData()

        # 添加节点特征
        for entity_type in entity_types:
            num_nodes = list(ID2EentityTypes.values()).count(entity_type)
            if entity_type == 'Pharmacologic_Class':
                if self.feature_size >= num_nodes:
                    data[entity_type].x = torch.eye(num_nodes, self.feature_size)
                else:
                    data[entity_type].x = torch.randn(num_nodes, self.feature_size)
            else:
                data[entity_type].x = torch.randn(num_nodes, self.feature_size)

        # 添加边关系
        relation_types = list(RelationType2EdgeIndex.keys())
        for src, rel, dst in relation_types:
            edge_list = RelationType2EdgeIndex[(src, rel, dst)]
            edge_index = torch.tensor(edge_list).t().contiguous()  # Transpose to [2, num_edges]
            data[src, rel, dst].edge_index = edge_index

        torch.save(data, os.path.join(self.processed_dir, f'drkg_hetero_data.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx: int):
        data = torch.load(os.path.join(self.processed_dir, f'drkg_hetero_data.pt'))
        return data


if __name__ == '__main__':
    with open('/home/work/workspace/liu_lei/MyDDi/config/finetune/train.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print(config['dataset'])
    # ddis, drkg, drkgID2ImagePath, entities = load_data(config['dataset']['path'])
    # basic_transforms = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     )
    # ])
    # d = DrugDataset(config['dataset'], ddis, drkgID2ImagePath, basic_transforms)
    # print(len(d))
    # print(d[999])

    d = DRKGPYGDataset(root=config['dataset']['path'])
    g = d[0]
    print(g)
