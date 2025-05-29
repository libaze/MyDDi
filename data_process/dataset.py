import os
import random
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from rdkit import Chem
import pandas as pd
from torchvision import transforms
import numpy as np

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



if __name__ == '__main__':
    with open('/home/work/workspace/liu_lei/MyDDi/config/finetune/train.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print(config['dataset'])
    ddis, drkg, drkgID2ImagePath, entities = load_data(config['dataset']['path'])
    basic_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    d = DrugDataset(config['dataset'], ddis, drkgID2ImagePath, basic_transforms)
    print(len(d))
    print(d[999])

