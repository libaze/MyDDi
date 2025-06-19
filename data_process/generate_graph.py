import os
from collections import defaultdict
# import dgl
import pandas as pd
import torch
from tqdm import tqdm

_BIDIRECTED_RELATION = [
    # 边类型	                            	                实体类型	    说明
    'DRUGBANK::ddi-interactor-in::Compound:Compound',       # 药物-药物	药物相互作用（对称）
    'Hetionet::CrC::Compound:Compound',                     # 药物-药物	药物相似性（对称）
    'Hetionet::DrD::Disease:Disease',                       # 疾病-疾病	疾病相似性（对称）
    'GNBR::H::Gene:Gene',                                   # 基因-基因	相同蛋白/复合物（对称）
    'INTACT::ASSOCIATION::Gene:Gene',                       # 基因-基因	关联（方向性模糊）
    'INTACT::PHYSICAL ASSOCIATION::Gene:Gene',              # 基因-基因	物理关联（对称）
    'STRING::BINDING::Gene:Gene',                           # 基因-基因	蛋白结合（对称）
    'bioarx::HumGenHumGen:Gene:Gene',                       # 基因-基因	蛋白-蛋白相互作用（对称）
    # 方向性模糊
    # 'GNBR::B::Compound:Gene',                               # 药物-基因	绑定（方向性可能不重要）
    # 'GNBR::B::Gene:Gene',                                   # 基因-基因	绑定（对称）
    # 'GNBR::I::Gene:Gene',                                   # 基因-基因	信号通路（方向性模糊）
    # 'INTACT::COLOCALIZATION::Gene:Gene',                    # 基因-基因	共定位（对称）
]


def gen_hetero_graph(drkg, drkg_relations, entities, ID2H_ID):
    # 初始化映射字典
    # DrkgRelation2ID = dict(zip(drkg_relations[:, 0], drkg_relations[:, 1]))
    ID2Entity = dict(zip(entities[:, 1], entities[:, 0]))
    ID2DrkgRelation = dict(zip(drkg_relations[:, 1], drkg_relations[:, 0]))

    hetero_graph_path = 'hetero_graph.dgl'
    if os.path.exists(hetero_graph_path):
        hetero_graph, _ = dgl.load_graphs(hetero_graph_path)
        return hetero_graph[0]

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
    hetero_graph = dgl.heterograph(graph_data)

    dgl.save_graphs(hetero_graph_path, [hetero_graph])
    return hetero_graph


if __name__ == '__main__':
    # 使用显式列规范加载数据
    entities = pd.read_csv('../data/finetune/DRKGWithDrugBank/entities_add_h_id.csv').values
    drkg_relations = pd.read_csv('../data/finetune/DRKGWithDrugBank/drkg_relations.csv').values
    drkg = pd.read_csv('../data/finetune/DRKGWithDrugBank/drkg.csv').values
    ID2H_ID = dict(zip(entities[:, 1], entities[:, 2]))
 
    # 打印基本信息
    print(f"Entities: {entities.shape[0]:,} rows")
    print(f"Relations: {drkg_relations.shape[0]:,} types")
    print(f"Edges: {drkg.shape[0]:,} connections")

    # 生成图
    hetero_graph = gen_hetero_graph(drkg, drkg_relations, entities, ID2H_ID)
    print(hetero_graph)
    

    # 打印图摘要
    print("\nGraph summary:")
    print(f"Node types: {hetero_graph.ntypes}")
    print(f"Edge types: {hetero_graph.etypes}")
    print(f"Total nodes: {hetero_graph.num_nodes():,}")
    print(f"Total edges: {hetero_graph.num_edges():,}")
