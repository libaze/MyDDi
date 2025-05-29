import pandas as pd


with open('drkg/embed/entities.tsv', 'r') as f:
    entity_infos = f.read().strip().split('\n')

compound_entities2id = []
for entity_info in entity_infos:
    _name, _id = entity_info.split('\t')
    _type, _nid = _name.split('::')
    if _type == 'Compound':
        compound_entities2id.append([_type, _nid, _id])

print(len(compound_entities2id))

drugbank2drkgID = {}
df = pd.read_csv('finetuning/drugbank-1704/drug_smiles.csv')
for drug_id, _ in df.values.tolist():
    for t, n, _id in compound_entities2id:
        if drug_id == n:
            drugbank2drkgID[drug_id] = _id

print(drugbank2drkgID)
print(len(drugbank2drkgID))

print(set(df.values[:, 0].tolist()) - set(drugbank2drkgID.keys()))

drugbank2drkgID_list = []
for k, v in drugbank2drkgID.items():
    drugbank2drkgID_list.append([k, v])

print(len(drugbank2drkgID_list))

ddff = pd.DataFrame(drugbank2drkgID_list, columns=['DrugBankID', 'ID'])
print(ddff)
# ddff.to_csv('DrugBankID2DrkgID.csv', index=False)










