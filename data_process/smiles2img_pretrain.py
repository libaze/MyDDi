import argparse
import json
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm


def Smiles2Image(smiles, save_path):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
        path: E:/a/b/c.png
        ==============================================================================================================
        demo:
            smiless = ["OC[C@@H](NC(=O)C(Cl)Cl)[C@H](O)C1=CC=C(C=C1)[N+]([O-])=O", "CN1CCN(CC1)C(C1=CC=CC=C1)C1=CC=C(Cl)C=C1",
              "[H][C@@](O)(CO)[C@@]([H])(O)[C@]([H])(O)[C@@]([H])(O)C=O", "CNC(NCCSCC1=CC=C(CN(C)C)O1)=C[N+]([O-])=O",
              "[H]C(=O)[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)CO", "CC[C@H](C)[C@H](NC(=O)[C@H](CC1=CC=C(O)C=C1)NC(=O)[C@@H](NC(=O)[C@H](CCCN=C(N)N)NC(=O)[C@@H](N)CC(O)=O)C(C)C)C(=O)N[C@@H](CC1=CN=CN1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CC1=CC=CC=C1)C(O)=O"]

            for idx, smiles in enumerate(smiless):
                loadSmilesAndSave(smiles, "{}.png".format(idx+1))
        ==============================================================================================================
    '''
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Draw.MolToFile(mol, save_path, size=(224, 224))
        return True, smiles
    else:
        print(f"Skipping {smiles}")
        return False, smiles


def get_args():
    '''
    demo of csv_file:
        mf,smiles1,smiles2,smiles3
        C15H12N2O,Cc1ccc([C@H]2[CH]c3cnccc3[N]C2=O)cc1,N#Cc1ccc([CH]N([O])Cc2ccccc2)cc1,N#C[C@@H]1C=C2CCCc3ccccc3C2=NC1=O
        C13H17NO3,COc1ccc(/C=C/N(C)C(C)=O)c(OC)c1,[H]/N=C(O)\C=C\c1ccc(OCC)c(OCC)c1,CCO[C@H]1ON=C(c2ccccc2)C[C@@H]1OC
        C14H17NO,C=CCN(/C=C/c1ccccc1C)C(C)=O,Cc1c([C@H]2CCCCN2)oc2ccccc12,CC(C)[C@](C)(C#N)CC(=O)c1ccccc1
        ...
    :return:
    '''
    parser = argparse.ArgumentParser(description='Pretraining Data Generation')
    parser.add_argument('--csv_file_path', type=str,
                        default="../datasets/finetune/drugbank/drug_smiles.csv",
                        help='Path to the CSV file containing SMILES strings for molecular structures')
    parser.add_argument('--img_save_folder', type=str,
                        default="../datasets/finetune/drugbank_images",
                        help='Directory path where generated molecular images will be saved')
    parser.add_argument('--error_save_path', type=str,
                        default="../datasets/pretrain/error_mf.json",
                        help='Path to the JSON file where error logs for failed molecular image generations will be stored')
    args = parser.parse_args()
    return args


def main(args):
    df = pd.read_csv(args.csv_file_path)
    smiles_list = df.values.tolist()
    error_mf = []

    with (tqdm(total=len(smiles_list), desc='Mol SMILES --> Mol Image') as pbar):
        for item in smiles_list:
            mf = item[0]
            for i, smiles in enumerate(item[1:]):
                output_path = os.path.join(args.img_save_folder, f'{mf}-{i}.png')
                bbb, sss = Smiles2Image(smiles, output_path)
                if not bbb:
                    print(f"Skipping {mf}-{smiles}")
                    error_mf.append(mf)
            pbar.update(1)

    if len(error_mf):
        with open(args.error_save_path, 'w') as f:
            json.dump({'error_mf': error_mf}, f)





if __name__ == '__main__':
    args = get_args()
    main(args)
