import torch
import torch.nn as nn
from .mol_model import MolModel
from .kg_model import DRKGModel
from .predictor import DDIClassifier


class DDINet(nn.Module):
    def __init__(self, config, g=None, id2h_id=None):
        super(DDINet, self).__init__()
        self.config = config
        self.model_type = self.config['type']
        self.num_classes = self.config['num_classes']
        self.id2h_id = id2h_id

        # Initialize models
        if self.model_type in ['only_mol', 'mol+kg']:
            self.mol_model = MolModel(
                self.config['mol_model']['type'],
                self.config['mol_model']['pretrained']
            )

        if self.model_type in ['only_kg', 'mol+kg']:
            self.kg_model = DRKGModel(g, **self.config['kg_model'])
            # self.g = g

        # Initialize classifier
        if self.model_type == 'only_mol':
            self.classifier = DDIClassifier(in_features=1024, num_classes=self.num_classes)
        elif self.model_type == 'only_kg':
            self.classifier = DDIClassifier(in_features=1024, num_classes=self.num_classes)
        elif self.model_type == 'mol+kg':
            self.fusion_method = self.config.get('fusion_method', 'concat')
            in_features = 2048 if self.fusion_method == 'concat' else 1024
            self.classifier = DDIClassifier(in_features=in_features, num_classes=self.num_classes)

    def forward(self, drug1, drug2, drug1_id, drug2_id, g=None, n_feat=None):
        if self.model_type == 'only_mol':
            drug1_mol_features = self.mol_model(drug1)
            drug2_mol_features = self.mol_model(drug2)
            drug1_features = drug1_mol_features
            drug2_features = drug2_mol_features
            return self.classifier(drug1_features, drug2_features)
        elif self.model_type == 'only_kg':
            drug1_kg_features = self.kg_model(g, n_feat)
            drug2_kg_features = self.kg_model(g, n_feat)
            drug1_features = drug1_kg_features[self.id2h_id[drug1_id]]
            drug2_features = drug2_kg_features[self.id2h_id[drug2_id]]
            return self.classifier(drug1_features, drug2_features)
        elif self.model_type == 'mol+kg':
            drug1_mol_features = self.mol_model(drug1)
            drug2_mol_features = self.mol_model(drug2)
            drug1_kg_features = self.kg_model(g, n_feat)
            drug2_kg_features = self.kg_model(g, n_feat)

            if self.fusion_method == 'concat':
                drug1_features = torch.cat([drug1_mol_features, drug1_kg_features], dim=1)
                drug2_features = torch.cat([drug2_mol_features, drug2_kg_features], dim=1)
            else:  # add
                drug1_features = drug1_mol_features + drug1_kg_features
                drug2_features = drug2_mol_features + drug2_kg_features
            return self.classifier(drug1_features, drug2_features)

    # def gen_hetero_graph(self):
    #     gen_hetero_graph()

