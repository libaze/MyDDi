import torch
import torch.nn as nn
from .base_utils import get_model


class MolModel(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        self.model_name = model_name
        backbone, in_features = get_model(model_name, pretrained)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        if self.model_name.startswith('swin'):
            self.layer = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.GELU(),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.model_name.startswith('ResNet'):
            self.layer = nn.Sequential(
                nn.Conv2d(in_features, 512, 1),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.backbone(x)
        if self.model_name.startswith('swin'):
            x = self.layer(x)
            x = x.permute(0, 3, 1, 2)
            x = self.avgpool(x)
        elif self.model_name.startswith('ResNet'):
            x = self.layer(x)
        return x.flatten(1)

if __name__ == '__main__':
    x = torch.randn(8, 3, 224, 224)
    net = MolModel('swin_base_patch4_window7_224')
    print(net)
    output = net(x)
    print(output.shape)