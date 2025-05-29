import torch
from torch import nn
from model.base_utils import weights_init


class DDIClassifier(nn.Module):
    def __init__(self, in_features, num_classes=2):
        super(DDIClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.apply(weights_init)

    def forward(self, x1, x2):
        # 类型与维度验证
        assert isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor), "输入必须为张量"
        assert x1.dim() == x2.dim() == 2, "输入必须为2D张量 (batch, features)"
        assert x1.size(0) == x2.size(0), "批次大小需一致"
        x = torch.cat((x1, x2), dim=1)
        logit = self.classifier(x)
        return logit


if __name__ == '__main__':
    x1 = torch.randn(8, 512)
    x2 = torch.randn(8, 512)
    ddi_classifier = DDIClassifier(in_features=1024, num_classes=86)
    output = ddi_classifier(x1, x2)
    print(output.shape)