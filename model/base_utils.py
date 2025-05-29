import timm
import torch
import torchvision
from torch import nn
from typing import Callable


_MODEL_DICT = {
    # ResNet系列
    "ResNet18": torchvision.models.resnet18,
    "ResNet34": torchvision.models.resnet34,
    "ResNet50": torchvision.models.resnet50,
    "ResNet101": torchvision.models.resnet101,
    "ResNet152": torchvision.models.resnet152,
    # Swin Transformer系列
    "swin_tiny_patch4_window7_224": lambda **kwargs: timm.create_model('swin_tiny_patch4_window7_224', **kwargs),
    "swin_base_patch4_window7_224": lambda **kwargs: timm.create_model('swin_base_patch4_window7_224', **kwargs)
}


# 初始化权重
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_timm_model_names():
    model_list = timm.list_models()
    return model_list


def get_model(model_name: str, pretrained: bool = False, only_feat: bool = True, ret_in_features: bool = True):
    """
    优化后的模型加载函数，支持缓存和统一接口

    参数:
        model_name (str): 预定义的模型名称
        pretrained (bool): 是否加载预训练权重
        only_feat (bool): 是否移除分类头

    返回:
        model: 加载的PyTorch模型
    """
    if model_name not in _MODEL_DICT:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(_MODEL_DICT.keys())}")

    # 加载基础模型
    model = _MODEL_DICT[model_name](pretrained=pretrained)
    in_features = list(model.children())[-1].in_features

    # 移除分类头（特征提取模式）
    if only_feat:
        if model_name.startswith('swin'):
            model.head = torch.nn.Identity()
        elif model_name.startswith('ResNet'):
            model.fc = torch.nn.Identity()

    # 返回最后输出特征维度
    if ret_in_features:
        return model, in_features
    return model


def get_activation_fn(activation: str) -> Callable:
    """ 返回与 'activation' 对应的激活函数 """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "softplus":
        return nn.Softplus()
    elif activation == "linear":
        return nn.Identity()
    else:
        raise RuntimeError("--激活函数 {} 不支持！".format(activation))


if __name__ == '__main__':
    model = get_model('ResNet18')
    print(model)
