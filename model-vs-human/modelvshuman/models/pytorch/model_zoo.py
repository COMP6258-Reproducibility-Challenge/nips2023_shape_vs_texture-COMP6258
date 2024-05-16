#!/usr/bin/env python3
import torch
from torchvision import models
from ..registry import register_model
from ..wrappers.pytorch import PytorchModel, PyContrastPytorchModel, ClipPytorchModel, \
    ViTPytorchModel, EfficientNetPytorchModel, SwagPytorchModel

_PYTORCH_IMAGE_MODELS = "rwightman/pytorch-image-models"

_EFFICIENTNET_MODELS = "rwightman/gen-efficientnet-pytorch"

"""
To test our own implementation:
Replace from .topK import topK_AlexNet
To from .topK.topK_custom import topK_AlexNet
And then follow instructions provided
"""


def model_pytorch(model_name, *args):
    import torchvision.models as zoomodels
    model = zoomodels.__dict__[model_name](pretrained=True)
    model = torch.nn.DataParallel(model)
    return PytorchModel(model, model_name, *args)

@register_model("pytorch")
def AlexNet_topK_50(model_name, *args):
    from .topK import topK_AlexNet
    alexNet = topK_AlexNet(pretrain_weigth="",topk=0.5)
    return PytorchModel(alexNet, model_name, *args)

@register_model("pytorch")
def AlexNet_topK_40(model_name, *args):
    from .topK import topK_AlexNet
    alexNet = topK_AlexNet(pretrain_weigth="",topk=0.4)
    return PytorchModel(alexNet, model_name, *args)

@register_model("pytorch")
def AlexNet_topK_30(model_name, *args):
    from .topK import topK_AlexNet
    alexNet = topK_AlexNet(pretrain_weigth="",topk=0.3)
    return PytorchModel(alexNet, model_name, *args)

@register_model("pytorch")
def AlexNet_topK_20(model_name, *args):
    from .topK import topK_AlexNet
    alexNet = topK_AlexNet(pretrain_weigth="",topk=0.2)
    return PytorchModel(alexNet, model_name, *args)

@register_model("pytorch")
def AlexNet_topK_10(model_name, *args):
    from .topK import topK_AlexNet
    alexNet = topK_AlexNet(pretrain_weigth="",topk=0.1)
    return PytorchModel(alexNet, model_name, *args)

@register_model("pytorch")
def AlexNet_topK_5(model_name, *args):
    from .topK import topK_AlexNet
    alexNet = topK_AlexNet(pretrain_weigth="",topk=0.05)
    return PytorchModel(alexNet, model_name, *args)

@register_model("pytorch")
def AlexNet_normal(model_name, *args):
    alexnet = models.alexnet(pretrained=True)
    return PytorchModel(alexnet, model_name, *args)