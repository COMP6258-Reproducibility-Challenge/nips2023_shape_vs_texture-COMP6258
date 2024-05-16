import torch
import torch.nn as nn
from torchvision import models


def AlexNet_topK_50():
    from .topK_edited import topK_AlexNet
    alexnet = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
    alexNet = topK_AlexNet(alexnet,topk=0.5)
    return alexNet


def AlexNet_topK_40():
    from .topK_edited import topK_AlexNet
    alexnet = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
    alexNet = topK_AlexNet(alexnet,topk=0.4)
    return alexNet


def AlexNet_topK_30():
    from .topK_edited import topK_AlexNet
    alexnet = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
    alexNet = topK_AlexNet(alexnet,topk=0.3)
    return alexNet


def AlexNet_topK_20():
    from .topK_edited import topK_AlexNet
    alexnet = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
    alexNet = topK_AlexNet(alexnet,topk=0.2)
    return alexNet


def AlexNet_topK_10():
    from .topK_edited import topK_AlexNet
    alexnet = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
    alexNet = topK_AlexNet(alexnet,topk=0.1)
    return alexNet

def AlexNet_topK_5():
    from .topK_edited import topK_AlexNet
    alexnet = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
    alexNet = topK_AlexNet(alexnet,topk=0.05)
    return alexNet