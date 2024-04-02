from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchvision import models
import torchvision.transforms as transforms
from torchvision.models.alexnet import AlexNet_Weights
from torchvision.models.vgg import VGG19_Weights
import torch.nn as nn
import hashlib



class TopKLayer(nn.Module):
    def __init__(self, topk = 1, device = 'cuda'):
        super().__init__()
        self.topk = topk
        self.device = device
        self.target = False

    def forward(self, x):
        n, c, h, w = x.shape

        x_reshape = x.view(n, c, h * w)
        topk_keep_num = max(1, int(self.topk * h * w))
        _, index = torch.topk(x_reshape.abs(), topk_keep_num, dim=2)
        mask = torch.zeros_like(x_reshape).scatter_(2, index, 1).to(self.device)
        sparse_x = mask * x_reshape
        out = sparse_x.view(n, c, h, w)

        self.activations = out
        if (self.target):
            self.loss = ((self.activations - self.targetActivations) ** 2).sum()

        return out

    def setTarget(self):
        self.targetActivations = self.activations
        self.target = True

class Model(nn.Module):
    def __init__(self, target_image_path, important_layers = [1,6,11],#,20,29],
                 dimensions = (500, 500), device = 'cuda'):
        print(hashlib.md5(open('./reconstruction.py','rb').read()).hexdigest()) # make sure we sync'd with nvidia server
        super(Model, self).__init__()
        self.dimensions = dimensions
        self.target_image = self.__openImage(target_image_path).to(device)
        self.model = nn.Sequential()
        self.topkLayers = []

        VGGLayers = list(models.vgg19(weights=VGG19_Weights.DEFAULT).features._modules.values())


        for i, layer in enumerate(VGGLayers):
            # Add the necessary layers
            if isinstance(layer, nn.MaxPool2d):
                self.model.append(nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride,
                                               padding=layer.padding, ceil_mode=False))
            else:
                self.model.append(layer)

            # Add TopK on the important layers
            if i in important_layers:
                self.topkLayers.append(TopKLayer())
                self.model.append(self.topkLayers[-1])

        self.to(device)
        # Set the target activations
        self(self.target_image)
        for t in self.topkLayers:
            t.setTarget()


    def __openImage(self, path) -> torch.Tensor:
        return transforms.ToTensor()(Image.open(path).resize(self.dimensions)).unsqueeze(0)

    def forward(self, x):
        return self.model(x)

    def loss(self):
        return torch.tensor([t.loss for t in self.topkLayers], requires_grad=True).sum()

    def analyze(self, texture_img, device="cpu"):
        synthesizer_model = nn.Sequential()
        synthesizers = []
        for i in range(max(TARGET_LAYERS) + 1):
            synthesizer_model.add_module(str(i), self.model[i])
            if i in TARGET_LAYERS:
                target_feature = synthesizer_model(texture_img).detach()
                G = gram_matrix(target_feature).detach()
                synthesizer = TextureSynthesizer(G).to(device)
                synthesizers.append(synthesizer)
                synthesizer_model.add_module(str(i) + "_synthesizer", synthesizer)
        #print(synthesizer_model)
        return synthesizer_model, synthesizers
        # Weight rescaling is not needed, as per this report https://github.com/rpetit/texture-synthesis/blob/master/report.pdf
