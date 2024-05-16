from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchvision import models
import torchvision.transforms as transforms
from torchvision.models.alexnet import AlexNet_Weights
from torchvision.models.vgg import VGG19_Weights
from torchvision.models.vgg import VGG16_Weights
import torch.nn as nn
import hashlib

class ActivationHook:
    def __init__(self, name=None):
        self.name = name

    def __call__(self, module, input, output):
        self.activations = output

class TopKLayer(nn.Module):
    def __init__(self, topk = 0.2, device = 'cuda', mode='topk'):
        super().__init__()
        self.topk = topk
        self.device = device
        self.targetActivations = torch.zeros(1).to(device)
        self.loss = 0
        self.mode = mode

    def forward(self, x):
        n, c, h, w = x.shape

        x_reshape = x.view(n, c, h * w)

        if not self.forward and self.targetSet:
            out = x_reshape.view(n, c, h, w)
        elif self.mode == 'topk':
            topk_keep_num = max(1, int(self.topk * h * w))
            _, index = torch.topk(x_reshape.abs(), topk_keep_num, dim=2)
            mask = torch.zeros_like(x_reshape).scatter_(2, index, 1).to(self.device)
            sparse_x = mask * x_reshape
            out = sparse_x.view(n, c, h, w)
        elif self.mode == 'non-topk':
            topk_keep_num = max(1, int(self.topk * h * w))
            _, index = torch.topk(x_reshape.abs(), topk_keep_num, dim=2)
            mask = torch.zeros_like(x_reshape).scatter_(2, index, 1).to(self.device)
            mask = torch.ones_like(mask) - mask
            sparse_x = mask * x_reshape
            out = sparse_x.view(n, c, h, w)
        elif self.mode == 'both':
            out = x_reshape.view(n, c, h, w)
        else:
            raise NotImplemented()

        self.loss = ((out - self.targetActivations) ** 2).sum()

        return out

    def setTarget(self, target):
        self.targetActivations = target
        self.targetSet = True

    def __repr__(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'TopK({self.topk})'

class Model(nn.Module):
    def __init__(self, target_image_path, important_layers = [1,6,11,20,29], topk = 0.2,
                 dimensions = (500, 500), device = 'cuda', mode='topk'):
        super(Model, self).__init__()
        self.dimensions = dimensions
        self.target_image = self.__openImage(target_image_path).to(device)
        self.model = nn.Sequential()
        self.topkLayers = []


        self.device = device

        VGGLayers = list(models.vgg16(weights=VGG16_Weights.DEFAULT).features._modules.values())
        actHooks = []
        actHandles = []
        for i, layer in enumerate(VGGLayers):
            # Add the necessary layers
            if isinstance(layer, nn.MaxPool2d):
                self.model.append(nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride,
                                               padding=layer.padding, ceil_mode=False))
            else:
                self.model.append(layer)


            # Add TopK on the important layers
            if i in important_layers:
                topkLayer = TopKLayer(topk=topk, mode=mode)
                ah = ActivationHook()
                actHandles.append(topkLayer.register_forward_hook(ah))

                actHooks.append(ah)
                self.topkLayers.append(topkLayer)
                self.model.append(topkLayer)


        self.to(device)
        # Set the target activations
        self.eval()
        self(self.target_image)
        for t in self.topkLayers:
            target = actHooks.pop(0).activations
            actHandles.pop(0).remove()
            t.setTarget(target)
        #self.train()


    def __loadTargetActivations(self, layers):
        l = []
        for i in range(len(layers)):
            l.append(torch.load(f'topk{i}'))
        return l


    def __openImage(self, path) -> torch.Tensor:
        return transforms.ToTensor()(Image.open(path).resize(self.dimensions)).unsqueeze(0)

    def forward(self, x):
        return self.model(x)

    def loss(self):
        l = torch.zeros(1, device=self.device)
        for t in self.topkLayers:
            l += t.loss
        return l


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
