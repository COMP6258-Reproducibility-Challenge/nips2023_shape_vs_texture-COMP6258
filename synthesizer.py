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

TARGET_LAYERS = [1,6,11,20,29] # Uncomment if using VGG
#TARGET_LAYERS = [1,4,7] # Uncomment if using AlexNet
IMAGE_SIZE = [100,100]
EPHOCS = 1000

def gram_matrix(input):
    b, c, h, w = input.size() # b=batch size, c=number of channels, (h, w)=dimensions of a feature map
    F = input.view(b, c, h * w) # Reshape feature matrix
    G = torch.bmm(F, F.transpose(1, 2)) / (h * w)
    return G

# Contribution of layer l to the total loss
def El (G, G_hat):
    return 1e9*torch.mean((G - G_hat)**2)

class TextureSynthesizer(nn.Module):
    def __init__(self, target_gram):
        super(TextureSynthesizer, self).__init__()
        self.target = target_gram
    def forward(self, input):
        G = gram_matrix(input)
        self.loss =  El(self.target, G)
        return input
        
    

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = models.vgg19(weights=VGG19_Weights.DEFAULT).features # Uncomment if using VGG
        #self.model = models.alexnet(weights=AlexNet_Weights.DEFAULT).features # Uncomment if using AlexNet
        # Replace maxpooling layers with average pooling
        for i in range(len(self.model)):
            if isinstance(self.model[i], nn.MaxPool2d):
                self.model[i] = nn.AvgPool2d(kernel_size=self.model[i].kernel_size, stride=self.model[i].stride, padding=self.model[i].padding, ceil_mode=False)

    def analyze(self, texture_img,device="cpu"):
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
        print(synthesizer_model)
        return synthesizer_model, synthesizers
        # Weight rescaling is not needed, as per this report https://github.com/rpetit/texture-synthesis/blob/master/report.pdf


    
def run_texture_synthesis(texture_img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
    model = Model().to(device)
    texture_img = texture_img.to(device)
    synthesizer_model, synthesizers =  model.analyze(texture_img,device)
    synthesized_img = torch.rand(1,3,IMAGE_SIZE[0],IMAGE_SIZE[1]).to(device)
    optimizer = optim.LBFGS([synthesized_img.requires_grad_(True)])
    run = [0]
    while run[0] < EPHOCS:
        def closure():
            optimizer.zero_grad()
            synthesizer_model(synthesized_img)
            texture_loss = 0
            for s in synthesizers:
                texture_loss += s.loss
            texture_loss.backward()
            run[0] += 1
            if run[0] % (EPHOCS / 10) == 0:
                print("run {}:".format(run))
                print('Loss : {:4f}'.format(texture_loss.item()))
                print()
            return texture_loss
        optimizer.step(closure)
        
    print(synthesized_img.size())
    return synthesized_img

'''
    VGG: 
    Sequential.forward of Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace=True)
    (18): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): ReLU(inplace=True)
    (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (26): ReLU(inplace=True)
    (27): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): ReLU(inplace=True)
    (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (33): ReLU(inplace=True)
    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): ReLU(inplace=True)
    (36): AvgPool2d(kernel_size=2, stride=2, padding=0)
    AlexNet
    Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
'''