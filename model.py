'''
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
'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchbearer
import torch.optim as optim
from torchvision import models
import torchvision.transforms as transforms

from torchvision.models.vgg import VGG19_Weights
import torch.nn as nn

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [1,1,1]

def gram_matrix(input):
    b, c, h, w = input.size() # b=batch size, c=number of channels, (h, w)=dimensions of a feature map
    F = input.view(b, c, h * w) # Reshape feature matrix
    G = torch.bmm(F, F.transpose(1, 2)) / (h * w)
    return G

# Contribution of layer l to the total loss
def El (G, G_hat):
    return 1e9*torch.mean((G - G_hat)**2)
# Total loss
def L(layers, weights, Gs):
    loss = 0
    for i in range(len(layers)):
        loss += weights[i] * El(Gs[i], gram_matrix(layers[i]))
    return loss
# derivative of El w.r.t. G_hat
# def dEl(G, G_hat):
#     b, c, h, w = G.size()
#     return (G - G_hat) / (b * c * h * w)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # reshape the mean as : [C x 1 x 1] so that it is compatible with image tensors of shape [B x C x H x W]
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class TextureSynthesizer(nn.Module):
    def __init__(self, target_gram):
        super(TextureSynthesizer, self).__init__()
        self.target = target_gram
    def forward(self, input):
        G = gram_matrix(input)
        #print(G.size())
        self.loss =  El(self.target, G)
        return input
    
def analyze(model, texture_img, target_layers=[1,6]): # 1,6,11,20,29
    new_model = nn.Sequential(Normalization(IMAGENET_MEAN, IMAGENET_STD))
    #new_model = nn.Sequential()
    synthesizers = []
    for i in range(len(model)):
        if i > target_layers[-1]:
            break
        new_model.add_module(str(i), model[i])
        if i in target_layers:
            target_feature = new_model(texture_img).detach()
            #print(target_feature.size())
            G = gram_matrix(target_feature).detach()
            #print(G.size())
            synthesizer = TextureSynthesizer(G)
            synthesizers.append(synthesizer)
            new_model.add_module(str(i) + "_synthesizer", synthesizer)

    return new_model, synthesizers
        
    

class Vgg19Avg(nn.Module):
    def __init__(self):
        super(Vgg19Avg, self).__init__()
        self.model = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        # Replace maxpooling layers with average pooling
        #self.synthesizer = TextureSynthesizer()
        for i in range(len(self.model)):
            if isinstance(self.model[i], nn.MaxPool2d):
                self.model[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)


    def analyze(self, texture_img):
        synthesizer_model, synthesizers = analyze(self.model,texture_img)
        return synthesizer_model, synthesizers
        # Weight rescaling is not needed, as per this report https://github.com/rpetit/texture-synthesis/blob/master/report.pdf


    
def run_texture_synthesis(texture_img):
    Vgg19Avg()
    synthesizer_model, synthesizers =  Vgg19Avg().analyze(texture_img)
    synthesized_img = torch.rand(1,3,100,100)
    #rescale = transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
    #synthesized_img = rescale(synthesized_img)
    optimizer = optim.LBFGS([synthesized_img.requires_grad_(True)])

    run = [0]
    while run[0] < 2000:
        def closure():
            optimizer.zero_grad()
            synthesizer_model(synthesized_img)
            texture_loss = 0
            for s in synthesizers:
                texture_loss += s.loss
            loss = texture_loss
            loss.backward()
            run[0] += 1
            if run[0] % 10 == 0:
                print("run {}:".format(run))
                print('Loss : {:4f}'.format(loss.item()))
                print()
            return texture_loss
        optimizer.step(closure)
        
    print(synthesized_img.size())
    return synthesized_img

texture = 'img.jpg'
texture_image_orig = Image.open(texture)
texture_image_orig = texture_image_orig.resize((100, 100))
to_tensor = transforms.ToTensor()

texture_image = to_tensor(texture_image_orig)
#change the tensor to 4D with batch size 1
texture_image = texture_image.unsqueeze(0)


synthesized_image = run_texture_synthesis(texture_image)
synthesized_image = synthesized_image.detach().squeeze(0).numpy().transpose(1, 2, 0)
synthesized_image = np.clip(synthesized_image, 0, 1)
# Display the original and synthesized images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.imshow(texture_image_orig)
ax1.axis('off')
ax1.set_title('original image')
ax2.imshow(synthesized_image)
ax2.axis('off')
ax2.set_title('synthesized image')
plt.show()
