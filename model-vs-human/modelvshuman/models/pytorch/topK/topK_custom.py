import torch
import torch.nn as nn
from torchvision import models

class TopKLayer(nn.Module):
    def __init__(self, topk=0.1, revert=False):
        super(TopKLayer, self).__init__()
        self.revert=revert
        self.topk=topk

    def sparse_hw(self, x, tau, topk, device='cpu'):
        n, c, h, w = x.size()
        topk_keep_num = int(max(1, topk * h * w))
        
        # Reshape to 2D tensor (batch_size * channels, height * width)
        x_reshaped = x.view(n * c, h * w)

        # Calculate the absolute values
        abs_x = torch.abs(x_reshaped)

        # Calculate ranks within each row
        ranks = abs_x.argsort(dim=1).argsort(dim=1)

        # Find the K-th largest absolute value in each row
        threshold_indices = torch.topk(abs_x, topk_keep_num, dim=1)[1][:, -1]

        # Create a mask based on the comparison with the threshold
        mask = ranks < threshold_indices.unsqueeze(1)

        # Reshape the mask back to the original shape
        mask = mask.view(n, c, h, w)

        # Apply the mask to the original tensor
        sparse_x = x * mask.float()

        return sparse_x

    def forward(self, x):
        return self.sparse_hw(x,1,self.topk)


def topK_AlexNet(pretrain_weigth,topk, **kwargs):
    if pretrain_weigth=="":
        alexnet = models.alexnet(pretrained=True)
        new_features = nn.Sequential(
            # layers up to the point of insertion
            *(list(alexnet.features.children())[:3]), 
            TopKLayer(topk),
            *(list(alexnet.features.children())[3:6]),
            TopKLayer(topk),
            *(list(alexnet.features.children())[6:8]),
            TopKLayer(topk),
            *(list(alexnet.features.children())[8:10]),
            TopKLayer(topk),
            *(list(alexnet.features.children())[10:]),
            TopKLayer(topk),
        )
        alexnet.features = new_features
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alexnet = alexnet.to(device)
        alexnet = torch.nn.DataParallel(alexnet)
        return alexnet