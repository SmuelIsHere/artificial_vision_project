import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights


class Backbone(nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()

        self.model = resnet50(weights=ResNet50_Weights)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
    def forward(self, x):
        return self.model(x)

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_last_layers(self, num_layers):
        for param in list(self.model.parameters())[-num_layers:]:
            param.requires_grad = True
