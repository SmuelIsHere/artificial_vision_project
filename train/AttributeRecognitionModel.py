import torch.nn as nn
from BinaryClassifier import BinaryClassifier
from backbone import Backbone
from AttentionModule import AttentionModule

class AttributeRecognitionModel(nn.Module):

    def __init__(self, num_attributes):
        super(AttributeRecognitionModel, self).__init__()

        self.backbone = Backbone()
        self.attention_modules = nn.ModuleList([AttentionModule(in_channels=2048) for _ in range(num_attributes)])
        binary_classifier = [BinaryClassifier() for _ in range(3)]
        self.classifiers = nn.ModuleList(binary_classifier)

    def forward(self, x):
        features = self.backbone(x)
        pred_list=[]
        attention_outputs = [attention(features) for attention in self.attention_modules]

        for att_output, classifier in zip(attention_outputs, self.classifiers):
            flattened_output = att_output.view(att_output.size(0), -1)
            pred = classifier(flattened_output)
            pred_list.append(pred)

        return pred_list

    def freeze_backbone_parameters(self):
      self.backbone.freeze_all()

    def unfreeze_parameters(self):
        for param in self.attention_modules.parameters():
            param.requires_grad = True

        for param in self.classifiers.parameters():
            param.requires_grad = True

    def unfreeze_last_layer_backbone(self, num_layers):
        self.backbone.unfreeze_last_layers(num_layers)
