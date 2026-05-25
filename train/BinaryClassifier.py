import torch.nn as nn

class BinaryClassifier(nn.Module):

  def __init__(self):
    super(BinaryClassifier, self).__init__()

    self.block1 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3))
    self.block2 = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    return x
