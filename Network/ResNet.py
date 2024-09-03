from torch import nn, stack, float16
from torchvision.models.resnet import ResNet, wide_resnet101_2, Wide_ResNet101_2_Weights
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNet152C(nn.Module) :
   def __init__(self) :
       super(ResNet152C, self).__init__()
       self.encoder = wide_resnet101_2(weights='IMAGENET1K_V2')
       self.fcl = nn.Sequential(
           nn.Linear(in_features=1000, out_features=10),
           nn.Softmax(dim=1)
    )
   def forward(self, x) :
       out = self.encoder(x)
       return self.fcl(out)
       