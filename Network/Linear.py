from torch import nn, stack, float16
import torch


class LinearModel(nn.Module) :
   def __init__(self) :
       super(LinearModel, self).__init__()
       self.encoder = nn.Sequential(nn.Linear(3, 16),
                                    nn.ReLU(),
                                    nn.Linear(16, 8),
                                    nn.Softmax(dim=1)
                                    )
       
       
   def forward(self, x) :
       
       return self.encoder(x)