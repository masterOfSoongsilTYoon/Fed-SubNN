from torch import nn, stack, float16, float64, float32, int64
import torch
from torch.autograd import Variable

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GRU(nn.Module) :
   def __init__(self, max_length, input_dim,hidden_dim,output_dim,bias=True) :
       super(GRU, self).__init__()
       self.input_dim = input_dim
       self.hidden_dim = hidden_dim
       self.GRU = nn.GRU(input_dim, hidden_size=hidden_dim, batch_first=True)
       self.fc = nn.Linear(hidden_dim, output_dim)
       
       
   def forward(self, x) :
       output, _ = self.GRU(x)
       out = self.fc(output)
       return out
