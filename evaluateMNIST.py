from torch import nn
from torch.utils.data import DataLoader
import torch
from utils import *
from Network import *
import numpy as np
from MNtrain import valid
import os
import warnings
def evaluate(net, testloader, lossf, DEVICE):
    net.eval()
    history = {'loss': [], 'acc': [], 'precision': [], 'f1score': [], "recall": []}
    with torch.no_grad():
        for key, value in valid(net, testloader, 0, lossf, DEVICE).items():
            history[key].append(value)
    return history

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = ResNet152C()
    args = Evaluateparaser()
    net.load_state_dict(torch.load(f"./Models/{args.version}/net.pt", weights_only=False))
    net.double()
    net.to(DEVICE)
    lossf = nn.CrossEntropyLoss()
    
    test_data = MNistDataset(train=False)
    test_loader=DataLoader(test_data, args.batch_size, shuffle=False, collate_fn=lambda x:x)
    
    history = evaluate(net, test_loader,lossf, DEVICE)