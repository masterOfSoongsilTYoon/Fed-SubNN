from torch import nn
from torch.utils.data import DataLoader
import torch
from utils import *
from Network import *
import numpy as np
from train import valid
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
    
    net = LSTMModel(3,4,1,2)
    args = Evaluateparaser()
    net.load_state_dict(torch.load(f"./Models/{args.version}/net.pt", weights_only=False))
    net.double()
    net.to(DEVICE)
    lossf = nn.CrossEntropyLoss()
    test_ids = [id for id  in os.listdir(os.path.join(args.wesad_path, "test")) if not "Cluster" in id]
    test_data = WESADDataset(pkl_files=[os.path.join(args.wesad_path, "test", id, id+".pkl") for id in test_ids])
    test_loader=DataLoader(test_data, 1, shuffle=False, collate_fn=lambda x:x)
    
    history = evaluate(net, test_loader,lossf, DEVICE)