from torch import nn,Tensor, stack, int64,float32, float64, argmax, save
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from utils import *
from Network import *
import Network
import numpy as np
import warnings
import random
import os
from tqdm import tqdm
def make_model_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        pass

def train(net, train_loader, valid_loader, epoch, lossf, optimizer, DEVICE, save_path):
    history = {'loss': [], 'acc': [], 'precision': [], 'f1score': [], "recall": []}
    for e in range(epoch):
        net.train()
        for sample in tqdm(train_loader,desc="Train: "):
            # print(f"{b+1} batch start")
            X= torch.stack([s["x"] for s in sample], 0)
            Y= torch.IntTensor([s["label"] for s in sample])
            
            out = net(X.type(float64).to(DEVICE))
            # print(out.size())
            loss = lossf(out.type(float32).to(DEVICE), Y.type(int64).to(DEVICE))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if valid_loader is not None:
            net.eval()
            # print("valid start")
            with torch.no_grad():
                for key, value in valid(net, valid_loader, e, lossf, DEVICE).items():
                    history[key].append(value)
        if save_path is not None:            
            save(net.state_dict(), f"./Models/{save_path}/net.pt")
    if valid_loader is not None:                    
        return history
    else:
        return None
    
def valid(net, valid_loader, e, lossf, DEVICE):
    acc=0
    precision=0
    f1score=0
    recall=0
    length = len(valid_loader)+1e-7
    for sample in tqdm(valid_loader, desc="Validation: "):
        # print(f"valid {b+1} batch start")
        X= torch.stack([s["x"] for s in sample], 0)
        Y= torch.IntTensor([s["label"] for s in sample])
        
        out = net(X.type(float64).to(DEVICE))
        loss = lossf(out.type(float32).to(DEVICE), Y.type(int64).to(DEVICE))
        
        out = argmax(out, dim=-1)
        acc+= accuracy_score(Y.cpu().squeeze().detach().numpy(), out.cpu().squeeze().detach().numpy())
        precision+= precision_score(Y.cpu().squeeze().detach().numpy(), out.cpu().squeeze().detach().numpy(), average='macro')
        f1score += f1_score(Y.cpu().squeeze().detach().numpy(), out.cpu().squeeze().detach().numpy(), average="macro")
        recall += recall_score(Y.cpu().squeeze().detach().numpy(), out.cpu().squeeze().detach().numpy(), average="macro")
    if e is not None:
        print(f"Result epoch {e+1}: loss:{loss.item(): .4f} acc:{acc/length: .4f} precision:{precision/length: .4f} f1score:{f1score/length: .4f} recall: {recall/length: .4f}")
        
    return {'loss': loss.item(), 'acc': acc/length, 'precision': precision/length, 'f1score': f1score/length, "recall": recall/length}

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    print("==== Centralized Learning by using WESAD Dataset ====")
    args = Centralparser()
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    make_model_folder(f"./Models/{args.version}")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    
    lossf = nn.CrossEntropyLoss()
    lossf.to(DEVICE)
    net = Network.ResNet152C()
    if args.pretrained is not None:
        net.load_state_dict(torch.load(args.pretrained, weights_only=True))
    net.double()
    net.to(DEVICE)
    optimizer = SGD(net.parameters(), lr=args.lr)
    
    print("==== LSTM 모델 ====")
    print(net)
    
    print("==== Loss ====")
    print(lossf.__class__.__name__)
    print("==== Args ====")
    print(f"seed value: {args.seed}")
    print(f"epoch number: {args.epoch}")
    train_data = MNistDataset(train=True)
    valid_data = MNistDataset(train=False)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn= lambda x:x)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn= lambda x:x)
    print("==== Training ====")
    history = train(net, train_loader, valid_loader, args.epoch, lossf, optimizer, DEVICE, args.version)