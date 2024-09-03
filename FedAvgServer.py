from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
import torch
import torch.nn as nn
from torch import save
from torch.utils.data import DataLoader
from train import valid, make_model_folder
import warnings
from utils import *
from Network import *

import os
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = Federatedparser()
if args.wesad_path !=None:
        eval_ids = os.listdir(os.path.join(args.wesad_path, "valid"))
        eval_data = WESADDataset(pkl_files=[os.path.join(args.wesad_path, "valid", id, id+".pkl") for id in eval_ids], test_mode=args.test)
        eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, collate_fn= lambda x:x)

lossf = nn.CrossEntropyLoss()
net = LSTMModel(3, 4, 1, 2)
net.double()
if args.pretrained is not None:
        net.load_state_dict(torch.load(args.pretrained))
net.to(DEVICE)

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def fl_evaluate(server_round:int, parameters: fl.common.NDArrays, config:Dict[str, fl.common.Scalar], validF=valid)-> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    set_parameters(net, parameters)
    history=validF(net, eval_loader, None, lossf, DEVICE)
    save(net.state_dict(), f"./Models/{args.version}/net.pt")
    print(f"Server-side evaluation loss {history['loss']} / accuracy {history['acc']} / precision {history['precision']} / f1score {history['f1score']}")
    return history['loss'], {key:value for key, value in history.items() if key != "loss" }

def fl_save(server_round:int, parameters: fl.common.NDArrays, config:Dict[str, fl.common.Scalar], validF=valid)-> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    set_parameters(net, parameters)
    save(net.state_dict(), f"./Models/{args.version}/net.pt")
    print("model is saved")
    return 0, {}
if __name__=="__main__":
    warnings.filterwarnings("ignore")
    make_model_folder(f"./Models/{args.version}")
    history=fl.server.start_server(server_address="[::]:8084",strategy=fl.server.strategy.FedAvg(evaluate_fn = fl_evaluate,inplace=False, min_fit_clients=4, min_available_clients=4, min_evaluate_clients=4), 
                           config=fl.server.ServerConfig(num_rounds=args.round))
    
    plt=pd.DataFrame(history.losses_centralized)
    plt.plot().figure.savefig(f"./Plot/{args.version}_loss.png")
    plt.to_csv(f"./Csv/{args.version}_loss.csv", index=False)