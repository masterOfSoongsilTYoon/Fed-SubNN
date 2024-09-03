import flwr as fl
import torch
import torch.nn as nn
from torch import save
from torch.utils.data import DataLoader
from MNtrain import valid, make_model_folder
import warnings
from utils import *
from Network import *
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import flwr as fl
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import aggregate, aggregate_inplace, weighted_loss_avg
import os
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = Federatedparser()  
eval_data = MNistDataset(train=False)
eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, collate_fn= lambda x:x)

lossf = nn.CrossEntropyLoss()
net = ResNet152C()
net.double()
net.to(DEVICE)

between = []

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

def cosine_similarity_cal(X):
    cosine = [cosine_similarity(x) for x in X]
    total_distance =[c[0,1] for c in cosine]
    
    return total_distance

def parameter_to_Ndarrays(param):
    return [v.flatten() for v in param]

class ProposedServer(fl.server.strategy.FedAvg):
    def __init__(self, *, main_model:nn.Module, omega:int, fraction_fit: float = 1, fraction_evaluate: float = 1, min_fit_clients: int = 2, min_evaluate_clients: int = 2, min_available_clients: int = 2, evaluate_fn, on_fit_config_fn: Callable[[int], Dict[str, bool | bytes | float | int | str]] | None = None, on_evaluate_config_fn: Callable[[int], Dict[str, bool | bytes | float | int | str]] | None = None, accept_failures: bool = True, initial_parameters: Parameters | None = None, fit_metrics_aggregation_fn: Callable[[List[Tuple[int, Dict[str, bool | bytes | float | int | str]]]], Dict[str, bool | bytes | float | int | str]] | None = None, evaluate_metrics_aggregation_fn: Callable[[List[Tuple[int, Dict[str, bool | bytes | float | int | str]]]], Dict[str, bool | bytes | float | int | str]] | None = None, inplace: bool = True) -> None:
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        self.main_net = main_model
        self.omega = omega
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        clusters={}
        
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            # only_params = [parameters_to_ndarrays(fit_res.parameters)
            #     for _, fit_res in results]
            aggregated_ndarrays = aggregate(weights_results)
        main_ndarrays = {k:v.cpu().detach().numpy() for k, v in self.main_net.state_dict().items()}
        DV = [sp-mp for sp, mp in zip(aggregated_ndarrays, main_ndarrays.values())]
        
        similar = cosine_similarity_cal(zip(parameter_to_Ndarrays(aggregated_ndarrays), parameter_to_Ndarrays(main_ndarrays.values())))
        
        between.append(sum(similar)/len(similar))

        proposed_eq = [sp-(dv*(1-sim)*self.omega) for sp, dv, sim in zip(aggregated_ndarrays, DV, similar)]
        
        parameters_eq = ndarrays_to_parameters(proposed_eq)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_eq, metrics_aggregated
          


if __name__=="__main__":
    warnings.filterwarnings("ignore")
    make_model_folder(f"./Models/{args.version}")
    main_model = ResNet152C()
    if args.pretrained is not None:
        main_model.load_state_dict(torch.load(args.pretrained, weights_only=True))
    if args.test:
        history=fl.server.start_server(server_address="[::]:8084", grpc_max_message_length=1024*1024*1024, strategy=ProposedServer(main_model=main_model, omega=args.omega ,evaluate_fn = fl_save, inplace=True, min_fit_clients=1, min_available_clients=1, min_evaluate_clients=1), 
                        config=fl.server.ServerConfig(num_rounds=args.round))
    else:
        history=fl.server.start_server(server_address="[::]:8084", grpc_max_message_length=1024*1024*1024, strategy=ProposedServer(main_model=main_model, omega=args.omega ,evaluate_fn = fl_evaluate, inplace=True, min_fit_clients=5, min_available_clients=5, min_evaluate_clients=5), 
                        config=fl.server.ServerConfig(num_rounds=args.round))
    loss_frame=pd.DataFrame({"Loss": list(map(lambda x: x[1] , history.losses_centralized))})
    loss_frame.plot(color='red').figure.savefig(f"./Plot/{args.version}_loss.png")
    loss_frame.to_csv(f"./Csv/{args.version}_loss.csv", index=False)
    similar_frame = pd.DataFrame({"Model Similarity": between})
    similar_frame.to_csv(f"./Csv/{args.version}_similarity_O{args.omega :.2f}.csv", index=False)
    similar_plot = similar_frame.plot(color='red')
    similar_plot.set_xlabel("Round")
    similar_plot.set_ylabel("Main-Sub model similarity")
    similar_plot.set_ylim(0.7, 1.0)
    similar_plot.figure.legend(fontsize="x-large")
    similar_plot.figure.savefig((f"./Plot/{args.version}_model_similarity.png"))