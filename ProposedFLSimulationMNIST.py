from ProposedServerMNIST import ProposedServer, warnings, make_model_folder, torch, np, args, DataLoader, ResNet152C, fl, pd, fl_evaluate, lossf, between
from flwr.simulation import start_simulation
from flwr.common import Context
from clientMNIST import FedAvgClient, MNistDataset, SGD, train, valid
from torch.utils.data import random_split
import random

def client_fn(context:Context):
    net = ResNet152C()
    net.double()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, _ = random_split(MNistDataset(train=True, noise=args.noise), [0.3, 0.7])
    trainloader = DataLoader(trainset, args.batch_size, shuffle=True, collate_fn=lambda x:x)
    return FedAvgClient(net.to(DEVICE), trainloader, None, args.epoch, lossf.to(DEVICE), SGD(net.parameters(), lr=1e-2), DEVICE, trainF=train, validF=valid).to_client()



if __name__=="__main__":
    warnings.filterwarnings("ignore")
    make_model_folder(f"./Models/{args.version}")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    main_model = ResNet152C()
    if args.pretrained is not None:
        main_model.load_state_dict(torch.load(args.pretrained, weights_only=True))
    history=start_simulation(
    client_fn=client_fn, # A function to run a _virtual_ client when required
    num_clients=5, # Total number of clients available
    config=fl.server.ServerConfig(num_rounds=args.round), # Specify number of FL rounds
    strategy=ProposedServer(main_model=main_model, omega=args.omega ,evaluate_fn = fl_evaluate, inplace=True, min_fit_clients=5, min_available_clients=5, min_evaluate_clients=5), # A Flower strategy
    client_resources = {"num_cpus": 1, "num_gpus": 1}
    )
    loss_frame=pd.DataFrame({"Loss": list(map(lambda x: x[1] , history.losses_centralized))})
    loss_frame.plot(color='red').figure.savefig(f"./Plot/{args.version}_loss.png")
    loss_frame.to_csv(f"./Csv/{args.version}_loss.csv", index=False)
    similar_frame = pd.DataFrame({"Model Similarity": between})
    similar_frame.to_csv(f"./Csv/{args.version}_similarity_O{args.omega :.2f}.csv", index=False)
    similar_plot = similar_frame.plot(color='red')
    similar_plot.set_xlabel("Round")
    similar_plot.set_ylabel("Main-Sub model similarity")
    similar_plot.figure.legend(fontsize="large")
    similar_plot.figure.savefig((f"./Plot/{args.version}_model_similarity.png"))