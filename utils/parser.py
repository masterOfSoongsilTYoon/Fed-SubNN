import argparse

def Centralparser():
    parser= argparse.ArgumentParser(
        prog="Central Learning in WESAD",
        description="centralized training code by using WESAD Dataset",
    )
    parser.add_argument("-v", "--version", type= str, default="default")
    parser.add_argument("-s", "--seed", type= int, default= 2024)
    parser.add_argument("-e", "--epoch", type= int, default= 10)
    parser.add_argument("-bs", "--batch-size", type= int, default=1)
    parser.add_argument("-l", "--lr", type= float, default= 1e-2)
    parser.add_argument("-w", "--wesad_path", type= str, default=None)
    parser.add_argument("-p", "--pretrained", type= str, default=None)
    args = parser.parse_args()
    return args

def Federatedparser():
    parser= argparse.ArgumentParser(
        prog="Federated Learning in WESAD",
        description="Federated Learning code by using WESAD Dataset",
    )
    parser.add_argument("-v", "--version", type= str, default="default")
    parser.add_argument("-s", "--seed", type= int, default= 2024)
    parser.add_argument("-r", "--round", type= int, default=10)
    parser.add_argument("-e", "--epoch", type= int, default= 3)
    parser.add_argument("-i", "--id", type= int, default=1)
    parser.add_argument("-bs", "--batch-size", type= int, default=1)
    parser.add_argument("-p", "--pretrained", type= str, default=None)
    parser.add_argument("-w", "--wesad_path", type= str, default=None)
    parser.add_argument("-t", "--test", type= bool, default=False)
    parser.add_argument("-m", "--mode", type= str, default="max")
    parser.add_argument("-o", "--omega", type= float, default= 0.5)
    args = parser.parse_args()
    return args

def Evaluateparaser():
    parser= argparse.ArgumentParser(
        prog="Evaluate model in WESAD",
        description="evaluate code by using WESAD Dataset",
    )
    parser.add_argument("-v", "--version", type= str, default="default")
    parser.add_argument("-w", "--wesad_path", type= str, required=False)
    parser.add_argument("-bs", "--batch-size", type= int, default=1)
    args = parser.parse_args()
    return args