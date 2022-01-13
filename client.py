import torch
import argparse

from utils import *
from simpleCNN import Net
from config import CifarClient

import flwr as fl

import time

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=str,
                    help='specify the gpu that you want to use')
parser.add_argument('--epochs', default=2, type=int,
                    help='specify the epochs that you want to train for')

args = parser.parse_args()

DEVICE = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

# Load model and data
net = Net().to(DEVICE)
trainloader, testloader, num_examples = load_data()
client=CifarClient(net, trainloader, testloader, num_examples, args.epochs, DEVICE)

fl.client.start_numpy_client("localhost:8080", client)
time.sleep(5)