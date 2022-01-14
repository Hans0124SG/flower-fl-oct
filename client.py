import torch
import argparse

from utils import *
from simpleCNN import Net
import torchvision.models as models
from config import CifarClient

import flwr as fl

import time

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=str,
                    help='specify the gpu that you want to use')
parser.add_argument('--epochs', default=2, type=int,
                    help='specify the epochs that you want to train for')
parser.add_argument('--train_start_idx', default=0, type=int,
                    help='specify the start_idx for the train set')
parser.add_argument('--train_end_idx', default=0, type=int,
                    help='specify the end_idx for the train set')
parser.add_argument('--test_start_idx', default=0, type=int,
                    help='specify the start_idx for the test set')
parser.add_argument('--test_end_idx', default=0, type=int,
                    help='specify the end_idx for the test set')

args = parser.parse_args()

DEVICE = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

# Load model and data
# net = Net().to(DEVICE)
net = models.efficientnet_b0(pretrained=True).to(DEVICE)

trainloader, testloader, num_examples = load_data(args.train_start_idx, args.train_end_idx, args.test_start_idx, args.test_end_idx)
client=CifarClient(net, trainloader, testloader, num_examples, args.epochs, DEVICE)

fl.client.start_numpy_client("localhost:8080", client)