import flwr as fl
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import test
from simpleCNN import Net

from collections import OrderedDict

def get_eval_fn(model, DEVICE):
    """Return an evaluation function for server-side evaluation."""
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    testset = CIFAR10(".", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128)
    
    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights):
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        loss, accuracy = test(model, testloader, DEVICE)
        return loss, {"accuracy": accuracy}

    return evaluate

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model and data
net = Net().to(DEVICE)

strategy = fl.server.strategy.FedAvg(
    eval_fn=get_eval_fn(net, DEVICE),
)

fl.server.start_server(config={"num_rounds": 5}, strategy=strategy)