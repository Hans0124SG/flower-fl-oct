import flwr as fl
import numpy as np
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader
from utils import test
from simpleCNN import Net

from collections import OrderedDict

def get_eval_fn(model, DEVICE):
    """Return an evaluation function for server-side evaluation."""
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
    )
    testset = CIFAR10(".", train=False, download=True, transform=transform)
    testloader = DataLoader(testnset, batch_size=128)
    
    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights):
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        loss, accuracy = test(model, testloader, DEVICE)
        return loss, {"accuracy": accuracy}

    return evaluate

class myStrategy(fl.server.strategy.FedAvg):
    
    def aggregate_fit(self,rnd,results,failures):
        aggregated_parameters_tuple = super().aggregate_fit(rnd, results, failures)
        aggregated_parameters, _ = aggregated_parameters_tuple
        if aggregated_parameters is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            aggregated_weights = fl.common.parameters_to_weights(aggregated_parameters)

            params_dict = zip(net.state_dict().keys(), aggregated_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            torch.save(net.state_dict(), f"model_round_{rnd}.pth")    
        return aggregated_parameters_tuple
    
    def aggregate_evaluate(self,rnd,results,failures):
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model and data
# net = Net().to(DEVICE)
net = models.efficientnet_b0(pretrained=False)

# strategy = fl.server.strategy.FedAvg(
#     eval_fn=get_eval_fn(net, DEVICE),
# )
strategy = myStrategy(min_available_clients=3, min_fit_clients=3)

fl.server.start_server(config={"num_rounds": 5}, strategy=strategy, force_final_distributed_eval=True)