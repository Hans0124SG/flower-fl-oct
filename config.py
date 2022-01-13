import flwr as fl
from utils import *

from collections import OrderedDict

class CifarClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader, num_examples, epochs, DEVICE):
        super()
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples
        self.epochs = epochs
        self.DEVICE = DEVICE
        
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = train(self.net, self.trainloader, self.epochs, self.DEVICE)
        return self.get_parameters(), self.num_examples["trainset"], {"accuracy": float(accuracy)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader, self.DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}
