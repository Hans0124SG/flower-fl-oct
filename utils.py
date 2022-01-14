import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset

def load_data(train_start_idx, train_end_idx, test_start_idx, test_end_idx):
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(".", train=True, download=True, transform=transform)
    trainset = Subset(trainset, range(train_start_idx, train_end_idx))      
    testset = CIFAR10(".", train=False, download=True, transform=transform)
    testset = Subset(testset, range(test_start_idx, test_end_idx))
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True)
    testloader = DataLoader(testset, batch_size=256)
    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples
    
def train(net, trainloader, epochs, DEVICE):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for i in range(epochs):
        correct, total, train_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / total
        accuracy = correct / total
        print("Epoch {0}: train_loss is {1}, accuracy is {2}".format(i, train_loss, accuracy))
    return loss, accuracy
            
def test(net, testloader, DEVICE):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item() / labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy
    
    