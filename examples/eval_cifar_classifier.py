#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def quantize(x, q):
    y = 255.0 * ((x + 1.0) / 2.0)  # unnormalize to [0, 255]
    y = torch.clamp(torch.round(y / q) * q, min=0.0, max=255.0)  # quantization
    y = 2.0 * y / 255.0 - 1.0  # normalize

    return y


def test(model, device, test_loader, q):
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            quantized_data = quantize(data, q)

            quantized_data, target = quantized_data.to(device), target.to(device)
            output = model(quantized_data)

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
          correct, len(test_loader.dataset),
          100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--load-model', required=True,
                        help='Load a classifier for CIFAR10')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    model = Net().to(device)
    model.load_state_dict(torch.load(args.load_model))
    model.eval()

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=1000, shuffle=False, **kwargs)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                             shuffle=False, num_workers=2)

    for q in range(400, 520, 10):
        print('Quantizer:', q)
        test(model, device, test_loader, q)


if __name__ == '__main__':
    main()
