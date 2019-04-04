#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = (img + 1.0) / 2.0  # unnormalize
    npimg = img.numpy()
    plt.imshow(img, cmap='gray', interpolation='None')
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--load-model', required=True,
                        help='Load a classifier for MNIST')
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

    for q in range(400, 520, 10):
        print('Quantizer:', q)
        test(model, device, test_loader, q)


if __name__ == '__main__':
    main()
