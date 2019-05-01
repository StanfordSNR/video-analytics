#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models import *


delta = None
transform = False
device = None


class Floor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Transform(nn.Module):
    def __init__(self):
        super(Transform, self).__init__()
        self.linear = nn.Linear(8*8, 8*8)

    def forward(self, x):
        # divide each 32*32 image into 4*4 macroblocks of size 8*8
        for i in range(4):
            for j in range(4):
                mb = x[:, :, 8*i:8*(i+1), 8*j:8*(j+1)].clone()
                mb = mb.view(x.shape[0], x.shape[1], 8*8)
                mb = self.linear(mb)
                mb = mb.view(x.shape[0], x.shape[1], 8, 8)
                x[:, :, 8*i:8*(i+1), 8*j:8*(j+1)] = mb

        return x


class QuantizeNet(nn.Module):
    def __init__(self, classifier):
        super(QuantizeNet, self).__init__()
        if delta is None:
            self.delta = nn.Parameter(1 - torch.rand(1))  # (0, 1]
        else:
            self.delta = delta

        self.transform = Transform()  # custom transform
        self.floor = Floor.apply  # custom floor function
        self.classifier = classifier  # original classifier on CIFAR-10

    def forward(self, x):
        if transform:
            x = self.transform(x)
        x = self.floor(x / self.delta)
        x = self.delta * (x + 0.5)
        return self.classifier(x)


def imshow(img):
    img = (img + 1.0) / 2.0  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(args, model, train_loader, criterion, optimizer, epoch):
    model.train()

    batch_idx = 0
    total_data_cnt = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        batch_idx += 1
        total_data_cnt += len(data)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx % args.log_interval == 0 or
                total_data_cnt == len(train_loader.dataset)):
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                  epoch, total_data_cnt, len(train_loader.dataset),
                  100.0 * total_data_cnt / len(train_loader.dataset),
                  loss.item()))
            if delta is None:
                print('Delta: {:.3f}, Gradient: {:.3f}'.format(
                      model.delta.item(), model.delta.grad.item()))


def test(args, model, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
          test_loss, correct, len(test_loader.dataset),
          100.0 * correct / len(test_loader.dataset)))


def main():
    # training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='batches to wait before logging training status')
    parser.add_argument('--delta', type=float,
                        help='set a fixed step size used in quantizer')
    parser.add_argument('--transform', action='store_true',
                        help='add a transform layer')
    parser.add_argument('--save-model', help='save the trained model')
    parser.add_argument('--load-model', help='load a trained model')
    args = parser.parse_args()

    global delta
    if args.delta:
        delta = args.delta

    global transform
    if args.transform:
        transform = args.transform

    # use CUDA if available
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    global device
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # load CIFAR-10 dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # initialize model
    model = QuantizeNet(ResNet18())
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # inference only
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
        test(args, model, test_loader, criterion)
        return

    # training
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, criterion, optimizer, epoch)
        test(args, model, test_loader, criterion)

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)


if __name__ == '__main__':
    main()
