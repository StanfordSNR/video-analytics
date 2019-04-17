#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

from models import *

def imshow(img):
    img = (img + 1.0) / 2.0  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, criterion):
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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset),
          100. * correct / len(test_loader.dataset)))


def main():
    # training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', help='save the current model')
    parser.add_argument('--load-model', help='load a trained model')
    args = parser.parse_args()

    # whether CUDA is available
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device('cuda:0' if use_cuda else 'cpu')

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

    model = ResNet18()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # inference only
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
        test(args, model, device, test_loader, criterion)
        return

    # training
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        test(args, model, device, test_loader, criterion)

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)


if __name__ == '__main__':
    main()
