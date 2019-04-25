#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.optim as optim


class Floor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        # zero (correct gradient)
        grad_input *= 0

        # linear
        # grad_input *= 1

        # quadratic
        # grad_input *= 2 * (input - torch.floor(input))

        # cubic
        # grad_input *= 3 * (input - torch.floor(input)) ** 2

        # Fourier series expansion
        # grad_input *= 1 + 2 * torch.cos(2 * math.pi * input)

        # log
        grad_input *= 1.0 / (input + 1.0)

        return grad_input


class Quantizer(nn.Module):
    def __init__(self):
        super(Quantizer, self).__init__()
        self.delta = nn.Parameter(torch.tensor(1.0))
        self.floor = Floor.apply

    def forward(self, x):
        encoded = self.floor(x / self.delta)
        decoded = self.delta * (encoded + 0.5)
        return decoded


def main():
    batch_size = 128
    lr = 0.01
    epochs = 100000

    # whether CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # training
    model = Quantizer().to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        data = 2 * torch.rand(batch_size, 1) - 1
        data = data.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('Epoch: {}, Loss: {:.6f}'.format(epoch, loss.item()))
            print(model.delta, model.delta.grad)

            if abs(model.delta.item()) < 0.0001:
                print('Done {} epochs'.format(epoch))
                return


if __name__ == '__main__':
    main()
