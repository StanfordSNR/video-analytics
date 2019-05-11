#!/usr/bin/env python3

import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o')
    parser.add_argument('--identity', action='store_true')
    args = parser.parse_args()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots()

    # prepare data points
    x = [1/16, 1/8, 1/4, 1/2, 1]
    ax.set_xscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(['1/16', '1/8', '1/4', '1/2', '1'])
    ax.minorticks_off()

    y = [84.68, 84.35, 83.09, 78.88, 69.01]
    ax.plot(x, y, label=r'No transform; grad approx has no effect', marker='p')

    if args.identity:
        y = [84.89, 84.75, 82.73, 78.81, 68.63]
        ax.plot(x, y, label=r'Linear transform ($I$ init); no grad approx: zero gradient', marker='o')
    else:
        y = [73.00, 72.38, 72.82, 71.26, 69.52]
        ax.plot(x, y, label=r'Linear transform (rand init); no grad approx: zero gradient', marker='o')

    if args.identity:
        y = [84.59, 84.54, 83.95, 82.97, 79.95]
        ax.plot(x, y, label=r'Linear transform ($I$ init); grad approx: poly $n=1$', marker='*')
    else:
        y = [72.19, 73.05, 72.39, 71.36, 68.59]
        ax.plot(x, y, label=r'Linear transform (rand init); grad approx: poly $n=1$', marker='*')

    if args.identity:
        y = [84.09, 83.98, 83.21, 82.55, 80.70]
        ax.plot(x, y, label=r'Linear transform ($I$ init); grad approx: poly $n=2$', marker='^')
    else:
        y = [72.44, 72.90, 71.27, 70.87, 69.98]
        ax.plot(x, y, label=r'Linear transform (rand init); grad approx: poly $n=2$', marker='^')

    if args.identity:
        y = [83.38, 83.57, 82.81, 81.13, 80.35]
        ax.plot(x, y, label=r'Linear transform ($I$ init); grad approx: poly $n=3$', marker='v')
    else:
        y = [72.84, 72.97, 71.70, 70.45, 69.73]
        ax.plot(x, y, label=r'Linear transform (rand init); grad approx: poly $n=3$', marker='v')

    if args.identity:
        y = [82.96, 82.59, 83.15, 82.28, 79.50]
        ax.plot(x, y, label=r'Linear transform ($I$ init); grad approx: Fourier $k_{\max}=1$', marker='d')
    else:
        y = [71.78, 72.99, 72.02, 70.74, 69.90]
        ax.plot(x, y, label=r'Linear transform (rand init); grad approx: Fourier $k_{\max}=1$', marker='d')

    ax.set_xlabel(r'$\Delta$')
    ax.set_ylabel(r'Accuracy (\%)')
    ax.legend(fontsize=12)
    fig.savefig(args.o)


if __name__ == '__main__':
    main()
