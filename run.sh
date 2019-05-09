#!/bin/bash

for approx in 2 3 1
do
    python cifar10_quantize_tensorboard.py --grad-approx $approx --gradient-clip 0.05
done
