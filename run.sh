#!/bin/bash

for delta in 0.001 0.125 0.25 0.5
do
    python cifar10_quantize_tensorboard.py --grad-approx 4 --delta $delta --gpu-id 6 --init-transform 'identity' --clamp 1.0 --gradient-clip 0.05 
done
