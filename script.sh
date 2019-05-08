#!/bin/sh

for delta in 0.0625 0.125 0.25 0.5 1.0
do
  for option in 0 1 2 3 4
  do
    echo "delta=$delta, grad_approx=$option" >> identity_result
    ./cifar10_quantize.py --transform --delta $delta --grad-approx $option >> identity_result
  done
done
