#!/bin/sh

# python3 main.py --arch resnet --depth 56 --save save/cifar100-resnet-56 --data cifar100
# python3 main.py --arch resnet --depth 56 --save save/cifar100+-resnet-56 --data cifar100+
python3 main.py --arch resnet --depth 110 --save save/cifar100-resnet-110 --data cifar100
python3 main.py --arch resnet --depth 110 --save save/cifar100+-resnet-110 --data cifar100+
