#!/bin/sh

# ResNet
# python3 main.py --arch resnet --depth 56 --save save/cifar10-resnet-56 --data cifar10 --epochs 164
# python3 main.py --arch resnet --depth 56 --save save/cifar10+-resnet-56 --data cifar10+ --epochs 164
# python3 main.py --arch resnet --depth 110 --save save/cifar10-resnet-110 --data cifar10 --epochs 164
# python3 main.py --arch resnet --depth 110 --save save/cifar10+-resnet-110 --data cifar10+ --epochs 164

python3 main.py --arch densent --depth 100 --bn_size 4 --compression 0.5 --data cifar10+ --epochs 300 --save save/cifar10+-densenet-bc-100 
