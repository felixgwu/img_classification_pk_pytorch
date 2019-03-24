#!/bin/sh

# python3 main.py --arch resnet --depth 56 --save save/cifar100-resnet-56 --data cifar100
# python3 main.py --arch resnet --depth 56 --save save/cifar100 --data_aug-resnet-56 --data cifar100 --data_aug
# python3 main.py --arch resnet --depth 110 --save save/cifar100-resnet-110 --data cifar100
# python3 main.py --arch resnet --depth 110 --save save/cifar100 --data_aug-resnet-110 --data cifar100 --data_aug

# python3 main.py --arch densenet --depth 100 --bn-size 4 --compression 0.5 --data cifar100 --epochs 300 --save save/cifar100-densenet-bc-100 
# python3 main.py --arch densenet --depth 100 --bn-size 4 --compression 0.5 --data cifar100 --data_aug --epochs 300 --save save/cifar100 --data_aug-densenet-bc-100 
