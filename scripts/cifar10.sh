#!/bin/sh

python3 main.py --arch resnet --depth 56 --save save/resnet-56 --data cifar10
python3 main.py --arch resnet --depth 110 --save save/resnet-110 --data cifar10
python3 main.py --arch resnet --depth 56 --save save/resnet-56 --data cifar10 --no_aug
python3 main.py --arch resnet --depth 110 --save save/resnet-110 --data cifar10 --no_aug
