#!/bin/sh

# ResNet
# python3 main.py --arch resnet --depth 56 --save save/cifar10-resnet-56 --data cifar10 --epochs 164
<<<<<<< HEAD
# python3 main.py --arch resnet --depth 56 --save save/cifar10+-resnet-56-e164-b64 --data cifar10+ --epochs 164
python3 main.py --arch resnet --depth 56 --save save/cifar10+-resnet-56-e164-b64-normalized --data cifar10+ --epochs 164 --normalized
# python3 main.py --arch resnet --depth 56 --save save/cifar10+-resnet-56-e300-b128 --data cifar10+ --epochs 300 --batch-size 128
python3 main.py --arch resnet --depth 56 --save save/cifar10+-resnet-56-e300-b128-normalized --data cifar10+ --epochs 300 -batch-size 128 --normalized
python3 main.py --arch resnet --depth 56 --save save/cifar10+-resnet-56-e300-b64-normalized --data cifar10+ --epochs 300 -batch-size 64 --normalized
=======
# python3 main.py --arch resnet --depth 56 --save save/cifar10+-resnet-56 --data cifar10+ --epochs 164
# python3 main.py --arch resnet --depth 56 --save save/cifar10+-resnet-56-e500-b128 --data cifar10+ --epochs 500 --batch-size 128
>>>>>>> 2e2668d5753ca7e865b6ad19348759aaf1395ca1
# python3 main.py --arch resnet --depth 110 --save save/cifar10-resnet-110 --data cifar10 --epochs 164
# python3 main.py --arch resnet --depth 110 --save save/cifar10+-resnet-110 --data cifar10+ --epochs 164
# python3 main.py --arch resnet --depth 110 --save save/cifar10+-resnet-110-e500-b128 --data cifar10+ --epochs 500 --batch-size 128

# ResNet with Stochastic Depth
# python3 main.py --arch resnet --depth 110 --save save/cifar10+-resnet-stoch-110 --data cifar10+ --epochs 164 --death-mode linear --death-rate 0.5
# python3 main.py --arch resnet --depth 56 --save save/cifar10+-resnet-stoch-56 --data cifar10+ --epochs 164 --death-mode linear --death-rate 0.5
# python3 main.py --arch resnet --depth 110 --save save/cifar10+-resnet-stoch-110 --data cifar10+ --epochs 164 --death-mode linear --death-rate 0.5
# python3 main.py --arch resnet --depth 110 --save save/cifar10+-resnet-110-stoch-e500-b128 --data cifar10+ --epochs 500 --batch-size 128 --death-mode linear --death-rate 0.5


# DenseNet
# python3 main.py --arch densenet --depth 100 --growth-rate 12 --bn-size 4 --compression 0.5 --data cifar10+ --epochs 300 --save save/cifar10+-densenet-bc-100 
# python3 main.py --arch densenet --depth 100 --growth-rate 12 --bn-size 4 --compression 0.5 --data cifar10+ --epochs 300 --save save/cifar10+-densenet-bc-100-normalized --normalized
# python3 main.py --arch densenet_andreas --depth 100 --growth-rate 12 --bn-size 4 --compression 0.5 --data cifar10+ --epochs 300 --save save/cifar10+-densenet-andreas-bc-100 
# python3 main.py --arch densenet_andreas --depth 100 --growth-rate 12 --bn-size 4 --compression 0.5 --data cifar10+ --epochs 300 --save save/cifar10+-densenet-andreas-bc-100-normalized --normalized
# python3 main.py --arch densenet_brandon --depth 100 --growth-rate 12 --bn-size 4 --compression 0.5 --data cifar10+ --epochs 300 --save save/cifar10+-densenet-brandon-bc-100
# python3 main.py --arch densenet_brandon --depth 100 --growth-rate 12 --bn-size 4 --compression 0.5 --data cifar10+ --epochs 300 --save save/cifar10+-densenet-brandon-bc-100-normalized --normalized
python3 main.py --arch densenet_andreas --depth 100 --growth-rate 12 --bn-size 4 --compression 0.5 --data cifar10+ --epochs 300 --save save/debug
