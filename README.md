# Image Classification Project Killer in PyTorch
Happy April Fools' Day!
This repo is designed for those who want to start their experiments two days before the deadline and kill the project in the last 6 hours.
Inspired by [fb.torch.resnet](https://github.com/facebook/fb.resnet.torch),
it provides fast experiment setup and attempts to maximize the number of projects killed within the given time.
Please feel free to summit issues or pull requests if you want to contribute.

## Usage
Use `python3 main.py -h` to show all arguments.

### Training
Train a ResNet-56 on CIFAR-10 with data augmentation using GPU0:
```sh
CUDA_VISIBLE_DEVICES=0 python3 main.py --data cifar10+ --arch resnet --depth 56 --save save/cifar10+-resnet-56 --epochs 164
```
Train a ResNet-110 on CIFAR-100 without data augmentation using GPU0 and GPU2:
```sh
CUDA_VISIBLE_DEVICES=0,2 python3 main.py --data cifar100 --arch resnet --depth 110 --save save/cifar100-resnet-110 --epochs 164
```

See *scripts/cifar10.sh* and *scripts/cifar100.sh* for more training examples.
### Evaluation
```sh
python3 main.py --resume save/resnet-56/model_best.pth.tar --evaluate test --data cifar10+
```
### Show Training & Validation Results
#### Python script
```sh
getbest.py save/* FOLDER_1 FOLDER_2
```
In short, this script reads the *scores.tsv* in the saving folders and display the best validation errors of them.

#### Using Tensorboard
```sh
tensorboard --logdir save --port PORT
```

## Features

### Experiment Setup & Logging
- [x] Preventing overwriting previous experiments
- [x] Saving training/validation loss, errors, and learning rate of each epoch to a TSV file
- [x] Automatically copying all source code to saving directory to prevent
- [x] TensorBoard support using tensorboard_logger
- [x] One script to show all experiment results
- [x] Display training time
- [x] Holding out testing set and using validation set for hyperparameter tuning experiments
- [ ] CPU support
- [x] GPU support
- [ ] Adding *save* & *data* folders to .gitignore to prevent commiting the datasets and models
- [ ] Multiple learning rate decay strategies

### Models (See *models* folder for details)
- [ ] AlexNet ([paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks))
- [ ] VGGNet ([paper](https://arxiv.org/abs/1409.1556))
- [ ] SqueezeNet ([paper](https://arxiv.org/abs/1602.07360)) ([code](https://github.com/DeepScale/SqueezeNet))
- [x] ResNet ([paper](https://arxiv.org/abs/1512.03385)) ([code](https://github.com/facebook/fb.resnet.torch))
- [ ] ResNet with stochastic depth ([paper](https://arxiv.org/abs/1603.09382)) ([code](https://github.com/yueatsprograms/Stochastic_Depth))
- [ ] Pre-ResNet ([paper](https://arxiv.org/abs/1603.05027)) ([code](https://github.com/facebook/fb.resnet.torch))
- [ ] Wide ResNet ([paper](https://arxiv.org/abs/1605.07146)) ([code](https://github.com/szagoruyko/wide-residual-networks))
- [ ] ResNeXt ([paper](https://arxiv.org/abs/1611.05431)) ([code](https://github.com/facebookresearch/ResNeXt))
- [ ] DenseNet (coming soon) ([paper](https://arxiv.org/abs/1608.06993)) ([code](https://github.com/liuzhuang13/DenseNet))
- [ ] MSDNet ([paper](https://arxiv.org/abs/1703.09844)) ([code](https://github.com/gaohuang/MSDNet))
- [ ] Steerable CNN ([paper](https://arxiv.org/abs/1612.08498))

### Datasets
- CIFAR (Last 5000 samples in the original training set is used for validation)
 - [x] CIFAR-10
 - [x] CIFAR-10+ (Horizontal flip and random cropping with padding 4)
 - [x] CIFAR-100
 - [x] CIFAR-100+ (Horizontal flip and random cropping with padding 4)
- SVHN
 - [ ] SVHN-small (without extra training data)
 - [ ] SVHN
- [ ] MNIST
- [ ] ImageNet

### Others
- [x] Learning rate scheduler
- [x] Test resume
- [ ] Result table
- [ ] Tutorials

### Todo List
- [ ] Comparing tensorboard\_logger v.s. pycrayon
- [ ] Adding result table
- [ ] Adding example scripts
- [ ] Comparing tensorboard\_logger v.s. pycrayon
- [ ] Adding acknowledgement
- [ ] Custom models & criterions tutorial
- [ ] Custom train & test functions tutorial
- [ ] Custom datasets tutorial
- [ ] Custom initialization
- [ ] Adding an example project killing scenario
- [ ] Adding license
- [ ] Pretrained models
- [ ] Pep8 check
- [ ] Iteration mode (Counting iterations instead of epochs)

## Results
### Top1 Error Rate (in percentage)
| Model      | CIFAR-10 | CIFAR-10+ | CIFAR-100 | CIFAR-100+ | SVHN-small | SVHN |
|------------|----------|-----------|-----------|------------|------------|------|
| ResNet-56  |          | 6.42      | 42.88     |            |            |      |
| ResNet-110 |          | 6.16      |           |            |            |      |

## References

## Acknowledgement
This code is based on the ImageNet training script provided in [PyTorch examples](https://github.com/pytorch/examples/blob/master/imagenet/main.py).

The author is not familiar with licensing. Please contact me there is there are any problems with it.
