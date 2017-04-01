# Image Classification Project Killer in PyTorch
Happy April Fools' Day!
This repo is designed for those who want to start their experiments two days before the deadline and kill the project in the last 6 hours.
Inspired by [fb.torch.resnet](https://github.com/facebook/fb.resnet.torch),
it provides fast experiment setup and attempts to maximize the number of projects killed within the given time.
Please feel free to summit issues or pull requests if you want to contribute.

## Usage
Use `python3 main.py -h` to show all arguments.

### Training
Train a ResNet-56 on CIFAR-10 with data augmentation:
```sh
python3 main.py --data cifar10+ --arch resnet --depth 56 --save save/resnet-56 --epochs 164
```
See *scripts/cifar10.sh* and *scripts/cifar100.sh* for more training examples.
### Evaluation
```sh
python3 main.py --resume save/resnet-56/model_best.pth.tar --evaluate test --data cifar10+
```
### Using Tensorboard
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
- [ ] Holding out testing set and use validation set for training
- [ ] CPU support
- [x] Single GPU support
- [ ] Multiple GPUs support
- [ ] Adding *save* & *data* folders to .gitignore to prevent commiting the datasets and models
- [ ] Multiple learning rate decay strategies

### Models
- [ ] AlexNet
- [ ] VGGNet
- [ ] SqueezeNet
- [x] ResNet
- [ ] ResNet with stochastic depth
- [ ] Pre-ResNet
- [ ] Wide ResNet
- [ ] ResNext
- [ ] DenseNet
- [ ] MSDNet
- [ ] Steerable CNN

### Datasets
- CIFAR (Last 5000 samples in the original training set is used for validation)
 - [x] CIFAR-10
 - [x] CIFAR-10+ (Horizontal flip and random cropping with padding 4)
 - [x] CIFAR-100
 - [x] CIFAR-100+ (Horizontal flip and random cropping with padding 4)
- SVHN
 - [ ] SVHN-small (without extra training data)
 - [ ] SVHN
- [ ] ImageNet

### Others
- [x] Learning rate scheduler
- [x] Test resume
- [ ] Result table
- [ ] Tutorials

### Todo List
- [ ] Adding iteration mode (counting iterations instead of epochs)
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
- [ ] SVHN & MNIST
- [ ] Checking Python 2 compatibility
- [ ] Adding license
- [ ] Adding links
- [ ] Pretrained models
- [ ] Pep8 check
- [ ] Iteration mode (Counting iterations instead of epochs)
- [ ] Considering switching to torch.util.trainer framework

## Results

## References

## Acknowledgement
This code is based on the ImageNet training script provided in [PyTorch examples](https://github.com/pytorch/examples/blob/master/imagenet/main.py).

The author is not familiar with licensing. Please contact me there is there are any problems with it.
