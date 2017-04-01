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

## Features

### Experiment
- [ ] Preventing overwriting previous experiments
- [ ] Saving training/validation loss, errors, and learning rate of each epoch to a TSV file
- [ ] Automatically copying all source code to saving directory to prevent
- [ ] TensorBoard support using tensorboard_logger
- [ ] One script to show all experiment results

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
- [x] CIFAR-10
- [x] CIFAR-10+ (Horizontal flip and random cropping with padding 4)
- [x] CIFAR-100
- [x] CIFAR-100+ (Horizontal flip and random cropping with padding 4)
- [ ] SVHN-small (without extra training data)
- [ ] SVHN
- [ ] ImageNet

## Progress
- [x] CIFAR 10 & 100
- [x] Progressbar
- [x] Logging
- [x] Data Augmentation
- [x] Copying experiment code automatically
- [x] Training, Validation, Testing split (Hiding test errors to prevent overfitting the testing set)
- [x] Learning rate scheduler
- [x] Test resume
- [ ] DenseNet
- [ ] AlexNet, VGGNet, SqueezeNet
- [ ] Adding result table
- [ ] Adding example scripts
- [ ] Extend README file
- [ ] Adding acknowledgement
- [ ] Multiple GPU support
- [ ] Custom models & criterions tutorial
- [ ] Custom train & test functions tutorial
- [ ] Custom initialization
- [ ] Adding an example project killing scenario 
- [ ] SVHN & MNIST
- [ ] Custom datasets tutorial
- [ ] Python 2.7 support
- [ ] Adding license
- [ ] ImageNet
- [ ] Comparing tensorboard\_logger v.s. pycrayon
- [ ] Pretrained models
- [ ] Pep8 check
- [ ] Iteration mode (Counting iterations instead of epochs)
- [ ] Considering switching to torch.util.trainer framework
- [ ] ResNext
- [ ] ResNet with stochastic depth
- [ ] MSDNet 
- [ ] Checking the checkboxes
- [ ] CPU support

## Results

## References

## Acknowledgement
This code is based on the ImageNet training script provided in [PyTorch examples](https://github.com/pytorch/examples/blob/master/imagenet/main.py)
