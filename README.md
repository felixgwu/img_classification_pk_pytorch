# Image Classification Project Killer in PyTorch
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
- Preventing overwriting previous experiments
- Saving training/validation loss, errors, and learning rate of each epoch to a TSV file
- Automatically copying all source code to saving directory to prevent accidental deleteion of codes. This is inspired by [SGAN code](https://github.com/xunhuang1995/SGAN/tree/master/mnist).
- [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) support using [tensorboard\_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
- One script to show all experiment results
- Display training time
- Holding out testing set and using validation set for hyperparameter tuning experiments
- GPU support
- Adding *save* & *data* folders to .gitignore to prevent commiting the datasets and trained models
- result table


### Models (See *models* folder for details)
- [ ] AlexNet ([paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks))
- [ ] VGGNet ([paper](https://arxiv.org/abs/1409.1556))
- [ ] SqueezeNet ([paper](https://arxiv.org/abs/1602.07360)) ([code](https://github.com/DeepScale/SqueezeNet))
- [x] ResNet ([paper](https://arxiv.org/abs/1512.03385)) ([code](https://github.com/facebook/fb.resnet.torch))
- [ ] ResNet with stochastic depth ([paper](https://arxiv.org/abs/1603.09382)) ([code](https://github.com/yueatsprograms/Stochastic_Depth))
- [ ] Pre-ResNet ([paper](https://arxiv.org/abs/1603.05027)) ([code](https://github.com/facebook/fb.resnet.torch))
- [ ] Wide ResNet ([paper](https://arxiv.org/abs/1605.07146)) ([code](https://github.com/szagoruyko/wide-residual-networks))
- [ ] ResNeXt ([paper](https://arxiv.org/abs/1611.05431)) ([code](https://github.com/facebookresearch/ResNeXt))
- [x] DenseNet ([paper](https://arxiv.org/abs/1608.06993)) ([code](https://github.com/liuzhuang13/DenseNet))
- [ ] MSDNet ([paper](https://arxiv.org/abs/1703.09844)) ([code](https://github.com/gaohuang/MSDNet))
- [ ] Steerable CNN ([paper](https://arxiv.org/abs/1612.08498))

### Datasets
#### CIFAR
Last 5000 samples in the original training set is used for validation. Each pixel is in [0, 1]. Based on experiments results, normalizing the data to zero mean and unit standard deviation seems to be redundant.
- CIFAR-10
- CIFAR-10+ (Horizontal flip and random cropping with padding 4)
- CIFAR-100
- CIFAR-100+ (Horizontal flip and random cropping with padding 4)

### Todo List
- [ ] copy the old results to */tmp* before overwriting them
- [ ] Python 2.7 support
- [ ] More learning rate decay strategies (currently only dropping at 1/2 and 3/4 of the epochs)
- [ ] CPU support
- [ ] SVHN-small (without extra training data)
- [ ] SVHN
- [ ] MNIST
- [ ] ImageNet
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
### Top1 Validation Error Rate (in percentage)
The number of parameters are calculated based on CIFAR-10 model.
ResNets were training with 164 epochs (like default in fb.resnet.torch) and DenseNets were trained 300 epochs.
Both are using batch_size=64.

| Model                    | Parameters | CIFAR-10 | CIFAR-10+ | CIFAR-100 | CIFAR-100+ | SVHN-small | SVHN |
|--------------------------| -----------|----------|-----------|-----------|------------|------------|------|
| ResNet-56                | 0.86M      | 11.92    | 6.42      | 42.88     | 29.66      |            |      |
| ResNet-110               | 1.73M      | 14.26    | 6.16      | 47.04     | 28.54      |            |      |
| DenseNet (k=12, d=40)    |            |          |           |           |            |            |      |
| DenseNet-BC (k=12,d=100) |            |          |           |           |            |            |      |
| Your model               |            |          |           |           |            |            |      |

### Top1 Testing Error Rate (in percentage)
Coming soon...

## File Descriptions
- *main.py*: main script to train or evaluate models
- *config*: storing configuration of datasets (and maybe other things in the future)
- *utils.pypy*: useful functions
- *getbest.py*: display the best validation error of each saving folder
- *dataloader.py*: defines *getDataloaders* function which is used to load datasets
- *models*: a folder storing all network models. Each script in it should contain a *createModel(\*\*kwargs)* function that takes the arguments and return a model (subclass of nn.Module) for training
- *scripts*: a folder storing example training commands in UNIX shell scripts

## References

## Acknowledgement
This code is based on the ImageNet training script provided in [PyTorch examples](https://github.com/pytorch/examples/blob/master/imagenet/main.py).

The author is not familiar with licensing. Please contact me there is there are any problems with it.
