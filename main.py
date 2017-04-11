#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import os
import shutil
import time
import sys
import glob

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from colorama import Fore
from importlib import import_module

import config
from dataloader import getDataloaders
from utils import (save_checkpoint, get_optimizer)
from train import Trainer

try:
    from tensorboard_logger import configure, log_value
except BaseException:
    configure = None

model_names = list(map(lambda n: os.path.basename(n)[:-3],
                       glob.glob('models/[A-Za-z]*.py')))

parser = argparse.ArgumentParser(
                description='Image classification PK main script')

exp_group = parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save', default='save/default-{}'.format(time.time()),
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory'
                       '(default: save/debug)')
exp_group.add_argument('--resume', default='', type=str, metavar='PATH',
                       help='path to latest checkpoint (default: none)')
exp_group.add_argument('--evaluate', dest='evaluate', default='',
                       choices=['', 'val', 'test'],
                       help='eval mode: evaluate model on val/test set'
                       ' (default: training mode)')
exp_group.add_argument('-f', '--force', dest='force', action='store_true',
                       help='force to overwrite existing save path')
exp_group.add_argument('--print-freq', '-p', default=100, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--no_tensorboard', dest='tensorboard',
                       action='store_false',
                       help='do not use tensorboard_logger for logging')

# dataset related
data_group = parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='cifar10',
                        choices=config.datasets.keys(),
                        help='datasets: ' +
                        ' | '.join(config.datasets.keys()) +
                        ' (default: cifar10)')
data_group.add_argument('--data_root', metavar='DIR', default='data',
                        help='path to dataset (default: data)')
data_group.add_argument('-j', '--workers', dest='num_workers', default=4,
                        type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
data_group.add_argument('--normalized', action='store_true',
                        help='normalize the data into zero mean and unit std')

# model arch related
arch_group = parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet)')
arch_group.add_argument('-d', '--depth', default=56, type=int, metavar='D',
                        help='depth (default=56)')
arch_group.add_argument('--drop-rate', default=0.0, type=float,
                        metavar='DROPRATE', help='dropout rate (default: 0.2)')
arch_group.add_argument('--death-mode', default='none',
                        choices=['none', 'linear', 'uniform'],
                        help='death mode (default: none)')
arch_group.add_argument('--death-rate', default=0.5, type=float,
                        help='death rate rate (default: 0.5)')
arch_group.add_argument('--growth-rate', default=12, type=int,
                        metavar='GR', help='Growth rate of DenseNet'
                        ' (1 means dot\'t use compression) (default: 0.5)')
arch_group.add_argument('--bn-size', default=4, type=int,
                        metavar='B', help='bottle neck ratio of DenseNet'
                        ' (0 means dot\'t use bottle necks) (default: 4)')
arch_group.add_argument('--compression', default=0.5, type=float,
                        metavar='C', help='compression ratio of DenseNet'
                        ' (1 means dot\'t use compression) (default: 0.5)')
# used to set the argument when to resume automatically
arch_resume_names = ['arch', 'depth', 'death_mode', 'death_rate', 'death_rate',
                     'growth_rate', 'bn_size', 'compression']

# training related
optim_group = parser.add_argument_group('optimization', 'optimization setting')
optim_group.add_argument('--epochs', default=164, type=int, metavar='N',
                         help='number of total epochs to run (default 164)')
optim_group.add_argument('--start-epoch', default=1, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')
optim_group.add_argument('--patience', default=0, type=int, metavar='N',
                         help='patience for early stopping'
                         '(0 means no early stopping)')
optim_group.add_argument('-b', '--batch-size', default=64, type=int,
                         metavar='N', help='mini-batch size (default: 64)')
optim_group.add_argument('--optimizer', default='sgd',
                         choices=['sgd', 'rmsprop', 'adam'], metavar='N',
                         help='optimizer (default=sgd)')
optim_group.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                         metavar='LR',
                         help='initial learning rate (default: 0.1)')
optim_group.add_argument('--decay_rate', default=0.1, type=float, metavar='N',
                         help='decay rate of learning rate (default: 0.1)')
optim_group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                         help='momentum (default=0.9)')
optim_group.add_argument('--no_nesterov', dest='nesterov',
                         action='store_false',
                         help='do not use Nesterov momentum')
optim_group.add_argument('--alpha', default=0.99, type=float, metavar='M',
                         help='alpha for ')
optim_group.add_argument('--beta1', default=0.9, type=float, metavar='M',
                         help='beta1 for Adam (default: 0.9)')
optim_group.add_argument('--beta2', default=0.999, type=float, metavar='M',
                         help='beta2 for Adam (default: 0.999)')
optim_group.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                         metavar='W', help='weight decay (default: 1e-4)')


def getModel(arch, **kargs):
    m = import_module('models.' + arch)
    model = m.createModel(**kargs)
    if arch.startswith('alexnet') or arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    return model


def main():
    # parse arg and start experiment
    global args
    best_err1 = 100.
    best_epoch = 0

    args = parser.parse_args()
    args.config_of_data = config.datasets[args.data]
    args.num_classes = config.datasets[args.data]['num_classes']
    if configure is None:
        args.tensorboard = False
        print(Fore.RED +
              'WARNING: you don\'t have tesnorboard_logger installed' +
              Fore.RESET)

    # optionally resume from a checkpoint
    if args.resume:
        if args.resume and os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            old_args = checkpoint['args']
            print('Old args:')
            print(old_args)
            # set args based on checkpoint
            if args.start_epoch <= 0:
                args.start_epoch = checkpoint['epoch'] + 1
            best_epoch = args.start_epoch - 1
            best_err1 = checkpoint['best_err1']
            for name in arch_resume_names:
                if name in vars(args) and name in vars(old_args):
                    setattr(args, name, getattr(old_args, name))
            model = getModel(**vars(args))
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print(
                "=> no checkpoint found at '{}'".format(
                    Fore.RED +
                    args.resume +
                    Fore.RESET),
                file=sys.stderr)
            return
    else:
        # create model
        print("=> creating model '{}'".format(args.arch))
        model = getModel(**vars(args))

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # define optimizer
    optimizer = get_optimizer(model, args)

    trainer = Trainer(model, criterion, optimizer, args)


    # create dataloader
    if args.evaluate == 'val':
        _, val_loader, _ = getDataloaders(
            splits=('val'), **vars(args))
        trainer.test(val_loader, best_epoch)
        return
    elif args.evaluate == 'test':
        _, _, test_loader = getDataloaders(
            splits=('test'), **vars(args))
        trainer.test(test_loader, best_epoch)
        return
    else:
        train_loader, val_loader, _ = getDataloaders(
            splits=('train', 'val'), **vars(args))

    # check if the folder exists
    if os.path.exists(args.save):
        print(Fore.RED + args.save + Fore.RESET
              + ' already exists!', file=sys.stderr)
        if not args.force:
            ans = input('Do you want to overwrite it? [y/N]:')
            if ans not in ('y', 'Y', 'yes', 'Yes'):
                os.exit(1)
        tmp_path = '/tmp/{}_{}'.format(os.path.basename(args.save),
                                       time.time())
        print('move existing {} to {}'.format(args.save, Fore.RED
                                              + tmp_path + Fore.RESET))
        shutil.copytree(args.save, tmp_path)
        shutil.rmtree(args.save)
    os.makedirs(args.save)
    print('create folder: ' + Fore.GREEN + args.save + Fore.RESET)

    # copy code to save folder
    if args.save.find('debug') < 0:
        shutil.copytree(
            '.',
            os.path.join(
                args.save,
                'src'),
            symlinks=True,
            ignore=shutil.ignore_patterns(
                '*.pyc',
                '__pycache__',
                '*.path.tar',
                '*.pth',
                '*.ipynb',
                '.*',
                'data',
                'save',
                'save_backup'))

    # set up logging
    global log_print, f_log
    f_log = open(os.path.join(args.save, 'log.txt'), 'w')

    def log_print(*args):
        print(*args)
        print(*args, file=f_log)
    log_print('args:')
    log_print(args)
    print('model:', file=f_log)
    print(model, file=f_log)
    log_print('# of params:',
              str(sum([p.numel() for p in model.parameters()])))
    f_log.flush()
    torch.save(args, os.path.join(args.save, 'args.pth'))
    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_err1'
              '\tval_err1\ttrain_err5\tval_err']
    if args.tensorboard:
        configure(args.save, flush_secs=5)

    for epoch in range(args.start_epoch, args.epochs + 1):

        # train for one epoch
        train_loss, train_err1, train_err5, lr = trainer.train(
            train_loader, epoch)

        if args.tensorboard:
            log_value('lr', lr, epoch)
            log_value('train_loss', train_loss, epoch)
            log_value('train_err1', train_err1, epoch)
            log_value('train_err5', train_err5, epoch)

        # evaluate on validation set
        val_loss, val_err1, val_err5 = trainer.test(val_loader, epoch)

        if args.tensorboard:
            log_value('val_loss', val_loss, epoch)
            log_value('val_err1', val_err1, epoch)
            log_value('val_err5', val_err5, epoch)

        # save scores to a tsv file, rewrite the whole file to prevent
        # accidental deletion
        scores.append(('{}\t{}' + '\t{:.4f}' * 6)
                      .format(epoch, lr, train_loss, val_loss,
                              train_err1, val_err1, train_err5, val_err5))
        with open(os.path.join(args.save, 'scores.tsv'), 'w') as f:
            print('\n'.join(scores), file=f)

        # remember best err@1 and save checkpoint
        is_best = val_err1 < best_err1
        if is_best:
            best_err1 = val_err1
            best_epoch = epoch
            print(Fore.GREEN + 'Best var_err1 {}'.format(best_err1) +
                  Fore.RESET)
            # test_loss, test_err1, test_err1 = validate(
            #     test_loader, model, criterion, epoch, True)
            # save test
        save_checkpoint({
            'args': args,
            'epoch': epoch,
            'best_epoch': best_epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
        }, is_best, args.save)
        if not is_best and epoch - best_epoch >= args.patience > 0:
            break
    print('Best val_err1: {:.4f} at epoch {}'.format(best_err1, best_epoch))


if __name__ == '__main__':
    main()
