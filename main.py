#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import os
import shutil
import time
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from colorama import Fore
from importlib import import_module

import config
from utils import save_checkpoint, AverageMeter, adjust_learning_rate, error, get_optimizer
from dataloader import getDataloaders

try:
    from tensorboard_logger import configure, log_value
except BaseException:
    configure = None


model_names = ['resnet']
# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    '--save',
    default='save/default-{}'.format(time.time()),
    type=str,
    metavar='SAVE',
    help='path to the experiment logging directory (default: save/debug)')
parser.add_argument('--data', metavar='D', default='cifar10',
                    choices=config.datasets.keys(),
                    help='datasets: ' +
                    ' | '.join(config.datasets.keys()) +
                    ' (default: cifar10)')
parser.add_argument('--data_root', metavar='DIR', default='data',
                    help='path to dataset (default: data)')
parser.add_argument(
    '-j',
    '--workers',
    dest='num_workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')

# experiment setting related
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', dest='evaluate', default='',
                    choices=['', 'val', 'test'],
                    help='eval mode: evaluate model on val/test set (default: training mode)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-f', '--force', dest='force', action='store_true',
                    help='force to overwrite existing save path')

# model arch related
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet', type=str,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet)')
parser.add_argument('-d', '--depth', default=56, type=int, metavar='D',
                    help='depth (default=56)')
parser.add_argument('--drop-rate', default=0.0, type=float,
                    metavar='LR', help='dropout rate (default: 0.2)')

# training related
parser.add_argument('--epochs', default=164, type=int, metavar='N',
                    help='number of total epochs to run (default 164)')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument(
    '--no_tensorboard',
    dest='tensorboard',
    action='store_false',
    help='do not use tensorboard_logger for logging')
parser.add_argument('--optimizer', default='sgd',
                    choices=['sgd', 'rmsprop', 'adam'], metavar='N',
                    help='optimizer (default=sgd)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--decay_rate', default=0.1, type=float, metavar='N',
                    help='decay rate of learning rate (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default=0.9)')
parser.add_argument('--no_nesterov', dest='nesterov', action='store_false',
                    help='do not use Nesterov momentum')
parser.add_argument('--alpha', default=0.99, type=float, metavar='M',
                    help='alpha for ')
parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
                    help='beta1 for Adam (default: 0.9)')
parser.add_argument('--beta2', default=0.999, type=float, metavar='M',
                    help='beta2 for Adam (default: 0.999)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--patience', default=0, type=int, metavar='N',
                    help='patience for early stopping'
                    '(0 means no early stopping)')

best_err1 = 100.
best_epoch = 0


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
    global args, best_err1, best_epoch
    args = parser.parse_args()
    args.config_of_data = config.datasets[args.data]
    args.num_classes = config.datasets[args.data]['num_classes']
    if configure is None:
        args.tensorboard = False

    # optionally resume from a checkpoint
    if args.resume:
        if args.resume and os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            old_args = checkpoint['args']
            print('Old args:')
            print(old_args)
            # set args based on checkpoint
            args.arch = checkpoint['arch']
            if args.start_epoch <= 0:
                args.start_epoch = checkpoint['epoch'] + 1
            best_epoch = args.start_epoch - 1
            best_err1 = checkpoint['best_err1']
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

    # create dataloader
    if args.evaluate == 'val':
        train_loader, val_loader, test_loader = getDataloaders(
            splits=('val'), **vars(args))
        validate(val_loader, model, criterion, best_epoch)
        return
    elif args.evaluate == 'test':
        train_loader, val_loader, test_loader = getDataloaders(
            splits=('test'), **vars(args))
        validate(test_loader, model, criterion, best_epoch)
        return
    else:
        train_loader, val_loader, test_loader = getDataloaders(
            splits=('train', 'val'), **vars(args))

    # define optimizer
    optimizer = get_optimizer(model, args)

    # check if the folder exists
    if os.path.exists(args.save):
        print(Fore.RED + args.save + Fore.RESET
              + ' already exists!', file=sys.stderr)
        if not args.force:
            ans = input('Do you want to overwrite it? [y/N]:')
            if ans not in ('y', 'Y', 'yes', 'Yes'):
                os.exit(1)
        print('remove existing ' + args.save)
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
    with open(os.path.join(args.save, 'log.txt'), 'w') as f:
        def log_print(*args):
            print(*args)
            print(*args, file=f)
        log_print('args:')
        log_print(args)
        log_print('model:')
        log_print(model)
        # log_print('optimizer:')
        # log_print(vars(optimizer))
        log_print('# of params:',
                  str(sum([p.numel() for p in model.parameters()])))
    torch.save(args, os.path.join(args.save, 'args.pth'))
    scores = [
        'epoch\tlr\ttrain_loss\tval_loss\ttrain_err1\tval_err1\ttrain_err5\tval_err']
    if args.tensorboard:
        configure(args.save, flush_secs=5)

    for epoch in range(args.start_epoch, args.epochs + 1):
        lr = adjust_learning_rate(
            optimizer,
            args.lr,
            args.decay_rate,
            epoch,
            args.epochs)  # TODO: add custom
        print('Epoch {:3d} lr = {:.6e}'.format(epoch, lr))
        if args.tensorboard:
            log_value('lr', lr, epoch)

        # train for one epoch
        train_loss, train_err1, train_err5 = train(
            train_loader, model, criterion, optimizer, epoch)
        if args.tensorboard:
            log_value('train_loss', train_loss, epoch)
            log_value('train_err1', train_err1, epoch)
            log_value('train_err5', train_err5, epoch)

        # evaluate on validation set
        val_loss, val_err1, val_err5 = validate(
            val_loader, model, criterion, epoch)
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
            print(Fore.GREEN + 'Best var_err1 {}'.format(best_err1) + Fore.RESET)
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


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure error and record loss
        err1, err5 = error(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(err1[0], input.size(0))
        top5.update(err5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.print_freq > 0 and (i + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.val:.4f}\t'
                  'Err@1 {top1.val:.4f}\t'
                  'Err@5 {top5.val:.4f}'.format(
                      epoch, i + 1, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('Epoch: {:3d} Train loss {loss.avg:.4f} Err@1 {top1.avg:.4f} Err@5 {top5.avg:.4f}'
          .format(epoch, loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, epoch, silence=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure error and record loss
        err1, err5 = error(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(err1[0], input.size(0))
        top5.update(err5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if not silence:
        print(
            'Epoch: {:3d} val   loss {loss.avg:.4f} Err@1 {top1.avg:.4f} Err@5 {top5.avg:.4f}'.format(
                epoch,
                loss=losses,
                top1=top1,
                top5=top5))

    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
