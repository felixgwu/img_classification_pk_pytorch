import torch
import os


def adjust_learning_rate(optimizer, lr_init, decay_rate, epoch, num_epochs):
    # """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
    lr = lr_init
    if epoch >= num_epochs * 0.75:
        lr *= decay_rate**2
    elif epoch >= num_epochs * 0.5:
        lr *= decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))


def get_optimizer(model, args):
    if args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), args.lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), args.lr,
                                   alpha=args.alpha,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), args.lr,
                                beta=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def error(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return res
