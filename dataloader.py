import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def getDataloaders(
        data,
        splits=[
            'train',
            'val',
            'test'],
    aug=True,
    data_root='data',
    batch_size=64,
        num_workers=3):
    train_loader, val_loader, test_loader = None, None, None

    if data in ('cifar10', 'cifar100'):
        d_func = dset.CIFAR10 if data == 'cifar10' else dset.CIFAR100
        print('loading ' + data)

        if aug:
            print('with data augmentation')
            ts = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            ts = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        # uses last 5000 images of the original training split as the
        # validation set
        if 'train' in splits or 'val' in splits:
            trainvalset = d_func(
                data_root,
                train=True,
                transform=ts,
                download=True)
        if 'train' in splits:
            train_loader = torch.utils.data.DataLoader(
                trainvalset,
                batch_size=batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    range(45000)),
                num_workers=num_workers,
                pin_memory=True)
        if 'val' in splits:
            val_loader = torch.utils.data.DataLoader(
                trainvalset,
                batch_size=batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    range(
                        45000,
                        50000)),
                num_workers=num_workers,
                pin_memory=True)
        if 'test' in splits:
            testset = d_func(
                data_root,
                train=False,
                transform=ts,
                download=True)
            test_loader = torch.utils.data.DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True)
    else:
        raise NotImplemented
    return train_loader, val_loader, test_loader
