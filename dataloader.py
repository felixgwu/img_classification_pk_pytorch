import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from utils import Cutout


def getDataloaders(data, config_of_data, splits=['train', 'val', 'test'],
                   aug=True, use_validset=True, data_root='data', batch_size=64, normalized=True,
                   data_aug=False, cutout=False, n_holes=1, length=16,
                   num_workers=3, **kwargs):
    train_loader, val_loader, test_loader = None, None, None

    if data.find('cifar10') >= 0:
        print('loading ' + data)
        print(config_of_data)
        if data.find('cifar100') >= 0:
            d_func = dset.CIFAR100
            normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                             std=[0.2675, 0.2565, 0.2761])
        else:
            d_func = dset.CIFAR10
            normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                             std=[0.2471, 0.2435, 0.2616])
        if data_aug:
            print('with data augmentation')
            aug_trans = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            aug_trans = []
        common_trans = [transforms.ToTensor()]
        if normalized:
            print('dataset is normalized')
            common_trans.append(normalize)
        train_compose = aug_trans + common_trans
        if cutout:
            train_compose.append(Cutout(n_holes=n_holes, length=length))
        train_compose = transforms.Compose(train_compose)
        test_compose = transforms.Compose(common_trans)

        if use_validset:
            # uses last 5000 images of the original training split as the
            # validation set
            if 'train' in splits:
                train_set = d_func(data_root, train=True, transform=train_compose,
                                   download=True)
                train_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        range(45000)),
                    num_workers=num_workers, pin_memory=True)
            if 'val' in splits:
                val_set = d_func(data_root, train=True, transform=test_compose)
                val_loader = torch.utils.data.DataLoader(
                    val_set, batch_size=batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        range(45000, 50000)),
                    num_workers=num_workers, pin_memory=True)

            if 'test' in splits:
                test_set = d_func(data_root, train=False, transform=test_compose)
                test_loader = torch.utils.data.DataLoader(
                    test_set, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True)
        else:
            if 'train' in splits:
                train_set = d_func(data_root, train=True, transform=train_compose,
                                   download=True)
                train_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True)
            if 'val' in splits or 'test' in splits:
                test_set = d_func(data_root, train=False, transform=test_compose)
                test_loader = torch.utils.data.DataLoader(
                    test_set, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True)
                val_loader = test_loader


    else:
        raise NotImplemented
    return train_loader, val_loader, test_loader
