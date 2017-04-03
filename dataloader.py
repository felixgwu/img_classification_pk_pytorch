import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def getDataloaders(data, config_of_data, splits=['train', 'val', 'test'],
                   aug=True, data_root='data', batch_size=64, normalized=True,
                   num_workers=3, **kwargs):
    train_loader, val_loader, test_loader = None, None, None

    if data.find('cifar10') >= 0:
        print('loading ' + data)
        print(config_of_data)
        if data.find('cifar100') >= 0:
            d_func = dset.CIFAR100
            normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                             std=[0.2471, 0.2435, 0.2616])
        else:
            d_func = dset.CIFAR10
            normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                             std=[0.2675, 0.2565, 0.2761])
        if config_of_data['augmentation']:
            print('with data augmentation')
            ts = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            ts = [
                transforms.ToTensor(),
            ]
        if normalized:
            print('dataset is normalized')
            ts.append(normalize)
        ts = transforms.Compose(ts)

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
                trainvalset, batch_size=batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    range(45000)),
                num_workers=num_workers, pin_memory=True)
        if 'val' in splits:
            val_loader = torch.utils.data.DataLoader(
                trainvalset, batch_size=batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    range( 45000, 50000)),
                num_workers=num_workers, pin_memory=True)
        if 'test' in splits:
            testset = d_func(data_root, train=False,
                transform=ts, download=True)
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True)
    else:
        raise NotImplemented
    return train_loader, val_loader, test_loader
