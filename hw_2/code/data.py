"""Dataset transforms: explicit Caffe recipe or conventional PyTorch adaptation."""
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

CIFAR_MEAN = (.4914, .4822, .4465)
CIFAR_STD = (.2470, .2435, .2616)


def center_image(x, mean):
    return x * 255 - mean


def identity_batch(batch):
    return batch


class CachedDataset(Dataset):
    """Cache normalized images once; DataLoader fetches/augments whole batches."""
    def __init__(self, dataset, name, mean=None, mirror=False):
        self.images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float() if name == 'cifar10' else dataset.data[:, None].float()
        self.labels = torch.as_tensor(dataset.targets, dtype=torch.long)
        # Match ToTensor followed by the existing Caffe transforms exactly.
        self.images.div_(255)
        if name == 'cifar10':self.images.mul_(255).sub_(mean)
        else:self.images.div_(256 / 255)
        self.mirror = mirror

    def __len__(self):return len(self.labels)

    def __getitems__(self, indices):
        x, y = self.images[indices], self.labels[indices]
        if self.mirror:
            # Mean subtraction happens BEFORE horizontal reflection.
            mask = torch.rand(len(indices)) < .5
            x = torch.where(mask[:, None, None, None], x.flip(-1), x)
        return x, y

    def __getitem__(self, index):
        x, y = self.__getitems__([index])
        return x[0], y[0]


def get_loaders(dataset, batch_size=128, root='data', num_workers=8,
                train_subset=None, pin_memory=True, normalization='caffe', subset_seed=0,
                cache=False, seed=0):
    if batch_size < 1 or num_workers < 0:
        raise ValueError('batch size must be positive; workers must be nonnegative')
    if dataset == 'cifar10':
        train_ds = datasets.CIFAR10(root, train=True, download=True)
        test_ds = datasets.CIFAR10(root, train=False, download=True)
        if normalization == 'caffe':
            mean = torch.from_numpy(train_ds.data.mean(axis=0).transpose(2, 0, 1)).float()
            common = [transforms.ToTensor(), transforms.Lambda(partial(center_image, mean=mean))]
            train_tf = transforms.Compose([*common, transforms.RandomHorizontalFlip()])
        else:
            common = [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
            train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(), *common])
        train_ds.transform = train_tf
        test_ds.transform = transforms.Compose(common)
        channels = 3
    elif dataset == 'mnist':
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            (0.,), (256 / 255,)) if normalization == 'caffe' else
            transforms.Normalize((.1307,), (.3081,))])
        train_ds = datasets.MNIST(root, train=True, download=True, transform=tf)
        test_ds = datasets.MNIST(root, train=False, download=True, transform=tf)
        channels = 1
    else:
        raise ValueError(f'unknown dataset {dataset}')
    if cache:
        if normalization != 'caffe':raise ValueError('batched cache currently supports the Caffe protocol only')
        train_ds = CachedDataset(train_ds, dataset, mean if dataset == 'cifar10' else None, dataset == 'cifar10')
        test_ds = CachedDataset(test_ds, dataset, mean if dataset == 'cifar10' else None)
    if train_subset is not None:
        if not batch_size <= train_subset <= len(train_ds):
            raise ValueError('subset must contain at least one batch and fit the training dataset')
        idx = np.random.RandomState(subset_seed).choice(len(train_ds), train_subset, replace=False)
        train_ds = Subset(train_ds, sorted(idx.tolist()))
    if batch_size > len(train_ds): raise ValueError('batch exceeds dataset size')
    # Full batches are used; the discarded remainder and steps/epoch are recorded.
    options = dict(num_workers=num_workers, pin_memory=pin_memory,
                   persistent_workers=num_workers > 0)
    if cache:options['collate_fn'] = identity_batch
    # Evaluation must not consume the training RNG when its frequency changes.
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                       generator=torch.Generator().manual_seed(seed), **options)
    test = DataLoader(test_ds, batch_size=1000, shuffle=False,
                      generator=torch.Generator().manual_seed(seed + 1), **options)
    return train, test, channels
