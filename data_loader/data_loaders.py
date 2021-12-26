import os.path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torchvision import datasets, transforms
from base import BaseDataLoader
from utils import is_master
CIFAR_TRAIN_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
CIFAR_TRAIN_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
CIFAR_TEST_MEAN = [0.5088964127604166, 0.48739301317401956, 0.44194221124387256]
CIFAR_TEST_STD = [0.2682515741720801, 0.2573637364478126, 0.2770957707973042]
CIFAR_TRAIN_PRE = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_TRAIN_MEAN, CIFAR_TRAIN_STD)])
CIFAR_TEST_PRE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_TEST_MEAN, CIFAR_TEST_STD)])


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CIFAR_data_loader():
    def __init__(self, data_dir, batch_size=4, download=False, shuffle=True, validation_split=0.1, num_workers=0, pin_memory=True, flavor=100, training=True):
        if training:
            self.preprocess = CIFAR_TRAIN_PRE
        else:
            self.preprocess = CIFAR_TEST_PRE

        if flavor == 10:
            dataset = datasets.CIFAR10(
                data_dir, train=training, download=download if is_master() else False, transform=self.preprocess)
        else:
            dataset = datasets.CIFAR100(
                data_dir, train=training, download=download if is_master() else False, transform=self.preprocess)

        if training:
            len_set = len(dataset)
            if isinstance(validation_split, int):
                assert validation_split >= 0, "validation_split can not be negative"
                assert len_set > validation_split, "validation_split is bigger than data set size"
                len_valid = validation_split
            else:
                len_valid = int(validation_split*len_set)
            len_train = len_set - len_valid
            train_set, valid_set = random_split(dataset, [len_train, len_valid])
            self.train_sampler = DistributedSampler(train_set) if dist.is_initialized() else None
            self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=(shuffle if self.train_sampler is None else False), sampler=self.train_sampler, num_workers=num_workers, pin_memory=pin_memory)
            self.valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        else:
            self.test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    def get_train_loader(self):
        return self.train_loader
    def get_valid_loader(self):
        return self.valid_loader
    def get_test_loader(self):
        return self.test_loader
# Based on https://github.com/pytorch/examples/blob/master/imagenet/main.py


class ImageNet_data_loader(BaseDataLoader):
    """
        ImageNet data loader

        Args:
        data_dir (str): Directory where data set is saved, should have 
                        three folders train, val & test
        batch_size (int): Batch size
        shuffle (bool):
        num_workers (int): Number of workers
        pin_memory (bool): 
        training (bool):
    """

    val_loader = None

    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=0, pin_memory=True, training=True):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if training:
            global val_loader
            traindir = os.path.join(data_dir, 'train')
            valdir = os.path.join(data_dir, 'val')
            train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
            val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            super().__init__(train_dataset, batch_size, shuffle, 0.0, num_workers)
        else:
            testdir = os.path.join(data_dir, 'test')
            test_dataset = datasets.ImageFolder(testdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
            super().__init__(test_dataset, batch_size, shuffle, 0.0, num_workers)

    def split_validation(self):
        return val_loader

class dist_ImageNet_data_loader():
    """
        ImageNet data loader

        Args:
        data_dir (str): Directory where data set is saved, should have 
                        three folders train, val & test
        batch_size (int): Batch size
        shuffle (bool):
        num_workers (int): Number of workers
        pin_memory (bool): 
        training (bool):
    """

    def __init__(self, data_dir, batch_size, num_workers=0, pin_memory=True, training=True):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if training:
            traindir = os.path.join(data_dir, 'train')
            valdir = os.path.join(data_dir, 'val')
            train_set = datasets.ImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
            valid_set = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
            self.train_sampler = DistributedSampler(train_set)
            self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=self.train_sampler, num_workers=num_workers, pin_memory=pin_memory)
            self.valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        else:
            testdir = os.path.join(data_dir, 'test')
            test_set = datasets.ImageFolder(testdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
            self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    def get_train_loader(self):
        return self.train_loader
    def get_valid_loader(self):
        return self.valid_loader
    def get_test_loader(self):
        return self.test_loader
