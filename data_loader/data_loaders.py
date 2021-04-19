from torchvision import datasets, transforms
from base import BaseDataLoader
import os.path
import torch


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


class CIFAR_data_loader(BaseDataLoader):
    def __init__(self, data_dir, batch_size=4, download=True, shuffle=True, validation_split=0.1, num_workers=0, flavor=10, training=True):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        if flavor == 10:
            self.dataset = datasets.CIFAR10(
                data_dir, train=training, download=download, transform=preprocess)
        else:
            self.dataset = datasets.CIFAR100(
                data_dir, train=training, download=download, transform=preprocess)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
# Based on https://github.com/pytorch/examples/blob/master/imagenet/main.py


class ImageNet_data_loader(BaseDataLoader):
    val_loader = None

    def __init__(self, data_dir, batch_size, shuffle, num_workers, pin_memory):
        global val_loader
        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    def split_validation(self):
        return val_loader
