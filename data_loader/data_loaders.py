from torchvision import datasets, transforms
from base import BaseDataLoader


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
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CIFAR_data_loader(BaseDataLoader):
    def __init__(self, data_dir,batch_size=4,download=True, shuffle=True, validation_split=0.1, num_workers=4,flavor=10, training=True):
        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if flavor == 10:
            self.dataset = datasets.CIFAR10(data_dir,train=training,download=download,transform=preprocess)
        else:
            self.dataset = datasets.CIFAR100(data_dir,train=training,download=download,transform=preprocess)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)