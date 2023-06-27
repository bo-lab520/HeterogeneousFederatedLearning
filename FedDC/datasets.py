import numpy as np
import torch

from torchvision import datasets, transforms


def get_dataset(dir, name):
    if name == 'mnist':
        train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
        eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())

    elif name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)

    elif name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR100(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR100(dir, train=False, transform=transform_test)

    return train_dataset, eval_dataset


def get_nonIID_data(conf):
    client_idx = {}
    all_data = []
    for i in range(conf["classes"]):
        all_data.append(i)
    for i in range(conf["clients"]):
        samples = np.random.choice(all_data, size=conf["client_classes"], replace=False)
        client_idx[i + 1] = samples

    return client_idx


if __name__ == '__main__':
    cls_priors = np.random.dirichlet(alpha=(1, 1), size=3)
    print(cls_priors)
