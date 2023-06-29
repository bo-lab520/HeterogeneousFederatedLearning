import time

from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

seed = 42


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


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max() + 1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    # return client_idcs

    client_idx = {}
    for i in range(len(client_idcs)):
        client_idx[i] = client_idcs[i]

    return client_idx


def dirichlet_nonIID_data(train_data, conf):
    np.random.seed(seed)

    classes = train_data.classes
    n_classes = len(classes)
    # labels = np.concatenate([np.array(train_data.targets), np.array(test_data.targets)], axis=0)
    labels = np.array(train_data.targets)
    # dataset = ConcatDataset([train_data, test_data])

    # 我们让每个client不同label的样本数量不同，以此做到Non-IID划分
    # client_idcs = dirichlet_split_noniid(labels, alpha=conf["dirichlet_alpha"], n_clients=conf["clients"])
    return dirichlet_split_noniid(labels, alpha=conf["dirichlet_alpha"], n_clients=conf["clients"])

    # 展示不同label划分到不同client的情况
    # plt.figure(figsize=(12, 8))
    # plt.hist([labels[idc] for idc in client_idcs], stacked=True,
    #          bins=np.arange(min(labels) - 0.5, max(labels) + 1.5, 1),
    #          label=["Client {}".format(i) for i in range(n_clients)],
    #          rwidth=0.5)
    # plt.xticks(np.arange(n_classes), train_data.classes)
    # plt.xlabel("Label type")
    # plt.ylabel("Number of samples")
    # plt.legend(loc="upper right")
    # plt.title("Display Label Distribution on Different Clients")
    # plt.show()


if __name__ == "__main__":
    dirichlet_nonIID_data()
