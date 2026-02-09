import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np

def get_imbalanced_data(root='./data', majority_class=3, minority_ratio=0.03):
    print(f"[DATA] Generowanie danych (Majority Class: {majority_class}, Ratio: {minority_ratio})...")
    
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

    full_trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    
    targets = np.array(full_trainset.targets)
    indices = []
    cls_counts = []
    
    for i in range(10):
        cls_idx = np.where(targets == i)[0]
        limit = 5000 if i == majority_class else int(5000 * minority_ratio)
        np.random.shuffle(cls_idx)
        indices.extend(cls_idx[:limit])
        cls_counts.append(limit)

    train_subset = Subset(full_trainset, indices)
    all_targets = np.array(full_trainset.targets)
    subset_targets = all_targets[indices]
    
    print(f"[DATA] Liczność klas: {cls_counts}")
    return train_subset, test_set, cls_counts, subset_targets, full_trainset.classes