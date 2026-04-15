from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from src.config import load_yaml

COARSE_TASK_TO_FINE = {
    "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household_electrical_devices": ["clock", "keyboard", "lamp", "telephone", "television"],
    "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large_man_made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large_omnivores_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium_sized_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non_insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}


def _read_cifar100_raw(root: Path, train: bool):
    base = root / "cifar-100-python"
    if not base.exists():
        raise FileNotFoundError(
            f"CIFAR100 not found under {root}. Expected directory: {base}. "
            "Please copy an existing cifar-100-python directory into this root instead of downloading."
        )
    split_file = base / ("train" if train else "test")
    meta_file = base / "meta"

    with open(split_file, "rb") as f:
        entry = pickle.load(f, encoding="latin1")
    with open(meta_file, "rb") as f:
        meta = pickle.load(f, encoding="latin1")

    data = entry["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    fine_labels = entry["fine_labels"]
    coarse_labels = entry["coarse_labels"]
    fine_names = meta["fine_label_names"]
    return data, fine_labels, coarse_labels, fine_names


class CIFAR100TaskDataset(Dataset):
    def __init__(self, root: str | Path, task_name: str, train: bool, transform=None):
        self.root = Path(root)
        self.task_name = task_name
        self.train = train
        self.transform = transform

        data, fine_labels, coarse_labels, fine_names = _read_cifar100_raw(self.root, train=train)
        fine_name_to_id = {name: idx for idx, name in enumerate(fine_names)}
        task_fines = COARSE_TASK_TO_FINE[task_name]
        self.global_fine_ids = [fine_name_to_id[name] for name in task_fines]
        self.global_to_local = {gid: idx for idx, gid in enumerate(self.global_fine_ids)}

        indices = [i for i, fid in enumerate(fine_labels) if fid in self.global_to_local]
        self.images = data[indices]
        self.labels = [self.global_to_local[fine_labels[i]] for i in indices]
        self.global_labels = [fine_labels[i] for i in indices]
        self.coarse_labels = [coarse_labels[i] for i in indices]
        self.class_names = task_fines

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "global_label": self.global_labels[index],
            "index": index,
        }


class CIFAR100TaskDatasetView(Dataset):
    def __init__(self, base_dataset: CIFAR100TaskDataset, transform):
        self.base = base_dataset
        self.transform = transform
        self.class_names = base_dataset.class_names
        self.labels = base_dataset.labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.base.images[index])
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(self.base.labels[index], dtype=torch.long),
            "global_label": self.base.global_labels[index],
            "index": index,
        }


def _stratified_split_indices(labels: List[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    by_class: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(label, []).append(idx)

    train_idx, val_idx = [], []
    rng = random.Random(seed)
    for label, indices in by_class.items():
        indices = indices[:]
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])
    return train_idx, val_idx


def _build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    return train_tf, eval_tf


def build_task_dataloaders(cfg):
    train_tf, eval_tf = _build_transforms()
    dataset_root = Path(cfg["data"]["root"])
    task_name = cfg["data"]["task_name"]
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["runtime"]["num_workers"])
    pin_memory = bool(cfg["runtime"].get("pin_memory", True))
    persistent_workers = bool(cfg["runtime"].get("persistent_workers", num_workers > 0))
    prefetch_factor = cfg["runtime"].get("prefetch_factor", 2)
    val_ratio = float(cfg["data"]["val_ratio"])
    seed = int(cfg["experiment"]["seed"])

    base_train_raw = CIFAR100TaskDataset(dataset_root, task_name=task_name, train=True, transform=None)
    base_test_raw = CIFAR100TaskDataset(dataset_root, task_name=task_name, train=False, transform=None)
    base_train = CIFAR100TaskDatasetView(base_train_raw, transform=train_tf)
    base_train_eval = CIFAR100TaskDatasetView(base_train_raw, transform=eval_tf)
    test_dataset = CIFAR100TaskDatasetView(base_test_raw, transform=eval_tf)

    train_idx, val_idx = _stratified_split_indices(base_train.labels, val_ratio=val_ratio, seed=seed)

    train_subset = Subset(base_train, train_idx)
    val_subset = Subset(base_train_eval, val_idx)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    loaders = {
        "train": DataLoader(train_subset, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_subset, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    }
    meta = {
        "train_size": len(train_subset),
        "val_size": len(val_subset),
        "test_size": len(test_dataset),
        "class_names": base_train.class_names,
    }
    return loaders, meta
