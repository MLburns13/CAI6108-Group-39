"""
Code for handling the data loading

    train_x, train_y, test_x, test_y, val_x, val_y = load_preprocess_project(
        root="sp26cai6108mle-project-aggregated/aggregated",
        val_prop=0.15,
        test_prop=0.10,
        normalize=True,
        seed=42,
        verbose=True,
    )
"""

from __future__ import annotations

import re
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LABEL_ORDER: List[str] = [
    "pen", "paper", "book", "clock", "phone", "laptop",
    "chair", "desk", "bottle", "keychain", "backpack", "calculator",
]

VALID_LABELS  = set(LABEL_ORDER)
LABEL_TO_IDX  = {label: i for i, label in enumerate(LABEL_ORDER)}
NUM_CLASSES   = len(LABEL_ORDER)

_IMG_RE = re.compile(r"^img(\S+)\.png$", re.IGNORECASE)

# ImageNet statistics (for pretrained backbone normalisation)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_preprocess_project(
    root: str = "sp26cai6108mle-project-aggregated",
    val_prop:   float = 0.15,
    test_prop:  float = 0.10,
    normalize:  bool  = True,
    seed:       int   = 42,
    verbose:    bool  = False,
) -> Tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    
    """
    verbose : Print shape and label-frequency info to stdout.
    Returns: (train_x, train_y, test_x, test_y, val_x, val_y)
    _x arrays have shape (N, 3, H, W)
    _y arrays have shape (N, 12)
    """
    
    all_x_list: List[np.ndarray] = []
    all_y_list: List[np.ndarray] = []

    root_path = Path(root)
    for subdir in sorted(root_path.iterdir()):
        if not subdir.is_dir():
            continue

        label_tokens = subdir.name.split("_")

        #skip directories with invalid or duplicate labels
        if not label_tokens or any(t not in VALID_LABELS for t in label_tokens):
            continue
        if len(label_tokens) != len(set(label_tokens)):
            continue

        target = np.zeros(NUM_CLASSES, dtype=np.float32)
        for token in label_tokens:
            target[LABEL_TO_IDX[token]] = 1.0

        for img_path in sorted(subdir.iterdir()):
            if not (img_path.is_file() and _IMG_RE.match(img_path.name)):
                continue

            img = Image.open(img_path) #(128, 128, 3)
            arr = np.array(img, dtype=np.uint8) #(H, W, 3)
            arr = arr.transpose(2, 0, 1) #(3, H, W) channel-first

            all_x_list.append(arr)
            all_y_list.append(target.copy())

    assert len(all_x_list) > 0, (
        f"No valid images found under '{root}'. "
        "Check that the directory contains label-named sub-folders."
    )

    all_x = np.stack(all_x_list, axis=0) #(N, 3, H, W)  uint8
    all_y = np.stack(all_y_list, axis=0) #(N, 12) float32
    del all_x_list, all_y_list

    """
    sklearn's train_test_split does not directly support multilabel
    stratification, so we encode each sample's active-label set as a single
    integer (treating the binary vector as a base-2 number) and use that as
    the stratify key.  This keeps rare label combinations proportionally
    distributed across splits.
    """
    
    strat_key = _proxy_stratify_key(all_y)

    if test_prop > 0.0:
        #test set
        tmp_x, test_x, tmp_y, test_y, tmp_key, _ = train_test_split(
            all_x, all_y, strat_key,
            test_size=test_prop,
            random_state=seed,
            stratify=strat_key,
        )
        #train/val sets
        adjusted_val = val_prop / (1.0 - test_prop)
        train_x, val_x, train_y, val_y = train_test_split(
            tmp_x, tmp_y,
            test_size=adjusted_val,
            random_state=seed,
            stratify=_proxy_stratify_key(tmp_y),
        )
    else:
        train_x, val_x, train_y, val_y = train_test_split(
            all_x, all_y,
            test_size=val_prop,
            random_state=seed,
            stratify=strat_key,
        )
        test_x = np.empty((0, *all_x.shape[1:]), dtype=np.uint8)
        test_y = np.empty((0, NUM_CLASSES),        dtype=np.float32)

    def _to_float(arr: np.ndarray) -> np.ndarray:
        out = arr.astype(np.float32)
        if normalize:
            out /= 255.0
        return out

    train_x = _to_float(train_x)
    val_x   = _to_float(val_x)
    test_x  = _to_float(test_x)

    #sanity checks
    n_total = len(all_x)
    assert len(train_x) + len(val_x) + len(test_x) == n_total, \
        "Split sizes do not sum to total sample count."
    assert train_x.dtype == np.float32 and train_y.dtype == np.float32

    if verbose:
        _print_split_summary(all_x, all_y, train_x, train_y,
                             val_x, val_y, test_x, test_y)

    return train_x, train_y, test_x, test_y, val_x, val_y

#HELPERS

def _proxy_stratify_key(y: np.ndarray, min_count: int = 2) -> np.ndarray:
    """
    Build a 1-D integer array suitable as sklearn stratify= key.
    Encodes each sample's active-label set as a base-2 integer.
    With 12 labels the maximum value is 2^12 - 1 = 4095, well within int32.
    """
    powers = (2 ** np.arange(NUM_CLASSES, dtype=np.int32)) #(12,)
    keys   = y.astype(np.int32) @ powers #(N,)
 
    #stop stratification from breaking due to rare label combinations like book_chair_bottle_backpack
    unique_keys, counts = np.unique(keys, return_counts=True)
    rare_keys = set(unique_keys[counts < min_count].tolist())
    if rare_keys:
        mask = np.isin(keys, list(rare_keys))
        keys = keys.copy()
        keys[mask] = -1
 
    return keys


def _print_split_summary(
    all_x, all_y, train_x, train_y, val_x, val_y, test_x, test_y
) -> None:
    n = len(all_x)
    H, W = all_x.shape[2], all_x.shape[3]
    print(f"\n{'─'*60}")
    print(f"  Dataset loaded — {n:,} images  ({H}×{W} RGB)")
    print(f"  x dtype / range  : {all_x.dtype}  "
          f"[{all_x.min():.3f}, {all_x.max():.3f}]")
    print(f"  Split sizes      : "
          f"train={len(train_x):,}  val={len(val_x):,}  test={len(test_x):,}")
    print(f"\n  Per-label positive-sample frequency:")
    label_counts = all_y.sum(axis=0)
    for label, cnt in zip(LABEL_ORDER, label_counts):
        freq = cnt / n
        bar  = "█" * int(freq * 30)
        print(f"    {label:<12}  {int(cnt):5d}  ({freq:.3f})  {bar}")
    lpi = all_y.sum(axis=1)
    print(f"\n  Labels per image : mean={lpi.mean():.2f}  "
          f"std={lpi.std():.2f}  min={lpi.min():.0f}  max={lpi.max():.0f}")
    print(f"{'─'*60}\n")

class ProjectDataset(Dataset):
    """
    Wraps the numpy arrays returned by load_preprocess_project in a
    PyTorch Dataset, applying an optional transform to each image.
    
    transform: torchvision transform pipeline
    label_smoothing: Epsilon for BCE label smoothing.  Pushes hard 0/1
                    targets to eps/2 and 1-eps/2 respectively, preventing
                    the model from becoming overconfident.
    """

    def __init__(
        self,
        x:               np.ndarray,
        y:               np.ndarray,
        transform=None,
        label_smoothing: float = 0.0,
    ) -> None:
        assert len(x) == len(y), "x and y must have the same number of samples."
        assert 0.0 <= label_smoothing < 0.5, "label_smoothing must be in [0, 0.5)"

        if x.dtype != np.uint8:
            self.x = (x * 255).clip(0, 255).astype(np.uint8)
        else:
            self.x = x

        # BCEWithLogitsLoss needs float32 targets.  Apply smoothing once here
        # so __getitem__ does zero per-sample arithmetic on labels.
        if label_smoothing > 0.0:
            self.y = (y * (1.0 - label_smoothing) +
                      label_smoothing * 0.5).astype(np.float32)
        else:
            self.y = y.astype(np.float32)

        self.transform       = transform
        self.label_smoothing = label_smoothing
        self.classes         = LABEL_ORDER

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        image = Image.fromarray(self.x[idx].transpose(1, 2, 0))

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        target = torch.from_numpy(self.y[idx])
        return image, target


def get_transforms(
    image_size: int  = 128,
    augment:    bool = False,
    mode:       str  = "standard",
) -> transforms.Compose:
    """
    augment: apply training-time augmentations.
    mode: "standard" or "strong" for heavier augmentation
    returns a torchvision transform pipeline for use with ProjectDataset.
    """
    norm = transforms.Normalize(mean=_IMAGENET_MEAN.tolist(),
                                std=_IMAGENET_STD.tolist())

    if not augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            norm,
        ])

    if mode == "strong":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.3, hue=0.15),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])

    #standard augmentation
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.10)),
    ])

def make_loader(
    dataset:     Dataset,
    batch_size:  int  = 64,
    shuffle:     bool = False,
    num_workers: int  = 4,
    drop_last:   bool = False,
    collate_fn         = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
    )

def compute_pos_weights(
    train_y: np.ndarray,
    device:  str = "cpu",
) -> torch.Tensor:
    """
    train_y: (N_train, 12) float32 numpy array of training targets.
    device: Destination device for the returned tensor.
    returns (12,) float tensor on device.
    """
    pos_counts = train_y.sum(axis=0).clip(min=1) #(12,)
    neg_counts = len(train_y) - pos_counts
    pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32)
    return pos_weight.to(device)

def mixup_collate_fn(alpha: float = 0.4):
    """
    higher alpha is more mixing
    returns collate function to pass directly to make_loader or torch.utils.data.DataLoader
    """

    _default_collate = torch.utils.data.default_collate

    def _collate(batch):
        images, targets = _default_collate(batch)

        if alpha <= 0.0:
            return images, targets

        n = images.size(0)
        #sample one lam per batch, simpler and faster than per sample
        lam = float(np.random.beta(alpha, alpha))

        #shuffle indices to create random pairs within the batch
        idx = torch.randperm(n)

        mixed_images  = lam * images  + (1.0 - lam) * images[idx]
        mixed_targets = lam * targets + (1.0 - lam) * targets[idx]

        return mixed_images, mixed_targets

    return _collate


#test with "python data.py --root sp26cai6108mle-project-aggregated/aggregated"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the project dataset")
    parser.add_argument("--root",       type=str,   required=True)
    parser.add_argument("--val_prop",   type=float, default=0.15)
    parser.add_argument("--test_prop",  type=float, default=0.10)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--image_size", type=int,   default=128)
    args = parser.parse_args()
    
    train_x, train_y, test_x, test_y, val_x, val_y = load_preprocess_project(
        root=args.root,
        val_prop=args.val_prop,
        test_prop=args.test_prop,
        normalize=True,
        seed=args.seed,
        verbose=True,
    )

    print("train_x:", train_x.shape, "train_y:", train_y.shape)
    print("val_x:  ", val_x.shape,   "val_y:  ", val_y.shape)
    print("test_x: ", test_x.shape,  "test_y: ", test_y.shape)

    assert train_x.ndim == 4 and train_x.shape[1] == 3
    assert train_y.ndim == 2 and train_y.shape[1] == NUM_CLASSES
    assert train_x.max() <= 1.0 and train_x.min() >= 0.0

    pos_w = compute_pos_weights(train_y)
    print("\nPositive weights (BCEWithLogitsLoss):")
    for label, w in zip(LABEL_ORDER, pos_w.tolist()):
        print(f"  {label:<12}  {w:.3f}")

    train_tf = get_transforms(args.image_size, augment=True)
    train_ds = ProjectDataset(train_x, train_y, transform=train_tf,
                              label_smoothing=0.05)
    del train_x
    loader = make_loader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    imgs, targets = next(iter(loader))
    print(f"\nPlain batch   — images: {tuple(imgs.shape)}  "
          f"targets: {tuple(targets.shape)}  dtype: {imgs.dtype}")
    print(f"  target range: [{targets.min():.3f}, {targets.max():.3f}]  "
          f"(smoothing shifts 0→{0.05/2:.3f}, 1→{1-0.05/2:.3f})")

    mx_loader = make_loader(train_ds, batch_size=8, shuffle=True, num_workers=0,
                            collate_fn=mixup_collate_fn(alpha=0.4))
    mx_imgs, mx_targets = next(iter(mx_loader))
    print(f"\nMixup batch   — images: {tuple(mx_imgs.shape)}  "
          f"targets: {tuple(mx_targets.shape)}")
    print(f"  target range: [{mx_targets.min():.3f}, {mx_targets.max():.3f}]  "
          f"(soft mix of two label vectors)")
    print("All done")