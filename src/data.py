"""
Cryo-ET 2.5D patch sampler and PyTorch Dataset for membrane segmentation.

Design:
- 2.5D input with k=7 slices as channels: X shape (7, 256, 256)
- Supervise ONLY center slice:            Y shape (1, 256, 256)
"""

import random
from typing import Dict, Tuple, Optional, Callable

import numpy as np
import mrcfile
import torch
from torch.utils.data import Dataset
import config


def load_and_preprocess(tomo_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load tomogram and mask from .mrc, binarize mask, robust-normalize tomogram.

    Returns:
        tomo: float32 array, shape (Z,Y,X)
        mask: uint8 array (0/1), shape (Z,Y,X)
    """
    with mrcfile.open(tomo_path, permissive=True) as m:
        tomo = m.data.astype(np.float32)

    with mrcfile.open(mask_path, permissive=True) as m:
        mask = m.data

    if tomo.shape != mask.shape:
        raise ValueError(f"Shape mismatch: tomo{tomo.shape} vs mask{mask.shape}")


    mask = (mask > 0).astype(np.uint8)

    #Normalization
    med = np.median(tomo)
    p5, p95 = np.percentile(tomo, [5, 95])
    tomo = (tomo - med) / (p95 - p5 + 1e-6)
    tomo = np.clip(tomo, -5.0, 5.0)

    return tomo, mask


def make_splits(Z: int, k: int) -> Dict[str, Tuple[int, int]]:
    """
    Define z-slab splits for 2.5D.
    """
    half = k // 2
    z_min = half
    z_max = Z - half - 1

    # Work in the valid center range, not raw Z
    valid_len = z_max - z_min + 1

    train_upper = z_min + int(0.7 * valid_len)
    val_upper   = z_min + int(0.85 * valid_len)

    splits = {
        "train": (z_min, train_upper),
        "val":   (train_upper + 1, val_upper),
        "test":  (val_upper + 1, z_max),
    }

    return splits


def build_anchor_pools(mask: np.ndarray, splits: Dict[str, Tuple[int, int]]) -> Dict[str, np.ndarray]:
    """
    Precompute membrane voxel coordinates per split.

    Returns:
        pools[split] = array of shape (N,3) with columns (z,y,x)
    """
    pools: Dict[str, np.ndarray] = {}

    Z, Y, X = mask.shape
    hk = config.K_SLICES // 2
    hp = config.PATCH_SIZE // 2

    z_min = hk
    z_max = Z - hk - 1
    y_min = hp
    y_max = Y - hp - 1
    x_min = hp
    x_max = X - hp - 1

    for name, (za, zb) in splits.items():
        z0 = max(za, z_min)
        z1 = min(zb, z_max)

        if z0 > z1:
            pools[name] = np.empty((0, 3), dtype=np.int64)
            continue

        coords = np.argwhere(mask[z0:z1 + 1] == 1)
        coords[:, 0] += z0

        valid = (
            (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max) &
            (coords[:, 2] >= x_min) & (coords[:, 2] <= x_max)
        )

        pools[name] = coords[valid]

    return pools



def extract_patch(
    tomo: np.ndarray,
    mask: np.ndarray,
    center: Tuple[int, int, int],
    k: int,
    P: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract one 2.5D patch centered at (z,y,x).

    Returns:
        X_patch: float array (k, P, P)
        Y_patch: uint8/float array (1, P, P) for center slice
    """
    z, y, x = center
    half_k = k // 2
    half_p = P // 2

    # Input stack (k slices)
    X_stack = tomo[z - half_k : z + half_k + 1]  # (k, Y, X)

    # Spatial crop
    y0, y1 = y - half_p, y + half_p
    x0, x1 = x - half_p, x + half_p

    X_patch = X_stack[:, y0:y1, x0:x1]           # (k, P, P)
    Y_patch = mask[z, y0:y1, x0:x1][None, :, :]  # (1, P, P)

    return X_patch, Y_patch


class CryoETDataset(Dataset):
    
    def __init__(
        self,
        split: str = "train",
        length: int = 50000,
    ):
        self.tomo, self.mask = load_and_preprocess(config.TOMO_PATH, config.MASK_PATH)

        self.k = config.K_SLICES
        self.P = config.PATCH_SIZE
        self.p_mem = config.P_MEM
        self.length = int(length)

        self.Z, self.Y, self.X = self.tomo.shape
        self.half_k = self.k // 2
        self.half_p = self.P // 2

        self.splits = make_splits(self.Z, self.k)
        self.split = split
        self.z_min, self.z_max = self.splits[split]

        self.pools = build_anchor_pools(self.mask, self.splits)


    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        memb = (random.random() < self.p_mem)

        if memb and len(self.pools[self.split]) > 0:
            coords = self.pools[self.split]
            z, y, x = coords[random.randint(0, len(coords) - 1)]
        else:
            z = random.randint(self.z_min, self.z_max)
            y = random.randint(self.half_p, self.Y - self.half_p - 1)
            x = random.randint(self.half_p, self.X - self.half_p - 1)

        X_patch, Y_patch = extract_patch(self.tomo, self.mask, (z, y, x), self.k, self.P)

        # Convert to torch tensors
        X_t = torch.from_numpy(X_patch).float()
        Y_t = torch.from_numpy(Y_patch).float()

        return X_t, Y_t

