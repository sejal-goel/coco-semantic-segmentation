"""Dataset and loaders. One shared pipeline for BOTH models (fair comparison).

Reads the masks written by the masks walkthrough:
  masks_index.csv  - one row per image: which photo belongs to which mask PNG
  masks/{train,val,test}/*.png - single-channel class-id masks (0 = background)

The masks-index column names are auto-detected, because the exact header
depends on how the walkthrough saved it. If detection fails the error
message tells you exactly what to check (send it to Instinct).
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # tolerate truncated image files (hit during the Colab runs)

import torch
from torch.utils.data import Dataset, DataLoader

import config as C


# ------------------------------------------------------- index resolution
def _find_col(cols, needles):
    for col in cols:
        low = col.lower()
        if any(n in low for n in needles):
            return col
    return None


def resolve_masks_index(index_path=None, verbose=True):
    """Return a DataFrame with columns [image, mask, split]."""
    index_path = index_path or C.MASKS_INDEX
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"masks index not found at {index_path}. Did the masks "
            f"walkthrough (Cell 4) finish? It writes masks_index.csv.")
    df = pd.read_csv(index_path)
    img_col = _find_col(df.columns, ['image', 'filepath', 'photo', 'img'])
    msk_col = _find_col(df.columns, ['mask'])
    if img_col is None or msk_col is None or img_col == msk_col:
        raise ValueError(
            f"Could not auto-detect image/mask columns. Found: {list(df.columns)}. "
            f"Tell Instinct these column names.")
    split_col = _find_col(df.columns, ['split'])
    out = pd.DataFrame({'image': df[img_col].astype(str),
                        'mask': df[msk_col].astype(str)})
    if split_col is not None:
        out['split'] = (df[split_col].astype(str).str.lower()
                        .str.replace('validation', 'val').str.replace('val_split', 'val')
                        .str.replace('test_split', 'test'))
    else:
        # fall back to which masks/<split>/ folder the mask sits in
        def split_from_path(p):
            for s in ('train', 'val', 'test'):
                if f'/{s}/' in p.replace('\\', '/'):
                    return s
            return 'unknown'
        out['split'] = out['mask'].map(split_from_path)

    def fix(p, kind):
        p = p.replace('\\', '/')
        if os.path.exists(p):
            return p
        base = os.path.basename(p)
        roots = ([C.MASKS_DIR] if kind == 'mask'
                 else glob.glob(os.path.join(C.ZOO_DIR, 'coco-2017', '*', 'data')))
        for root in roots:
            hits = glob.glob(os.path.join(root, '**', base), recursive=True)
            if hits:
                return hits[0]
        return p  # leave broken; dataset init reports it clearly

    out['image'] = out['image'].map(lambda p: fix(p, 'image'))
    out['mask'] = out['mask'].map(lambda p: fix(p, 'mask'))
    n_missing = int((~out['image'].map(os.path.exists)).sum()
                    + (~out['mask'].map(os.path.exists)).sum())
    if verbose:
        print(f"masks index: {len(out)} rows | splits: "
              f"{out['split'].value_counts().to_dict()} | missing files: {n_missing}")
        print(out.head(2).to_string())
    if n_missing:
        raise FileNotFoundError(f"{n_missing} files from the index are missing on Drive. "
                                f"Send this message to Instinct.")
    return out


# ------------------------------------------------------- dataset
class CocoSegDataset(Dataset):
    """(image tensor 3xHxW float, mask HxW long) with identical resize for
    both models. Train mode applies the augmentation pipeline."""

    def __init__(self, frame, split, img_size=C.IMG_SIZE, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.split = split
        self.size = img_size
        self.augment = augment
        bad = [i for i in range(min(len(self.frame), 25))
               if not (os.path.exists(self.frame['image'][i]) and os.path.exists(self.frame['mask'][i]))]
        if bad:
            raise FileNotFoundError(f"Missing files, e.g. row {bad[0]}: "
                                    f"{self.frame['image'][bad[0]]}")

    def __len__(self):
        return len(self.frame)

    def _augment(self, img, mask, rng):
        # horizontal flip
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        # scale jitter 0.75 - 1.25, then center crop / pad back to size
        s = float(rng.uniform(0.75, 1.25))
        w = h = self.size
        nw, nh = max(8, int(round(w * s))), max(8, int(round(h * s)))
        img = img.resize((nw, nh), Image.BILINEAR)
        mask = mask.resize((nw, nh), Image.NEAREST)
        canvas_i = Image.new('RGB', (w, h))
        canvas_m = Image.new('L', (w, h))
        ox = (nw - w) // 2
        oy = (nh - h) // 2
        # crop if larger, paste if smaller
        src = (max(0, ox), max(0, oy), max(0, ox) + w, max(0, oy) + h)
        img = img.crop(src) if nw >= w and nh >= h else img
        mask = mask.crop(src) if nw >= w and nh >= h else mask
        if img.size != (w, h):
            img = img.resize((w, h), Image.BILINEAR)
            mask = mask.resize((w, h), Image.NEAREST)
        # colour jitter (image only, never the mask)
        img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.8, 1.2)))
        img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.8, 1.2)))
        img = ImageEnhance.Color(img).enhance(float(rng.uniform(0.8, 1.2)))
        return img, mask

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        img = Image.open(row['image']).convert('RGB').resize(
            (self.size, self.size), Image.BILINEAR)
        mask = Image.open(row['mask']).resize(
            (self.size, self.size), Image.NEAREST)
        if self.augment:
            rng = np.random.default_rng(torch.initial_seed() % (2**31) + i)
            img, mask = self._augment(img, mask, rng)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1)
        mean = torch.tensor(C.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(C.IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std
        y = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        return x, y


def get_loader(frame, split, batch_size, augment=False, subset=None,
               seed=42, shuffle=None, num_workers=C.NUM_WORKERS):
    """One loader factory used by every experiment."""
    if subset:
        frame = frame.sample(n=min(subset, len(frame)), random_state=seed)
    ds = CocoSegDataset(frame, split, augment=augment)
    if shuffle is None:
        shuffle = (split == 'train')
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=shuffle)


def load_frames():
    """Resolve the index once and return {'train','val','test'} DataFrames."""
    idx = resolve_masks_index()
    frames = {s: idx[idx.split == s] for s in ('train', 'val', 'test')}
    for s, f in frames.items():
        if len(f) == 0:
            raise ValueError(f"Split '{s}' is empty. Splits found: "
                             f"{sorted(idx.split.unique())}. Send this to Instinct.")
    return frames
