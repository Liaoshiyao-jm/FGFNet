import csv
import os
import random
from pathlib import Path

import numpy as np
from skimage import io
from torch import from_numpy
from torch.utils.data import Dataset


LABELS = ["background", "farmland", "city", "village", "water", "forest", "road", "others"]
N_CLASSES = len(LABELS)


def read_ids(dataset_root, split):
    dataset_root = Path(dataset_root)
    split_file = dataset_root / f"{split}_ids.csv"
    all_file = dataset_root / "data.csv"

    if split_file.exists():
        with split_file.open("r", encoding="utf-8") as f:
            return [row[0] for row in csv.reader(f) if row]

    with all_file.open("r", encoding="utf-8") as f:
        file_names = [row[0] for row in csv.reader(f) if row]
    num_train = int(len(file_names) * 0.9)
    return file_names[:num_train] if split == "train" else file_names[num_train:]


class WHUDataIO:
    def __init__(self, dataset_root, ids):
        self.dataset_root = Path(dataset_root)
        self.ids = ids
        self.LABELS = LABELS
        self.N_CLASSES = N_CLASSES
        self.dataName = "WHU"

        self.opt_files = [self.dataset_root / "optical" / f"{idx}.tif" for idx in ids]
        self.sar_files = [self.dataset_root / "sar" / f"{idx}.tif" for idx in ids]
        self.lbl_files = [self.dataset_root / "lbl" / f"{idx}.tif" for idx in ids]
        self._check_files()

    def _check_files(self):
        for file_path in self.opt_files + self.sar_files + self.lbl_files:
            if not file_path.is_file():
                raise FileNotFoundError(file_path)

    def opt_io(self, idx):
        data = io.imread(self.opt_files[idx])[:, :, :3]
        return np.asarray(data.transpose((2, 0, 1)), dtype="float32") / 255.0

    def sar_io(self, idx):
        sar = np.asarray(io.imread(self.sar_files[idx]), dtype="float32")
        sar = (sar - np.min(sar)) / (np.max(sar) - np.min(sar) + 1e-8)
        if len(sar.shape) == 2:
            sar = np.expand_dims(sar, axis=0)
        return sar

    def lbl_io(self, idx):
        return np.asarray(io.imread(self.lbl_files[idx]), dtype="int64") // 10


class WHUTrainDataset(Dataset):
    def __init__(self, dataset_root, ids=None, set_len=10000, window_size=(256, 256), augmentation=True, cache=True):
        if ids is None:
            ids = read_ids(dataset_root, "train")
        self.Data = WHUDataIO(dataset_root, ids)
        self.idx_len = len(ids)
        self.set_len = set_len
        self.window_size = window_size
        self.augmentation = augmentation
        self.cache = {} if cache else None

    def __len__(self):
        return self.set_len

    def __getitem__(self, _):
        image_idx = random.randint(0, self.idx_len - 1)
        if self.cache is not None and image_idx in self.cache:
            data = self.cache[image_idx]
            opt, sar, lbl = data["opt"], data["sar"], data["lbl"]
        else:
            opt = self.Data.opt_io(image_idx)
            sar = self.Data.sar_io(image_idx)
            lbl = self.Data.lbl_io(image_idx)
            if self.cache is not None:
                self.cache[image_idx] = {"opt": opt, "sar": sar, "lbl": lbl}

        x1, x2, y1, y2 = self._random_window(lbl.shape[-2:])
        opt_patch = opt[:, x1:x2, y1:y2]
        sar_patch = sar[:, x1:x2, y1:y2]
        lbl_patch = lbl[x1:x2, y1:y2]

        if self.augmentation:
            opt_patch, sar_patch, lbl_patch = self._augment(opt_patch, sar_patch, lbl_patch)
        return from_numpy(opt_patch), from_numpy(sar_patch), from_numpy(lbl_patch)

    def _random_window(self, input_shape):
        patch_w, patch_h = self.window_size
        image_w, image_h = input_shape
        x1 = random.randint(0, image_w - patch_w - 1)
        y1 = random.randint(0, image_h - patch_h - 1)
        return x1, x1 + patch_w, y1, y1 + patch_h

    @staticmethod
    def _augment(*arrays):
        flip = random.random() < 0.5
        mirror = random.random() < 0.5
        results = []
        for array in arrays:
            if flip:
                array = array[::-1, :] if len(array.shape) == 2 else array[:, ::-1, :]
            if mirror:
                array = array[:, ::-1] if len(array.shape) == 2 else array[:, :, ::-1]
            results.append(np.copy(array))
        return tuple(results)


class WHUTestDataset(Dataset):
    def __init__(self, dataset_root, ids=None, window_size=(256, 256), stride=64):
        if ids is None:
            ids = read_ids(dataset_root, "test")
        self.Data = WHUDataIO(dataset_root, ids)
        self.window_size = window_size
        self.stride = stride
        self.test_ixy = []
        self.lbl_shape_list = []
        for idx in range(len(ids)):
            lbl_shape = self.Data.lbl_io(idx).shape[-2:]
            self.lbl_shape_list.append(lbl_shape)
            for x, y in self._sliding_window(lbl_shape):
                self.test_ixy.append((idx, x, y))
        self.cache = {"idx": -1}

    def __len__(self):
        return len(self.test_ixy)

    def __getitem__(self, i):
        idx, x, y = self.test_ixy[i]
        patch_w, patch_h = self.window_size
        if self.cache["idx"] == idx:
            opt, sar = self.cache["opt"], self.cache["sar"]
        else:
            opt = self.Data.opt_io(idx)
            sar = self.Data.sar_io(idx)
            self.cache = {"idx": idx, "opt": opt, "sar": sar}
        return from_numpy(np.copy(opt[:, x:x + patch_w, y:y + patch_h])), from_numpy(np.copy(sar[:, x:x + patch_w, y:y + patch_h]))

    def get_lbl(self):
        return [self.Data.lbl_io(idx) for idx in range(len(self.lbl_shape_list))]

    def _sliding_window(self, shape):
        image_w, image_h = shape
        patch_w, patch_h = self.window_size
        if image_w < patch_w or image_h < patch_h:
            raise ValueError(f"window_size {self.window_size} is larger than image shape {shape}.")

        max_x = image_w - patch_w
        max_y = image_h - patch_h
        x = 0
        while True:
            y = 0
            while True:
                yield x, y
                if y == max_y:
                    break
                y = min(y + self.stride, max_y)
            if x == max_x:
                break
            x = min(x + self.stride, max_x)
