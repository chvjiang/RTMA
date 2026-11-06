import json
import os
import os.path as osp
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class TextToMotionBaseDataset(Dataset):
    """Base dataset for HY-Motion-style text-to-motion training.

    Returns per sample:
        motion:          (L, 201) float32, raw (un-normalized) motion
        motion_mask:     (L,) bool, True for valid frames
        text:            str
        sample_id:       str
    """

    def __init__(
        self,
        motion_dir: str,
        annotation_file: str,
        train_frames: int = 360,
        min_frames: int = 20,
        motion_key: str = "motion",
        length_key: str = "length",
        annotation_format: Optional[str] = None,
        normalize: bool = False,
        mean_std_dir: Optional[str] = None,
        random_crop: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.motion_dir = motion_dir
        self.train_frames = train_frames
        self.min_frames = min_frames
        self.motion_key = motion_key
        self.length_key = length_key
        self.random_crop = random_crop
        self.rng = random.Random(seed)

        # ---- load annotations ----
        self.samples = self._load_annotations(annotation_file, annotation_format)

        # ---- optional normalization ----
        self.normalize = normalize
        if normalize:
            assert mean_std_dir is not None, "mean_std_dir required when normalize=True"
            mean = np.load(osp.join(mean_std_dir, "Mean.npy")).astype(np.float32)
            std = np.load(osp.join(mean_std_dir, "Std.npy")).astype(np.float32)
            assert mean.shape == (201,) and std.shape == (201,), (
                f"mean/std should be (201,), got {mean.shape}/{std.shape}"
            )
            self.mean = torch.from_numpy(mean)
            self.std = torch.from_numpy(std)

    # ------------------------------------------------------------------
    # annotation loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_annotations(path: str, fmt: Optional[str]):
        if fmt is None:
            fmt = osp.splitext(path)[1].lower().lstrip(".")
        if fmt == "json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return [(str(k), str(v)) for k, v in data.items()]
            if isinstance(data, list):
                out = []
                for item in data:
                    if isinstance(item, dict):
                        sid = str(item.get("id") or item.get("sample_id"))
                        txt = str(item.get("text") or item.get("caption"))
                        out.append((sid, txt))
                    else:
                        raise ValueError(f"Unsupported json item: {item}")
                return out
            raise ValueError(f"Unsupported json root type: {type(data)}")
        if fmt == "txt":
            out = []
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    sid, txt = ln.split("#", 1) if "#" in ln else ln.split("\t", 1)
                    out.append((sid.strip(), txt.strip()))
            return out
        if fmt == "csv":
            import csv

            out = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                id_col = header.index("id") if "id" in header else 0
                txt_col = header.index("text") if "text" in header else 1
                for row in reader:
                    out.append((str(row[id_col]).strip(), str(row[txt_col]).strip()))
            return out
        raise ValueError(f"Unsupported annotation format: {fmt}")

    # ------------------------------------------------------------------
    # motion loading
    # ------------------------------------------------------------------
    def _load_motion(self, sample_id: str) -> np.ndarray:
        # try common extensions
        for ext in (".npz", ".npy", ".pkl"):
            path = osp.join(self.motion_dir, sample_id + ext)
            if osp.exists(path):
                if ext == ".npz":
                    with np.load(path) as z:
                        motion = z[self.motion_key]
                        if self.length_key in z:
                            length = int(z[self.length_key])
                            motion = motion[:length]
                elif ext == ".npy":
                    motion = np.load(path)
                else:
                    import pickle

                    with open(path, "rb") as f:
                        obj = pickle.load(f)
                    motion = obj[self.motion_key] if isinstance(obj, dict) else obj
                motion = np.asarray(motion, dtype=np.float32)
                if motion.ndim != 2 or motion.shape[1] != 201:
                    raise ValueError(f"{path}: expected (L,201), got {motion.shape}")
                return motion
        raise FileNotFoundError(f"No motion file for sample {sample_id} in {self.motion_dir}")

    # ------------------------------------------------------------------
    # cropping / padding
    # ------------------------------------------------------------------
    def _crop_or_pad(self, motion: np.ndarray):
        L = motion.shape[0]
        if L < self.min_frames:
            # pad with zeros (invalid frames) up to min_frames
            pad = np.zeros((self.min_frames - L, motion.shape[1]), dtype=motion.dtype)
            motion = np.concatenate([motion, pad], axis=0)
            valid_len = L
            L = self.min_frames
        else:
            valid_len = L

        if L > self.train_frames:
            if self.random_crop:
                start = self.rng.randint(0, L - self.train_frames)
            else:
                start = 0
            motion = motion[start : start + self.train_frames]
            valid_len = min(valid_len, self.train_frames)
            if start > 0:
                valid_len = max(0, valid_len - start)
        elif L < self.train_frames:
            pad = np.zeros((self.train_frames - L, motion.shape[1]), dtype=motion.dtype)
            motion = np.concatenate([motion, pad], axis=0)

        motion_mask = np.zeros((self.train_frames,), dtype=bool)
        motion_mask[:valid_len] = True
        return motion, motion_mask

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id, text = self.samples[idx]
        motion = self._load_motion(sample_id)       # (L_raw, 201) np
        # print(torch.isnan(torch.from_numpy(motion)).any())
        motion, motion_mask = self._crop_or_pad(motion)  # (train_frames,201), (train_frames,)
        # print(torch.isnan(torch.from_numpy(motion)).any())
        motion_t = torch.from_numpy(motion)         # (T, 201)
        # print(torch.isnan(motion_t).any())
        mask_t = torch.from_numpy(motion_mask)      # (T,) bool

        if self.normalize:
            # only normalize valid frames; invalid frames stay zero
            valid = mask_t
            if valid.any():
                motion_t = motion_t.clone()
                std_safe = torch.where(self.std < 1e-3, torch.ones_like(self.std), self.std)
                motion_t[valid] = (motion_t[valid] - self.mean) / std_safe
        # print(torch.isnan(motion_t).any())
        
        return {
            "motion": motion_t,          # (T, 201) float32, normalized if enabled
            "motion_mask": mask_t,       # (T,) bool
            "text": text,                # str
            "sample_id": sample_id,      # str
        }