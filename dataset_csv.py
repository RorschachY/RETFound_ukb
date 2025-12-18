import csv
import os
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class CSVDataset(Dataset):
    """
    CSV-backed dataset that can optionally load soft labels (posterior probabilities)
    for distillation. When `require_soft_labels=True`, probability columns are required.
    Columns:
        - split: which split the sample belongs to (train/val/test)
        - path: image path (relative to image_root or absolute)
        - label: hard label
        - prob_*: optional per-class probabilities (e.g., prob_0, prob_1, ...)
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        transform=None,
        image_root: str | None = None,
        path_key: str = "path",
        label_key: str = "label",
        split_key: str = "split",
        require_soft_labels: bool = False,
        prob_prefix: str = "prob_",
        num_classes: int | None = None,
    ):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.csv_path = csv_path
        self.split = split.lower()
        self.transform = transform
        self.image_root = image_root or os.path.dirname(os.path.abspath(csv_path))
        self.path_key = path_key
        self.label_key = label_key
        self.split_key = split_key
        self.prob_prefix = prob_prefix
        self.num_classes = num_classes
        self.require_soft_labels = require_soft_labels

        paths, raw_labels, soft_labels = self._load_split_rows()
        self.classes, self.class_to_idx = self._build_classes(raw_labels)
        targets = [self.class_to_idx[label] for label in raw_labels]

        self.samples: List[Tuple[str, int]] = list(zip(paths, targets))
        self.targets: List[int] = targets
        self.soft_targets = soft_labels or []
        self.use_soft_labels = soft_labels is not None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.use_soft_labels:
            soft = torch.tensor(self.soft_targets[index], dtype=torch.float32)
            return img, target, soft
        return img, target

    def _load_split_rows(self) -> Tuple[List[str], List[object], List[List[float]] | None]:
        paths: List[str] = []
        labels: List[object] = []
        soft_labels: List[List[float]] | None = [] if self.require_soft_labels else None
        split_value = self.split

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"No header found in CSV: {self.csv_path}")

            missing = [
                key
                for key in (self.path_key, self.label_key, self.split_key)
                if key not in reader.fieldnames
            ]
            if missing:
                raise ValueError(
                    f"Missing required columns {missing} in CSV {self.csv_path}"
                )

            prob_cols = [c for c in reader.fieldnames if c.startswith(self.prob_prefix)]
            if self.require_soft_labels and not prob_cols:
                raise ValueError(
                    f"require_soft_labels=True but no probability columns (prefix '{self.prob_prefix}') "
                    f"found in CSV header of {self.csv_path}"
                )

            def _prob_sort_key(name: str):
                suffix = name[len(self.prob_prefix):]
                return int(suffix) if suffix.isdigit() else suffix

            prob_cols = sorted(prob_cols, key=_prob_sort_key)

            for row_idx, row in enumerate(reader, start=2):
                row_split = str(row.get(self.split_key, "")).strip().lower()
                if row_split != split_value:
                    continue

                raw_path = str(row.get(self.path_key, "")).strip()
                raw_label = row.get(self.label_key, "")

                if raw_path == "":
                    raise ValueError(
                        f"Empty image path at row {row_idx} in {self.csv_path}"
                    )
                if raw_label is None or str(raw_label).strip() == "":
                    raise ValueError(
                        f"Empty label at row {row_idx} in {self.csv_path}"
                    )

                image_path = self._resolve_path(raw_path)
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(
                        f"Image file not found: {image_path} (row {row_idx})"
                    )

                label_value = self._coerce_label(raw_label)
                paths.append(image_path)
                labels.append(label_value)

                if soft_labels is not None:
                    if not prob_cols:
                        raise ValueError(
                            f"Soft labels required but no probability columns found at row {row_idx} in {self.csv_path}"
                        )
                    raw_probs = [row.get(c, "") for c in prob_cols]
                    try:
                        probs = [float(p) for p in raw_probs]
                    except Exception as exc:  # noqa: BLE001
                        raise ValueError(
                            f"Invalid probability values at row {row_idx} in {self.csv_path}: {raw_probs}"
                        ) from exc

                    if self.num_classes is not None and len(probs) != self.num_classes:
                        raise ValueError(
                            f"Expected {self.num_classes} probability columns (prefix '{self.prob_prefix}') "
                            f"but got {len(probs)} at row {row_idx}"
                        )

                    prob_tensor = torch.tensor(probs, dtype=torch.float32)
                    prob_sum = prob_tensor.sum()
                    if prob_sum > 0:
                        prob_tensor = prob_tensor / prob_sum
                    soft_labels.append(prob_tensor.tolist())

        if not paths:
            raise ValueError(
                f"No samples found for split '{self.split}' in CSV {self.csv_path}"
            )

        return paths, labels, soft_labels

    def _resolve_path(self, path: str) -> str:
        expanded = os.path.expandvars(os.path.expanduser(path))
        if not os.path.isabs(expanded):
            expanded = os.path.join(self.image_root, expanded)
        return os.path.normpath(expanded)

    @staticmethod
    def _coerce_label(label_value: object) -> object:
        value = label_value
        if isinstance(value, str):
            value = value.strip()

        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _build_classes(raw_labels: Sequence[object]) -> Tuple[List[object], dict]:
        def sort_key(val: object):
            if isinstance(val, int):
                return (0, val)
            return (1, str(val))

        classes = sorted(set(raw_labels), key=sort_key)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        return classes, class_to_idx


def build_csv_dataset(is_train: str, args):
    """
    Build dataset based on CSV annotations, with optional soft-label distillation.

    Args:
        is_train: split name ('train', 'val', or 'test')
        args: configuration object that should contain at least `csv_path`.
              Optional overrides: csv_path_key, csv_label_key, csv_split_key, image_root.
              For distillation: set args.use_soft_labels=True and prob columns with prefix (default prob_).
    """
    transform = build_csv_transform(is_train, args)

    csv_path = getattr(args, "csv_path", None) or getattr(args, "csv_file", None)
    if csv_path is None:
        raise ValueError("Please provide `csv_path` (or `csv_file`) in args for CSV-based dataset.")

    dataset = CSVDataset(
        csv_path=csv_path,
        split=is_train,
        transform=transform,
        image_root=getattr(args, "image_root", None) or getattr(args, "data_path", None),
        path_key=getattr(args, "csv_path_key", "path"),
        label_key=getattr(args, "csv_label_key", "label"),
        split_key=getattr(args, "csv_split_key", "split"),
        require_soft_labels=bool(getattr(args, "use_soft_labels", False)),
        prob_prefix=getattr(args, "csv_prob_prefix", "prob_"),
        num_classes=getattr(args, "nb_classes", None),
    )

    if is_train == "train":
        ratio = float(getattr(args, "dataratio", 1.0))
        seed = int(getattr(args, "seed", 0))
        stratified = bool(getattr(args, "stratified", False))

        if 0.0 < ratio < 1.0:
            if stratified:
                idx = _stratified_indices(dataset.targets, ratio, seed)
            else:
                g = torch.Generator().manual_seed(seed)
                n = len(dataset)
                k = max(1, int(n * ratio))
                idx = torch.randperm(n, generator=g)[:k].tolist()
            dataset = Subset(dataset, idx)

    return dataset


def build_csv_transform(is_train: str, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train == "train":
        return create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )

    crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
    size = int(args.input_size / crop_pct)
    t = [
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(t)


def _stratified_indices(targets: Sequence[int], ratio: float, seed: int):
    t = torch.as_tensor(targets)
    classes = torch.unique(t)
    g = torch.Generator().manual_seed(seed)

    keep: List[int] = []
    for c in classes.tolist():
        cls_idx = torch.nonzero(t == c, as_tuple=False).view(-1)
        if len(cls_idx) == 0:
            continue
        k = max(1, int(round(len(cls_idx) * ratio)))
        sel = cls_idx[torch.randperm(len(cls_idx), generator=g)[:k]]
        keep.extend(sel.tolist())

    g2 = torch.Generator().manual_seed(seed + 1)
    keep = torch.tensor(keep)[torch.randperm(len(keep), generator=g2)].tolist()
    return keep


# Backward-compatible aliases
build_dataset = build_csv_dataset
build_transform = build_csv_transform
