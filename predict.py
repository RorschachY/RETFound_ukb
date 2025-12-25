#!/usr/bin/env python3

import argparse
import csv
import os
from dataclasses import dataclass
from typing import List

import torch
from PIL import Image
import timm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset_csv import build_csv_transform


@dataclass
class PredictionRow:
    index: int
    probs: List[float]
    pred: int


class CSVPredictDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        transform,
        image_root: str | None = None,
        path_key: str = "path",
    ):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.csv_path = csv_path
        self.transform = transform
        self.image_root = image_root or os.path.dirname(os.path.abspath(csv_path))
        self.path_key = path_key

        self.fieldnames: List[str]
        self.rows: List[dict]
        self.paths: List[str]

        self.fieldnames, self.rows, self.paths = self._load_rows()

    def _load_rows(self):
        rows: List[dict] = []
        paths: List[str] = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"No header found in CSV: {self.csv_path}")

            if self.path_key not in reader.fieldnames:
                raise ValueError(
                    f"Missing required column '{self.path_key}' in CSV {self.csv_path}"
                )

            for row_idx, row in enumerate(reader, start=2):
                raw_path = str(row.get(self.path_key, "")).strip()
                if raw_path == "":
                    raise ValueError(
                        f"Empty image path at row {row_idx} in {self.csv_path}"
                    )
                image_path = self._resolve_path(raw_path)
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(
                        f"Image file not found: {image_path} (row {row_idx})"
                    )
                rows.append(row)
                paths.append(image_path)

        return reader.fieldnames, rows, paths

    def _resolve_path(self, path: str) -> str:
        expanded = os.path.expandvars(os.path.expanduser(path))
        if not os.path.isabs(expanded):
            expanded = os.path.join(self.image_root, expanded)
        return os.path.normpath(expanded)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, index


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Predict with a trained timm model", add_help=True
    )
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--csv_path", required=True, type=str)
    parser.add_argument("--output_csv", default=None, type=str)
    parser.add_argument("--data_path", default="./data/", type=str)
    parser.add_argument("--path_key", default="path", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--nb_classes", default=None, type=int)
    parser.add_argument("--input_size", default=None, type=int)

    parser.add_argument("--color_jitter", type=float, default=None)
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--reprob", type=float, default=0.0)
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)

    return parser


def _load_checkpoint(weights_path):
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return checkpoint, cleaned


def main():
    args = get_args_parser().parse_args()

    checkpoint, state_dict = _load_checkpoint(args.weights)

    if args.nb_classes is None:
        if isinstance(checkpoint, dict) and "args" in checkpoint:
            args.nb_classes = getattr(checkpoint["args"], "nb_classes", None)
        if args.nb_classes is None:
            raise ValueError("--nb_classes is required when not found in checkpoint args")

    if args.input_size is None:
        if isinstance(checkpoint, dict) and "args" in checkpoint:
            args.input_size = getattr(checkpoint["args"], "input_size", None)
        if args.input_size is None:
            args.input_size = 224

    model = timm.create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"Unexpected keys when loading checkpoint: {unexpected}")

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    transform_args = argparse.Namespace(
        input_size=args.input_size,
        color_jitter=args.color_jitter,
        aa=args.aa,
        reprob=args.reprob,
        remode=args.remode,
        recount=args.recount,
    )
    transform = build_csv_transform("test", transform_args)

    dataset = CSVPredictDataset(
        csv_path=args.csv_path,
        transform=transform,
        image_root=args.data_path,
        path_key=args.path_key,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    predictions: List[PredictionRow] = []
    with torch.no_grad():
        for samples, indices in tqdm(data_loader, desc="Predict"):
            samples = samples.to(device, non_blocking=True)
            outputs = model(samples)
            probs = torch.softmax(outputs, dim=1)
            pred_labels = torch.argmax(probs, dim=1)

            for idx, prob_row, pred in zip(indices.tolist(), probs.cpu().tolist(), pred_labels.cpu().tolist()):
                predictions.append(PredictionRow(index=idx, probs=prob_row, pred=pred))

    predictions.sort(key=lambda x: x.index)

    if args.output_csv is None:
        weights_base = os.path.splitext(os.path.basename(args.weights))[0]
        args.output_csv = os.path.join(
            os.path.dirname(args.weights),
            f"predictions_{weights_base}.csv",
        )

    prob_headers = [f"prob_{i}" for i in range(args.nb_classes)]
    output_fieldnames = dataset.fieldnames + prob_headers + ["pred"]

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        for row, pred_row in zip(dataset.rows, predictions):
            output_row = dict(row)
            for i, prob in enumerate(pred_row.probs):
                output_row[f"prob_{i}"] = f"{prob:.6f}"
            output_row["pred"] = pred_row.pred
            writer.writerow(output_row)

    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
