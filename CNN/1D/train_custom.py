#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import warnings
import collections
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    f1_score
)

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainCfg:
    root_dir: str
    save_dir: str
    mem_regions: str           # stack | data_stack | all
    mode: str                  # rgb | gray
    model_name: str

    epochs: int
    batch_size: int
    lr: float
    weight_decay: float

    hidden_dim: int
    use_attention: bool
    dropout: float

    loss: str                  # focal | ce
    augment: bool

    train_n: int
    val_n: int

    apk_agg: str               # mean | max  (validation aggregation)
    threshold: str             # f1 | youden (threshold selection on region-level, as in your improved RGB)
    seed: int


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Losses
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


# -----------------------------
# Model
# -----------------------------
class OneDCNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int = 2,
                 hidden_dim: int = 128, use_attention: bool = True, dropout: float = 0.3):
        super().__init__()
        self.use_attention = bool(use_attention)

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),

            nn.AdaptiveMaxPool1d(32)
        )

        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=1),
                nn.Sigmoid()
            )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        if self.use_attention:
            x = x * self.attention(x)
        return self.classifier(x)


def logits_to_pos_prob(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1]


# -----------------------------
# Dataset
# -----------------------------
class ApkRegionDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        mem_regions: str,
        mode: str,
        apk_list: Optional[List[str]] = None,
        class_filter: Optional[int] = None,
        augment: bool = False
    ):
        super().__init__()
        self.mode = mode.lower()
        self.augment = bool(augment)

        self.selection_map = {
            "stack": [r"\[stack\]"],
            "data_stack": [r"^/data/.*", r"\[stack\]"],
            "all": []
        }

        class_dirs = {0: "benign_dumps", 1: "dumps_dataset"}
        if class_filter is not None:
            class_dirs = {int(class_filter): class_dirs[int(class_filter)]}

        self.samples: List[Tuple[str, int, str, str]] = []

        for label, cdir in class_dirs.items():
            base_path = os.path.join(root_dir, cdir)
            if not os.path.exists(base_path):
                continue

            apks = apk_list if apk_list is not None else sorted(os.listdir(base_path))
            for apk in tqdm(apks, desc=f"Scanning {cdir}", unit="apk"):
                apk_path = os.path.join(base_path, apk)
                maps_path = os.path.join(apk_path, "maps.txt")
                img_dir = os.path.join(apk_path, "images", "horizontal", "RGB")

                if not (os.path.exists(maps_path) and os.path.exists(img_dir)):
                    continue

                try:
                    with open(maps_path, "r", errors="ignore") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 6:
                                continue
                            addr = parts[0].split("-")[0]
                            desc = line.split(None, 5)[-1] if len(line.split(None, 5)) >= 6 else ""

                            matched = (mem_regions == "all") or any(
                                re.search(p, desc) for p in self.selection_map.get(mem_regions, [])
                            )
                            if not matched:
                                continue

                            img_path = os.path.join(img_dir, f"0x{addr}_dump_RGB_horizontal.png")
                            if os.path.exists(img_path):
                                self.samples.append((img_path, int(label), os.path.basename(apk_path), desc))
                except Exception as e:
                    logger.warning(f"[{apk_path}] skipped: {e}")

    def __len__(self):
        return len(self.samples)

    def _rgb(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert("RGB").resize((1024, 3))
        arr = np.array(img).astype(np.float32) / 255.0   # [3,1024,3]
        ten = torch.from_numpy(arr).permute(2, 1, 0)     # [3,1024,3]
        ten = ten.mean(dim=2)                            # [3,1024]
        if self.augment:
            ten = ten + 0.01 * torch.randn_like(ten)
            ten = torch.clamp(ten, 0, 1)
        return ten

    def _gray(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert("L").resize((1024, 1))
        arr = np.array(img).astype(np.float32).squeeze() / 255.0  # (1024,)
        ten = torch.from_numpy(arr).unsqueeze(0)                  # [1,1024]
        if self.augment:
            ten = ten + 0.01 * torch.randn_like(ten)
            ten = torch.clamp(ten, 0, 1)
        return ten

    def __getitem__(self, idx):
        img_path, label, apk, desc = self.samples[idx]
        try:
            x = self._rgb(img_path) if self.mode == "rgb" else self._gray(img_path)
            return x, torch.tensor(label, dtype=torch.long), apk, desc
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))


# -----------------------------
# Train helpers
# -----------------------------
def split_apks(root_dir: str, train_n: int, val_n: int):
    benign = sorted(os.listdir(os.path.join(root_dir, "benign_dumps")))
    malware = sorted(os.listdir(os.path.join(root_dir, "dumps_dataset")))
    return (
        benign[:train_n], benign[train_n:train_n + val_n],
        malware[:train_n], malware[train_n:train_n + val_n],
    )


def make_sampler(train_b: ApkRegionDataset, train_m: ApkRegionDataset):
    labels = [lbl for (_, lbl, _, _) in train_b.samples] + [lbl for (_, lbl, _, _) in train_m.samples]
    counts = np.bincount(labels)
    counts = np.maximum(counts, 1)
    w = 1.0 / torch.tensor(counts, dtype=torch.float)
    sample_w = [w[int(l)] for l in labels]
    return WeightedRandomSampler(sample_w, len(sample_w))


def save_ckpt(path: str, model: nn.Module, cfg: TrainCfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg)}, path)


# -----------------------------
# Main train
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", default="/mnt/malware_ram/Android")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--mem_regions", default="data_stack", choices=["stack", "data_stack", "all"])
    ap.add_argument("--mode", default="rgb", choices=["rgb", "gray"])
    ap.add_argument("--model_name", default="1dcnn")

    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--hidden_dim", type=int, default=None)
    ap.add_argument("--use_attention", action="store_true")
    ap.add_argument("--no_attention", action="store_true")
    ap.add_argument("--dropout", type=float, default=None)

    ap.add_argument("--loss", default=None, choices=["focal", "ce"])
    ap.add_argument("--augment", action="store_true")

    ap.add_argument("--train_n", type=int, default=1300)
    ap.add_argument("--val_n", type=int, default=200)

    ap.add_argument("--apk_agg", default="max", choices=["mean", "max"])
    ap.add_argument("--threshold", default="f1", choices=["f1", "youden"])
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    mode = args.mode.lower()
    set_seed(args.seed)

    # Defaults aligned with your previous scripts
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else (128 if mode == "rgb" else 256)
    dropout = args.dropout if args.dropout is not None else (0.3 if mode == "rgb" else 0.4)

    if args.no_attention:
        use_attention = False
    elif args.use_attention:
        use_attention = True
    else:
        use_attention = (mode == "rgb")  # rgb: True, gray: False by default

    loss = args.loss if args.loss is not None else ("focal" if mode == "rgb" else "ce")

    cfg = TrainCfg(
        root_dir=args.root_dir,
        save_dir=args.save_dir,
        mem_regions=args.mem_regions,
        mode=mode,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=int(hidden_dim),
        use_attention=bool(use_attention),
        dropout=float(dropout),
        loss=loss,
        augment=bool(args.augment),
        train_n=args.train_n,
        val_n=args.val_n,
        apk_agg=args.apk_agg,
        threshold=args.threshold,
        seed=args.seed
    )

    os.makedirs(cfg.save_dir, exist_ok=True)
    log_path = os.path.join(cfg.save_dir, f"train_{cfg.model_name}_{cfg.mode}_{cfg.mem_regions}.log")
    logger.remove()
    logger.add(log_path, format="{message}")
    logger.info(json.dumps(asdict(cfg), indent=2))

    benign_train, benign_val, malware_train, malware_val = split_apks(cfg.root_dir, cfg.train_n, cfg.val_n)

    train_b = ApkRegionDataset(cfg.root_dir, cfg.mem_regions, cfg.mode, benign_train, class_filter=0, augment=cfg.augment)
    train_m = ApkRegionDataset(cfg.root_dir, cfg.mem_regions, cfg.mode, malware_train, class_filter=1, augment=cfg.augment)
    val_b = ApkRegionDataset(cfg.root_dir, cfg.mem_regions, cfg.mode, benign_val, class_filter=0, augment=False)
    val_m = ApkRegionDataset(cfg.root_dir, cfg.mem_regions, cfg.mode, malware_val, class_filter=1, augment=False)

    train_ds = ConcatDataset([train_b, train_m])
    val_ds = ConcatDataset([val_b, val_m])

    sampler = make_sampler(train_b, train_m)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    in_ch = 3 if cfg.mode == "rgb" else 1
    model = OneDCNN(in_ch, hidden_dim=cfg.hidden_dim, use_attention=cfg.use_attention, dropout=cfg.dropout).to(DEVICE)

    criterion = FocalLoss() if cfg.loss == "focal" else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))

    hist = {"train_loss": [], "val_loss": [], "val_acc": [], "roc_auc": [], "pr_auc": []}
    best_val_acc = -1.0
    best_train_acc = -1.0

    for epoch in tqdm(range(cfg.epochs), desc="Training", unit="epoch"):
        # ---- train
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for x, y, _, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            lossv = criterion(out, y)
            lossv.backward()
            optimizer.step()

            run_loss += float(lossv.item()) * x.size(0)
            correct += int((out.argmax(1) == y).sum().item())
            total += int(y.size(0))

        train_loss = run_loss / max(total, 1)
        train_acc = (correct / max(total, 1)) * 100.0
        hist["train_loss"].append(train_loss)

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            save_ckpt(os.path.join(cfg.save_dir, f"ckpt_{cfg.model_name}_{cfg.mode}_{cfg.mem_regions}_best_train.pth"), model, cfg)

        # ---- val (region probs + apk aggregation)
        model.eval()
        val_loss = 0.0

        region_probs, region_labels = [], []
        apk_probs = collections.defaultdict(list)
        apk_labels = {}
        apk_regions = collections.defaultdict(list)

        with torch.no_grad():
            for x, y, apks, regions in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                lossv = criterion(out, y)
                val_loss += float(lossv.item()) * x.size(0)

                probs = logits_to_pos_prob(out).detach().cpu().numpy()
                for i in range(len(probs)):
                    apk = apks[i]
                    lbl = int(y[i].item())
                    p = float(probs[i])
                    reg = regions[i]

                    region_labels.append(lbl)
                    region_probs.append(p)

                    apk_probs[apk].append(p)
                    apk_labels[apk] = lbl
                    apk_regions[apk].append((reg, p))

        val_loss /= max(len(val_loader.dataset), 1)

        # threshold selection on region-level (like your improved rgb script)
        if len(set(region_labels)) < 2:
            thr = 0.5
            auc = pr_auc = 0.0
        else:
            if cfg.threshold == "f1":
                best_thr, best_f1 = 0.5, -1.0
                for t in np.linspace(0.1, 0.9, 81):
                    preds = [int(p > t) for p in region_probs]
                    f1 = f1_score(region_labels, preds)
                    if f1 > best_f1:
                        best_f1, best_thr = f1, float(t)
                thr = best_thr
            else:
                fpr, tpr, thr_list = roc_curve(region_labels, region_probs)
                thr = float(thr_list[int(np.argmax(tpr - fpr))])

            auc = float(roc_auc_score(region_labels, region_probs))
            pr_auc = float(average_precision_score(region_labels, region_probs))

        # apk aggregation
        apk_scores = {}
        for apk, ps in apk_probs.items():
            apk_scores[apk] = float(np.max(ps)) if cfg.apk_agg == "max" else float(np.mean(ps))

        apk_pred = {apk: int(score > thr) for apk, score in apk_scores.items()}
        apk_true = [apk_labels[a] for a in apk_pred]
        apk_hat = [apk_pred[a] for a in apk_pred]
        val_acc = float((np.array(apk_true) == np.array(apk_hat)).mean() * 100.0)

        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        hist["roc_auc"].append(auc)
        hist["pr_auc"].append(pr_auc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_ckpt(os.path.join(cfg.save_dir, f"ckpt_{cfg.model_name}_{cfg.mode}_{cfg.mem_regions}_best_val.pth"), model, cfg)

        logger.info(f"Epoch {epoch+1}/{cfg.epochs} | TrainLoss={train_loss:.4f} TrainAcc={train_acc:.2f}% "
                    f"| ValLoss={val_loss:.4f} ValAPKAcc={val_acc:.2f}% | thr={thr:.4f} auc={auc:.4f} pr={pr_auc:.4f}")
        logger.info("APK-level report:\n" + classification_report(apk_true, apk_hat, target_names=["Benign", "Malicious"], zero_division=0))
        logger.info("APK-level CM:\n" + str(confusion_matrix(apk_true, apk_hat)))

        logger.info("\nMisclassified APKs:")
        for apk in apk_pred:
            if apk_pred[apk] != apk_labels[apk]:
                logger.info(f"APK={apk} True={apk_labels[apk]} Pred={apk_pred[apk]} Score={apk_scores[apk]:.4f}")
                for reg, p in sorted(apk_regions[apk], key=lambda x: x[1], reverse=True):
                    logger.info(f"    Region: {reg} | Prob: {p:.4f}")

        scheduler.step()

    # plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist["train_loss"], label="Train Loss")
    plt.plot(hist["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epoch")

    plt.subplot(1, 2, 2)
    plt.plot(hist["val_acc"], label="Val APK Acc")
    plt.plot(hist["roc_auc"], label="ROC AUC (region)")
    plt.plot(hist["pr_auc"], label="PR AUC (region)")
    plt.legend()
    plt.title("Metrics")
    plt.xlabel("Epoch")

    plt.tight_layout()
    out_plot = os.path.join(cfg.save_dir, f"train_{cfg.model_name}_{cfg.mode}_{cfg.mem_regions}_metrics.png")
    plt.savefig(out_plot)
    plt.show()

    logger.info(f"Saved plot: {out_plot}")
    logger.info(f"Best Val APK Acc: {best_val_acc:.2f}% | Best Train Acc: {best_train_acc:.2f}%")


if __name__ == "__main__":
    main()