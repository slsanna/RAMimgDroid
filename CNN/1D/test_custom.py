#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import warnings
import collections
from typing import List, Optional, Tuple, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score
)

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Model (same as training)
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
# Dataset (same preprocessing)
# -----------------------------
class ApkRegionDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        mem_regions: str,
        mode: str,
        apk_list: Optional[List[str]] = None,
        class_filter: Optional[int] = None
    ):
        super().__init__()
        self.mode = mode.lower()

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
                apk_path = apk if os.path.isabs(apk) else os.path.join(base_path, apk)
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
        arr = np.array(img).astype(np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 1, 0).mean(dim=2)  # [3,1024]
        return ten

    def _gray(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert("L").resize((1024, 1))
        arr = np.array(img).astype(np.float32).squeeze() / 255.0
        ten = torch.from_numpy(arr).unsqueeze(0)  # [1,1024]
        return ten

    def __getitem__(self, idx):
        img_path, label, apk, desc = self.samples[idx]
        try:
            x = self._rgb(img_path) if self.mode == "rgb" else self._gray(img_path)
            return x, torch.tensor(label, dtype=torch.long), apk, desc
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))


def load_ckpt(ckpt_path: str) -> Tuple[Dict, Dict]:
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"], state.get("cfg", {})
    return state, {}


def build_model_from_cfg(cfg: Dict, fallback_mode: str) -> OneDCNN:
    mode = str(cfg.get("mode", fallback_mode)).lower()
    in_ch = 3 if mode == "rgb" else 1
    hidden_dim = int(cfg.get("hidden_dim", 128 if mode == "rgb" else 256))
    use_attention = bool(cfg.get("use_attention", mode == "rgb"))
    dropout = float(cfg.get("dropout", 0.3 if mode == "rgb" else 0.4))
    return OneDCNN(in_ch, hidden_dim=hidden_dim, use_attention=use_attention, dropout=dropout).to(DEVICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", default="/mnt/malware_ram/Android")
    ap.add_argument("--mem_regions", default="data_stack", choices=["stack", "data_stack", "all"])
    ap.add_argument("--mode", default="rgb", choices=["rgb", "gray"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--save_dir", required=True)

    ap.add_argument("--apk_agg", default="mean", choices=["mean", "max"])
    ap.add_argument("--threshold", default="youden", choices=["youden", "f1"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--test_start_idx", type=int, default=1500)
    ap.add_argument("--single_apk_path", default=None)

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, f"test_{os.path.basename(args.ckpt).replace('.pth','')}.log")
    logger.remove()
    logger.add(log_path, format="{message}")
    logger.info(f"[TEST] root={args.root_dir} mem={args.mem_regions} mode={args.mode} ckpt={args.ckpt} agg={args.apk_agg} thr={args.threshold}")

    # dataset
    if args.single_apk_path:
        class_filter = 1 if "dumps_dataset" in args.single_apk_path else 0
        test_ds = ApkRegionDataset(args.root_dir, args.mem_regions, args.mode, apk_list=[args.single_apk_path], class_filter=class_filter)
    else:
        benign_dir = os.path.join(args.root_dir, "benign_dumps")
        malware_dir = os.path.join(args.root_dir, "dumps_dataset")

        benign_apks = sorted(os.listdir(benign_dir)) if os.path.exists(benign_dir) else []
        malware_apks = sorted(os.listdir(malware_dir)) if os.path.exists(malware_dir) else []

        benign_test = [os.path.join(benign_dir, a) for i, a in enumerate(benign_apks) if i >= args.test_start_idx]
        malware_test = [os.path.join(malware_dir, a) for i, a in enumerate(malware_apks) if i >= args.test_start_idx]

        test_b = ApkRegionDataset(args.root_dir, args.mem_regions, args.mode, apk_list=benign_test, class_filter=0)
        test_m = ApkRegionDataset(args.root_dir, args.mem_regions, args.mode, apk_list=malware_test, class_filter=1)
        test_ds = ConcatDataset([test_b, test_m])
        logger.info(f"[DEBUG] test_benign={len(test_b)} test_malware={len(test_m)} total={len(test_ds)}")

    if len(test_ds) == 0:
        logger.error("Empty test dataset.")
        return

    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # model
    state_dict, cfg = load_ckpt(args.ckpt)
    model = build_model_from_cfg(cfg, args.mode)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    apk_probs = collections.defaultdict(list)
    apk_labels = {}
    apk_region_details = collections.defaultdict(list)

    with torch.no_grad():
        for x, y, apks, regions in tqdm(loader, desc="Evaluating", unit="batch"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            probs = logits_to_pos_prob(logits).detach().cpu().numpy()

            for i in range(len(probs)):
                apk = apks[i]
                lbl = int(y[i].item())
                p = float(probs[i])
                reg = regions[i]
                apk_probs[apk].append(p)
                apk_labels[apk] = lbl
                apk_region_details[apk].append((reg, p))

    all_apks = list(apk_probs.keys())
    y_true = np.array([apk_labels[a] for a in all_apks], dtype=int)

    if args.apk_agg == "max":
        y_score = np.array([np.max(apk_probs[a]) for a in all_apks], dtype=float)
    else:
        y_score = np.array([np.mean(apk_probs[a]) for a in all_apks], dtype=float)

    # threshold (APK-level)
    if len(set(y_true.tolist())) < 2:
        thr = 0.5
        auc = pr_auc = 0.0
    else:
        if args.threshold == "f1":
            best_thr, best_f1 = 0.5, -1.0
            for t in np.linspace(0.1, 0.9, 81):
                preds = (y_score > t).astype(int)
                f1 = f1_score(y_true, preds)
                if f1 > best_f1:
                    best_f1, best_thr = f1, float(t)
            thr = best_thr
        else:
            fpr, tpr, thr_list = roc_curve(y_true, y_score)
            thr = float(thr_list[int(np.argmax(tpr - fpr))])

        auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))

    y_pred = (y_score > thr).astype(int)
    acc = float((y_pred == y_true).mean() * 100.0)

    logger.info(f"thr({args.threshold})={thr:.4f} | acc={acc:.2f}%")
    if len(set(y_true.tolist())) >= 2:
        logger.info(f"ROC AUC={auc:.4f} | PR AUC={pr_auc:.4f}")

    logger.info("\nReport:\n" + classification_report(y_true, y_pred, target_names=["Benign", "Malicious"], zero_division=0))
    logger.info("\nCM:\n" + str(confusion_matrix(y_true, y_pred)))

    # misclassified + region errors at thr
    fp = fn = 0
    for apk, score, pred in zip(all_apks, y_score, y_pred):
        lbl = int(apk_labels[apk])
        if int(pred) != lbl:
            kind = "FP" if int(pred) == 1 else "FN"
            fp += int(kind == "FP")
            fn += int(kind == "FN")
            logger.info(f"[APK_MISCLASS] {kind} APK={apk} True={lbl} Pred={int(pred)} Score={float(score):.4f}")
            for reg, p in sorted(apk_region_details[apk], key=lambda x: x[1], reverse=True):
                if int(p > thr) != lbl:
                    logger.info(f"    [REGION_MISCLASS@THR] Region={reg} Prob={p:.4f} Thr={thr:.4f}")

    logger.info(f"[SUMMARY] FP={fp} FN={fn}")

    # plots
    if len(set(y_true.tolist())) >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title("ROC (APK-level)")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f"PR AUC={pr_auc:.2f}")
        plt.title("PR (APK-level)")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()

        plt.tight_layout()
        out_plot = os.path.join(args.save_dir, f"test_{args.mode}_{args.mem_regions}_{os.path.basename(args.ckpt).replace('.pth','')}.png")
        plt.savefig(out_plot)
        plt.show()
        logger.info(f"Saved plot: {out_plot}")


if __name__ == "__main__":
    main()