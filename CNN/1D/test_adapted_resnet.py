import os
import re
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    average_precision_score, precision_recall_curve
)
import matplotlib.pyplot as plt
import warnings
from loguru import logger
from collections import defaultdict

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Model: same as training (2D ResNet18 -> pool -> 1D head)
# in_ch is inferred from checkpoint if not specified
# hidden_dim is inferred from checkpoint if not specified
# =========================
class ResNet1DTransfer(nn.Module):
    def __init__(self, num_classes=2, in_ch=3, hidden_dim=128):
        super().__init__()

        # No need for pretrained downloads at test time; weights come from checkpoint
        try:
            base_model = models.resnet18(weights=None)
        except TypeError:
            base_model = models.resnet18(pretrained=False)

        base_model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])  # -> [B,512,H',W']
        self.pool = nn.AdaptiveAvgPool2d((1, None))  # collapse H', keep W' as time axis

        self.classifier = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(32),
            nn.Flatten(),
            nn.Linear(256 * 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x).squeeze(2)  # [B,512,T]
        return self.classifier(x)


def infer_in_ch_and_hidden_dim(state_dict: dict):
    # infer in_ch from first conv weight if present
    in_ch = None
    for k in ["feature_extractor.0.weight", "conv1.weight", "0.conv1.weight"]:
        if k in state_dict:
            in_ch = int(state_dict[k].shape[1])  # [out,in,kH,kW]
            break
    if in_ch is None:
        in_ch = 3

    # infer hidden_dim from classifier first Linear layer weight: classifier.4.weight
    hidden_dim = None
    for k in ["classifier.4.weight", "classifier.7.weight"]:
        if k in state_dict:
            hidden_dim = int(state_dict[k].shape[0])  # out_features
            break
    if hidden_dim is None:
        hidden_dim = 128

    return in_ch, hidden_dim


def load_model_from_ckpt(ckpt_path: str):
    state = torch.load(ckpt_path, map_location=device)

    # unwrap if needed
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
        cfg = state.get("cfg", {})
        in_ch = int(cfg.get("in_ch", 3))
        hidden_dim = int(cfg.get("hidden_dim", 128))
    else:
        sd = state
        in_ch, hidden_dim = infer_in_ch_and_hidden_dim(sd)

    model = ResNet1DTransfer(num_classes=2, in_ch=in_ch, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path} | inferred in_ch={in_ch} hidden_dim={hidden_dim}")
    return model, in_ch


def to_pos_prob(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.size(1) == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    if logits.ndim == 2 and logits.size(1) == 1:
        return torch.sigmoid(logits[:, 0])
    if logits.ndim == 1:
        return torch.sigmoid(logits)
    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")


# =========================
# Dataset: unified RGB/Gray preprocessing
# =========================
class EvalImageDataset(Dataset):
    """
    Mirrors your training dataset I/O:
      - reads maps.txt
      - selects regions by selection_mode (stack, data_stack, all)
      - loads from images/horizontal/RGB with filename: 0x{addr}_dump_RGB_horizontal.png
      - converts to RGB or grayscale based on color_mode
      - resize to (1024,224), ToTensor
      - returns (tensor, label, apk_name, region_desc)
    """
    def __init__(self, root_dir, selection_mode="stack", apk_list=None, class_filter=None, color_mode="RGB"):
        self.samples = []
        self.color_mode = color_mode
        self.transform = transforms.ToTensor()

        self.selection_map = {
            "stack": [r"\[stack\]"],
            "data_stack": [r"^/data/.*", r"\[stack\]"],
            "all": []
        }

        class_dirs = {0: "benign_dumps", 1: "dumps_dataset"}
        if class_filter is not None:
            class_dirs = {class_filter: class_dirs[class_filter]}

        for label, class_dir in class_dirs.items():
            class_path = os.path.join(root_dir, class_dir)
            if not os.path.exists(class_path):
                continue

            apk_folders = apk_list if apk_list else sorted(os.listdir(class_path))
            for apk in tqdm(apk_folders, desc=f"Scanning {class_dir}", unit="apk"):
                apk_path = apk if os.path.isabs(apk) else os.path.join(class_path, apk)

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

                            matched = (selection_mode == "all") or any(
                                re.search(p, desc) for p in self.selection_map.get(selection_mode, [])
                            )
                            if matched:
                                img_path = os.path.join(img_dir, f"0x{addr}_dump_RGB_horizontal.png")
                                if os.path.exists(img_path):
                                    self.samples.append((img_path, label, os.path.basename(apk_path), desc))
                except Exception as e:
                    logger.warning(f"[{apk_path}] skipped: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, apk, desc = self.samples[idx]
        try:
            if self.color_mode == "Grayscale":
                img = Image.open(img_path).convert("L").resize((1024, 224))
            else:
                img = Image.open(img_path).convert("RGB").resize((1024, 224))
            x = self.transform(img)
            return x, torch.tensor(label, dtype=torch.long), apk, desc
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))


def resolve_checkpoint(
    path_ds: str,
    mem_regions: str,
    model_name: str,
    color_mode: str,
    split: str,
    pretrained_val: str = "True"
):
    """
    Tries a list of common checkpoint naming schemes (including your older grayscale prefix
    and the 'renet18' typo for last).
    """
    if os.path.isabs(split) and split.endswith(".pth"):
        return split if os.path.exists(split) else None

    candidates = []

    # Unified naming (recommended from your unified train):
    if split == "best_validation":
        candidates += [
            f"resnet18-{mem_regions}_{model_name}_{color_mode}_best_validation_{pretrained_val}.pth",
        ]
    elif split == "last":
        candidates += [
            f"renet18-{mem_regions}_{model_name}_{color_mode}_last_{pretrained_val}.pth",   # keep typo support
            f"resnet18-{mem_regions}_{model_name}_{color_mode}_last_{pretrained_val}.pth",
        ]
    else:
        raise ValueError("split must be 'best_validation', 'last', or an absolute .pth path")

    # Backward-compat naming from your existing scripts:
    # RGB
    if split == "best_validation":
        candidates += [
            f"resnet18-{mem_regions}_{model_name}_RGB_best_validation_{pretrained_val}.pth",
        ]
    else:
        candidates += [
            f"renet18-{mem_regions}_{model_name}_RGB_last_{pretrained_val}.pth",
        ]

    # Grayscale (older prefix)
    if split == "best_validation":
        candidates += [
            f"resnet18-grayscale-{mem_regions}_{model_name}_Grayscale_best_validation_{pretrained_val}.pth",
            f"resnet18-grayscale-{mem_regions}_{model_name}_{color_mode}_best_validation_{pretrained_val}.pth",
        ]
    else:
        candidates += [
            f"renet18-grayscale-{mem_regions}_{model_name}_Grayscale_last_{pretrained_val}.pth",
            f"renet18-grayscale-{mem_regions}_{model_name}_{color_mode}_last_{pretrained_val}.pth",
        ]

    for name in candidates:
        p = os.path.join(path_ds, name)
        if os.path.exists(p):
            return p
    return None


def test_memory_regions_unified(
    mem_regions: str,
    model_name: str,
    color_mode: str,                 # "RGB" or "Grayscale"
    path_ds: str,                    # folder containing checkpoints; logs/plots written here
    split: str = "best_validation",  # "best_validation" | "last" | absolute .pth
    pretrained_val: str = "True",
    agg: str = "mean",               # "mean" | "max"
    single_apk_path: str = None,
    log_region_errors_at_thr: bool = True,
    batch_size: int = 16,
):
    root_dir = "/mnt/malware_ram/Android"
    os.makedirs(path_ds, exist_ok=True)

    log_path = os.path.join(path_ds, f"test_{model_name}_{color_mode}_{mem_regions}_{split}.log")
    logger.remove()
    logger.add(log_path, format="{message}")
    logger.info(f"[SETTINGS] mem_regions={mem_regions} model_name={model_name} color={color_mode} split={split} agg={agg}")

    # --- dataset build ---
    if single_apk_path:
        class_filter = 1 if "dumps_dataset" in single_apk_path else 0
        test_dataset = EvalImageDataset(
            root_dir=root_dir,
            selection_mode=mem_regions,
            apk_list=[single_apk_path],
            class_filter=class_filter,
            color_mode=color_mode
        )
        logger.info(f"Loaded {len(test_dataset)} samples from single APK: {single_apk_path}")
    else:
        benign_dir = os.path.join(root_dir, "benign_dumps")
        malware_dir = os.path.join(root_dir, "dumps_dataset")

        benign_apks = sorted(os.listdir(benign_dir)) if os.path.exists(benign_dir) else []
        malware_apks = sorted(os.listdir(malware_dir)) if os.path.exists(malware_dir) else []

        test_start = 1500
        benign_test = [os.path.join(benign_dir, a) for i, a in enumerate(benign_apks) if i >= test_start]
        malware_test = [os.path.join(malware_dir, a) for i, a in enumerate(malware_apks) if i >= test_start]

        ds_b = EvalImageDataset(root_dir, mem_regions, apk_list=benign_test, class_filter=0, color_mode=color_mode)
        ds_m = EvalImageDataset(root_dir, mem_regions, apk_list=malware_test, class_filter=1, color_mode=color_mode)
        test_dataset = ConcatDataset([ds_b, ds_m])
        logger.info(f"[DEBUG] test_benign={len(ds_b)} | test_malicious={len(ds_m)} | total={len(test_dataset)}")

    if len(test_dataset) == 0:
        logger.error("Dataset is empty. Check paths and selection_mode.")
        return

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- checkpoint + model ---
    ckpt_path = resolve_checkpoint(path_ds, mem_regions, model_name, color_mode, split, pretrained_val)
    if not ckpt_path:
        logger.error("Checkpoint not found. Searched common patterns in path_ds.")
        logger.error(f"path_ds={path_ds}")
        return

    model, inferred_in_ch = load_model_from_ckpt(ckpt_path)

    # sanity: warn if color_mode mismatches inferred channels
    expected_in_ch = 1 if color_mode == "Grayscale" else 3
    if inferred_in_ch != expected_in_ch:
        logger.warning(f"[WARN] color_mode={color_mode} but checkpoint conv1 expects in_ch={inferred_in_ch}. "
                       f"(This is OK only if you trained that way.)")

    # --- eval loop: region -> APK aggregation ---
    apk_probs = defaultdict(list)
    apk_labels = {}
    apk_region_details = defaultdict(list)

    with torch.no_grad():
        for x, y, apks, regions in tqdm(test_loader, desc="Evaluating", unit="batch"):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = to_pos_prob(logits).detach().cpu().numpy()

            for i in range(len(probs)):
                apk = apks[i]
                p = float(probs[i])
                lbl = int(y[i].item())
                desc = regions[i]
                apk_probs[apk].append(p)
                apk_labels[apk] = lbl
                apk_region_details[apk].append((desc, p))

    all_apks = list(apk_probs.keys())
    if not all_apks:
        logger.error("No samples collected.")
        return

    y_true = np.array([apk_labels[a] for a in all_apks], dtype=int)
    if agg == "max":
        y_score = np.array([np.max(apk_probs[a]) for a in all_apks], dtype=float)
    else:
        y_score = np.array([np.mean(apk_probs[a]) for a in all_apks], dtype=float)

    uniq = set(y_true.tolist())

    # --- threshold via Youden J on APK-level scores ---
    if len(uniq) < 2:
        thr = 0.5
        y_pred = (y_score > thr).astype(int)
        auc = 0.0
        pr_auc = 0.0
        logger.warning(f"Only one class present: {uniq}. Using thr=0.5; skipping ROC/PR.")
    else:
        fpr, tpr, thr_list = roc_curve(y_true, y_score)
        thr = float(thr_list[int(np.argmax(tpr - fpr))])
        y_pred = (y_score > thr).astype(int)
        auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))

    acc = float((y_pred == y_true).mean() * 100.0)

    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Aggregation: {agg}")
    logger.info(f"Best Threshold (Youden): {thr:.4f}")
    logger.info(f"Test Accuracy (APK-level): {acc:.2f}%")
    if len(uniq) >= 2:
        logger.info(f"ROC AUC: {auc:.4f} | PR AUC: {pr_auc:.4f}")

    logger.info("\nClassification Report:\n" + str(
        classification_report(y_true, y_pred, labels=[0, 1], target_names=["Benign", "Malicious"], zero_division=0)
    ))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    logger.info("\nConfusion Matrix:\n" + str(cm))

    # --- Misclassified APKs ---
    fp = fn = 0
    for apk, score in zip(all_apks, y_score):
        lbl = int(apk_labels[apk])
        pred = int(score > thr)
        if pred != lbl:
            kind = "FP" if pred == 1 else "FN"
            fp += (kind == "FP")
            fn += (kind == "FN")
            logger.info(f"[APK_MISCLASS] APK={apk} | Type={kind} | True={lbl} | Pred={pred} "
                        f"| {agg.capitalize()}Prob={score:.4f} | Thr={thr:.4f} | Regions={len(apk_region_details[apk])}")
    logger.info(f"[SUMMARY] APK-level misclassifications -> FP={fp}, FN={fn}")

    # --- Region-level errors at thr (optional) ---
    if log_region_errors_at_thr:
        for apk in all_apks:
            lbl = int(apk_labels[apk])
            for desc, p in apk_region_details[apk]:
                pred = int(p > thr)
                if pred != lbl:
                    logger.info(f"[REGION_MISCLASS@THR] APK={apk} | Region={desc} | True={lbl} "
                                f"| Pred={pred} | Prob={p:.4f} | Thr={thr:.4f}")

    # --- Plots (ROC + PR) ---
    if len(uniq) >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"ROC Curve (APK-level) - {color_mode}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
        plt.title(f"Precision-Recall (APK-level) - {color_mode}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()

        plt.tight_layout()
        out_plot = os.path.join(path_ds, f"test_{model_name}_{mem_regions}_{color_mode}_{split}_{agg}.png")
        plt.savefig(out_plot)
        plt.show()
        logger.info(f"Saved evaluation plot to: {out_plot}")


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # RGB
    test_memory_regions_unified(
        mem_regions="data_stack",
        model_name="resnet1d",
        color_mode="RGB",
        path_ds="path/dataset",
        split="best_validation",
        pretrained_val="True",
        agg="mean"
    )

    # Grayscale
    # test_memory_regions_unified(
    #     mem_regions="data_stack",
    #     model_name="resnet1d",
    #     color_mode="Grayscale",
    #     path_ds=".",
    #     split="best_validation",
    #     pretrained_val="True",
    #     agg="mean"
    # )