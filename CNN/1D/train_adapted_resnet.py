import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, f1_score
import matplotlib.pyplot as plt
import warnings
from loguru import logger
import collections

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Model: ResNet18 (2D) -> pool -> 1D head
# Supports RGB (3ch) and Gray (1ch)
# =========================
class ResNet1DTransfer(nn.Module):
    def __init__(self, num_classes=2, in_ch=3, use_pretrained=True):
        super().__init__()

        # Load resnet18
        # Newer torchvision uses weights=..., older uses pretrained=...
        try:
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None)
        except Exception:
            base_model = models.resnet18(pretrained=use_pretrained)

        # If grayscale, replace conv1 to accept 1-channel input
        if in_ch != 3:
            old_conv = base_model.conv1
            base_model.conv1 = nn.Conv2d(
                in_ch, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )

            # If we loaded pretrained weights for RGB, initialize new conv sensibly
            # by averaging RGB filters into 1 channel (common trick).
            if use_pretrained and old_conv.weight.shape[1] == 3 and in_ch == 1:
                with torch.no_grad():
                    base_model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, None))  # collapse height, keep width as "time"

        self.classifier = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(32),
            nn.Flatten(),
            nn.Linear(256 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x).squeeze(2)  # [B, 512, T]
        return self.classifier(x)


# =========================
# Dataset: unified RGB/Gray preprocessing
# =========================
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, selection_mode="stack", apk_list=None, class_filter=None, color_mode="RGB"):
        """
        color_mode: "RGB" or "Grayscale"
        """
        self.samples = []
        self.color_mode = color_mode

        self.selection_map = {
            "stack": [r"\\[stack\\]"],
            "data_stack": [r"^/data/.*", r"\\[stack\\]"],
            "all": []
        }

        class_dirs = {0: 'benign_dumps', 1: 'dumps_dataset'}
        if class_filter is not None:
            class_dirs = {class_filter: class_dirs[class_filter]}

        for label, class_dir in class_dirs.items():
            class_path = os.path.join(root_dir, class_dir)
            if not os.path.exists(class_path):
                continue

            apk_folders = apk_list if apk_list else os.listdir(class_path)
            for apk in tqdm(apk_folders, desc=f"Scanning {class_dir}", unit="apk"):
                apk_path = os.path.join(class_path, apk)
                maps_path = os.path.join(apk_path, "maps.txt")
                img_dir = os.path.join(apk_path, "images", "horizontal", "RGB")  # keep your folder structure

                if not os.path.exists(maps_path) or not os.path.exists(img_dir):
                    continue

                try:
                    with open(maps_path, 'r', errors="ignore") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 6:
                                continue
                            addr = parts[0].split('-')[0]
                            desc = line.split(None, 5)[-1] if len(line.split(None, 5)) >= 6 else ""

                            if selection_mode == "all":
                                match = True
                            else:
                                match = any(re.search(p, desc) for p in self.selection_map.get(selection_mode, []))

                            if match:
                                img_path = os.path.join(img_dir, f"0x{addr}_dump_RGB_horizontal.png")
                                if os.path.exists(img_path):
                                    self.samples.append((img_path, label, apk, desc))
                except Exception as e:
                    logger.warning(f"[{apk}] skipped: {e}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, apk, desc = self.samples[idx]
        try:
            if self.color_mode == "Grayscale":
                img = Image.open(img_path).convert("L").resize((1024, 224))
            else:
                img = Image.open(img_path).convert("RGB").resize((1024, 224))

            x = self.to_tensor(img)  # [C,H,W] where C=1 or 3
            return x, label, apk, desc
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))


# =========================
# Training function (unified)
# =========================
def train_memory_regions(mem_regions, model_name, pretrained_val, color_mode, epochs=10):
    """
    color_mode: "RGB" or "Grayscale"
    pretrained_val: string token used in filenames (e.g., "True")
    """
    root_dir = "dataset/path"

    log_path = f"resnet18-train_{model_name}_{color_mode}_{mem_regions}.log"
    logger.remove()
    logger.add(log_path, format="{message}")

    # APK splits (same as your apk-based script)
    benign_apks = sorted(os.listdir(os.path.join(root_dir, "benign_dumps")))
    malicious_apks = sorted(os.listdir(os.path.join(root_dir, "malware_dumps")))

    train_benign_apks = benign_apks[:1300]
    train_malicious_apks = malicious_apks[:1300]
    val_benign_apks = benign_apks[1300:1500]
    val_malicious_apks = malicious_apks[1300:1500]

    # Datasets (only difference is color_mode)
    train_benign = CustomImageDataset(root_dir, mem_regions, apk_list=train_benign_apks, class_filter=0, color_mode=color_mode)
    train_malicious = CustomImageDataset(root_dir, mem_regions, apk_list=train_malicious_apks, class_filter=1, color_mode=color_mode)
    val_benign = CustomImageDataset(root_dir, mem_regions, apk_list=val_benign_apks, class_filter=0, color_mode=color_mode)
    val_malicious = CustomImageDataset(root_dir, mem_regions, apk_list=val_malicious_apks, class_filter=1, color_mode=color_mode)

    logger.info(f"Train benign: {len(train_benign)} | Train malicious: {len(train_malicious)}")
    logger.info(f"Val benign: {len(val_benign)} | Val malicious: {len(val_malicious)}")

    train_dataset = ConcatDataset([train_benign, train_malicious])
    val_dataset = ConcatDataset([val_benign, val_malicious])

    # Weighted sampler to balance classes
    train_labels = [label for ds in [train_benign, train_malicious] for _, label, _, _ in ds.samples]
    class_counts = np.bincount(train_labels)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model
    in_ch = 1 if color_mode == "Grayscale" else 3
    use_pretrained = (str(pretrained_val).lower() == "true")
    model = ResNet1DTransfer(num_classes=2, in_ch=in_ch, use_pretrained=use_pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs
    )

    best_val_accuracy = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'roc_auc': [], 'pr_auc': []}

    for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        # ---- Train ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y, _, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * x.size(0)
            correct += (torch.argmax(out, dim=1) == y).sum().item()
            total += y.size(0)

        train_accuracy = correct / total * 100
        train_loss = running_loss / total

        # ---- Val (APK-level aggregation) ----
        model.eval()
        val_loss = 0.0
        apk_probs = collections.defaultdict(list)
        apk_labels = {}
        apk_regions = collections.defaultdict(list)
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for x, y, apks, regions in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                probs = torch.softmax(out, dim=1)[:, 1]
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)

                for i in range(len(y)):
                    apk = apks[i]
                    label = y[i].item()
                    prob = probs[i].item()
                    region = regions[i]
                    apk_probs[apk].append(prob)
                    apk_labels[apk] = label
                    apk_regions[apk].append((region, prob))
                    all_labels.append(label)
                    all_probs.append(prob)

        val_loss /= len(val_dataset)

        # threshold chosen by best F1 on region-level probs (same as your script)
        best_thresh, best_f1 = 0.5, 0
        for t in np.linspace(0.1, 0.9, 81):
            preds = [int(p > t) for p in all_probs]
            f1 = f1_score(all_labels, preds)
            if f1 > best_f1:
                best_thresh, best_f1 = t, f1

        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        pr_auc = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0

        # APK prediction by mean probability
        apk_preds = {apk: int(np.mean(probs) > best_thresh) for apk, probs in apk_probs.items()}
        apk_true = [apk_labels[apk] for apk in apk_preds]
        apk_pred = [apk_preds[apk] for apk in apk_preds]
        val_accuracy = np.mean(np.array(apk_true) == np.array(apk_pred)) * 100

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['roc_auc'].append(auc)
        history['pr_auc'].append(pr_auc)

        # Save best
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                model.state_dict(),
                f"resnet18-{mem_regions}_{model_name}_{color_mode}_best_validation_{pretrained_val}.pth"
            )

        logger.info(f"\nEpoch {epoch+1}: Train Acc={train_accuracy:.2f}%, Val Acc={val_accuracy:.2f}%, AUC={auc:.4f}, PR AUC={pr_auc:.4f}")
        logger.info("Classification Report:\n" + classification_report(apk_true, apk_pred, target_names=["Benign", "Malicious"]))
        logger.info("Confusion Matrix:\n" + str(confusion_matrix(apk_true, apk_pred)))

        logger.info("\nMisclassified APKs:")
        for apk in apk_preds:
            if apk_preds[apk] != apk_labels[apk]:
                logger.info(f"APK: {apk} | True: {apk_labels[apk]} | Pred: {apk_preds[apk]}")
                for region, prob in sorted(apk_regions[apk], key=lambda x: x[1], reverse=True):
                    logger.info(f"    Region: {region} | Prob: {prob:.4f}")

    # Save last (keep your original naming if you want; here I keep your "renet18" typo to match test scripts)
    torch.save(
        model.state_dict(),
        f"renet18-{mem_regions}_{model_name}_{color_mode}_last_{pretrained_val}.pth"
    )

    # ---- Plots ----
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.plot(history['roc_auc'], label='ROC AUC')
    plt.plot(history['pr_auc'], label='PR AUC')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{mem_regions}_{model_name}_{color_mode}_training_metrics_{pretrained_val}.png")
    plt.show()


# =========================
# Run examples
# =========================
if __name__ == "__main__":
    # RGB
    train_memory_regions(mem_regions="data_stack", model_name="resnet1d", pretrained_val="True", color_mode="RGB", epochs=10)

    # Grayscale
    # train_memory_regions(mem_regions="data_stack", model_name="resnet1d", pretrained_val="True", color_mode="Grayscale", epochs=10)