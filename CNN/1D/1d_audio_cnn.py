import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, average_precision_score, f1_score
import torchaudio
from torchaudio.models import wav2vec2_base
import matplotlib.pyplot as plt
import warnings
from loguru import logger
import collections

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Pretrained Audio Model ===
class Pretrained1DAudioCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(Pretrained1DAudioCNN, self).__init__()
        self.feature_extractor = wav2vec2_base()
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.mean(dim=1)
        with torch.no_grad():
            feats, _ = self.feature_extractor.extract_features(x)
        x = feats[-1].mean(dim=1)
        return self.classifier(x)

# === Custom Dataset ===
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, selection_mode="stack", apk_list=None, class_filter=None):
        self.samples = []
        self.selection_map = {
            "stack": [r"\\[stack\\]"],
            "data_stack": [r"^/data/.*", r"\\[stack\\]"]
        }

        class_dirs = {0: 'benign_dumps', 1: 'dumps_dataset'}
        if class_filter is not None:
            class_dirs = {class_filter: class_dirs[class_filter]}

        for label, class_dir in class_dirs.items():
            class_path = os.path.join(root_dir, class_dir)
            if not os.path.exists(class_path):
                continue

            apk_folders = apk_list if apk_list is not None else os.listdir(class_path)
            for apk in tqdm(apk_folders, desc=f"Scanning {class_dir}", unit="apk"):
                apk_path = os.path.join(class_path, apk)
                maps_path = os.path.join(apk_path, "maps.txt")
                img_dir = os.path.join(apk_path, "images", "horizontal", "RGB")

                if not os.path.exists(maps_path) or not os.path.exists(img_dir):
                    continue

                try:
                    with open(maps_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 6:
                                continue
                            addr = parts[0].split('-')[0]
                            desc = line.split(None, 5)[-1] if len(line.split(None, 5)) == 6 else ""

                            match = selection_mode == "all" or any(re.search(p, desc) for p in self.selection_map.get(selection_mode, []))
                            if match:
                                img_path = os.path.join(img_dir, f"0x{addr}_dump_RGB_horizontal.png")
                                if os.path.exists(img_path):
                                    self.samples.append((img_path, label, apk, desc))
                except Exception as e:
                    logger.warning(f"[{apk}] skipped: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, apk_name, region_desc = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = image.resize((1024, 3))
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.tensor(image).permute(2, 1, 0).mean(dim=2)
            return image, label, apk_name, region_desc
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))

# === Training Function ===
def train_memory_regions(mem_regions, model_name, pretrained_val, color):
    root_dir = "/mnt/malware_ram/Android"
    log_path = f"train_{model_name}_{color}_{mem_regions}_APK_1Daudio.log"
    logger.remove()
    logger.add(log_path, format="{message}")

    # Load actual APK folder names
    benign_apks = sorted(os.listdir(os.path.join(root_dir, "benign_dumps")))
    malicious_apks = sorted(os.listdir(os.path.join(root_dir, "dumps_dataset")))
    train_benign_apks = benign_apks[:50]
    train_malicious_apks = malicious_apks[:50]
    val_benign_apks = benign_apks[50:70]
    val_malicious_apks = malicious_apks[50:70]

    # Datasets
    train_benign = CustomImageDataset(root_dir, selection_mode=mem_regions, apk_list=train_benign_apks, class_filter=0)
    train_malicious = CustomImageDataset(root_dir, selection_mode=mem_regions, apk_list=train_malicious_apks, class_filter=1)
    val_benign = CustomImageDataset(root_dir, selection_mode=mem_regions, apk_list=val_benign_apks, class_filter=0)
    val_malicious = CustomImageDataset(root_dir, selection_mode=mem_regions, apk_list=val_malicious_apks, class_filter=1)

    logger.info(f"Train benign: {len(train_benign)} | Train malicious: {len(train_malicious)}")
    logger.info(f"Val benign: {len(val_benign)} | Val malicious: {len(val_malicious)}")

    train_dataset = ConcatDataset([train_benign, train_malicious])
    val_dataset = ConcatDataset([val_benign, val_malicious])

    train_labels = [label for ds in [train_benign, train_malicious] for _, label, _, _ in ds.samples]
    class_counts = np.bincount(train_labels)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Pretrained1DAudioCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=10)

    best_val_accuracy = 0.0
    best_train_accuracy = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'roc_auc': [], 'pr_auc': []}

    for epoch in tqdm(range(100), desc="Training Progress", unit="epoch"):
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
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            torch.save(model.state_dict(), f"{mem_regions}_{model_name}_{color}_best_train_{pretrained_val}.pth")

        model.eval()
        val_loss = 0.0
        apk_probs = collections.defaultdict(list)
        apk_labels = {}
        apk_regions = collections.defaultdict(list)
        all_labels, all_probs = [], []

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

        best_thresh, best_f1 = 0.5, 0
        for t in np.linspace(0.1, 0.9, 81):
            preds = [int(p > t) for p in all_probs]
            f1 = f1_score(all_labels, preds)
            if f1 > best_f1:
                best_thresh, best_f1 = t, f1

        auc = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)
        apk_preds = {apk: int(np.max(probs) > best_thresh) for apk, probs in apk_probs.items()}
        apk_true = [apk_labels[apk] for apk in apk_preds]
        apk_pred = [apk_preds[apk] for apk in apk_preds]
        val_accuracy = np.mean(np.array(apk_true) == np.array(apk_pred)) * 100

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['roc_auc'].append(auc)
        history['pr_auc'].append(pr_auc)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f"{mem_regions}_{model_name}_{color}_best_validation_{pretrained_val}.pth")

        logger.info(f"\nEpoch {epoch+1}: Train Acc={train_accuracy:.2f}%, Val Acc={val_accuracy:.2f}%, AUC={auc:.4f}, PR AUC={pr_auc:.4f}")
        logger.info("Classification Report:")
        logger.info(classification_report(apk_true, apk_pred, target_names=["Benign", "Malicious"]))
        logger.info("Confusion Matrix:")
        logger.info(confusion_matrix(apk_true, apk_pred))

        logger.info("\nMisclassified APKs:")
        for apk in apk_preds:
            if apk_preds[apk] != apk_labels[apk]:
                logger.info(f"APK: {apk} | True: {apk_labels[apk]} | Pred: {apk_preds[apk]}")
                for region, prob in sorted(apk_regions[apk], key=lambda x: x[1], reverse=True):
                    logger.info(f"    Region: {region} | Prob: {prob:.4f}")

    torch.save(model.state_dict(), f"{mem_regions}_{model_name}_{color}_last_{pretrained_val}.pth")

    # === Plot ===
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
    plt.title('Validation Metrics over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{mem_regions}_{model_name}_{color}_training_metrics_{pretrained_val}.png")
    plt.show()

# === Run ===
train_memory_regions('data_stack', 'audio', 'False', 'RGB')
