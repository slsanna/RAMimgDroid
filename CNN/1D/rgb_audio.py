import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms
from torchaudio.models import wav2vec2_base
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from collections import defaultdict
import re

# === Device ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Dataset: RGB image to fake audio waveform ===
class RGBtoFakeAudioDataset(Dataset):
    def __init__(self, root_dir, selection_mode="stack", apk_list=None, class_filter=None, min_length=16000):
        self.samples = []
        self.min_length = min_length
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

            apk_folders = apk_list if apk_list else os.listdir(class_path)
            for apk in apk_folders:
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
                except Exception:
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, apk, region = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((1024, 3))
        image = np.array(image).astype(np.float32) / 255.0
        mono = image.mean(axis=0)
        mono = 2.0 * mono - 1.0

        if mono.shape[0] < self.min_length:
            pad = self.min_length - mono.shape[0]
            mono = np.pad(mono, (0, pad), mode='constant')
        else:
            mono = mono[:self.min_length]

        audio_tensor = torch.tensor(mono, dtype=torch.float32)
        return audio_tensor, label, apk, region

# === Model ===
class FakeAudioClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(FakeAudioClassifier, self).__init__()
        self.wav2vec = wav2vec2_base()
        self.wav2vec.eval()
        for p in self.wav2vec.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features, _ = self.wav2vec.extract_features(x)
        pooled = features[-1].mean(dim=1)
        return self.classifier(pooled)

# === Training and Evaluation ===
def train_and_evaluate(root_dir, selection_mode="data_stack"):
    dataset = RGBtoFakeAudioDataset(root_dir, selection_mode=selection_mode)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = FakeAudioClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

    model.train()
    for epoch in range(10):
        losses = []
        for x, y, _, _ in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1} | Loss: {np.mean(losses):.4f}")

    # Save model and training parameters after training
    save_path_train = os.path.join("path", "trained_fake_audio_model.pth")
    torch.save(model.state_dict(), save_path_train)
    print(f"Model and parameters saved to {save_path_train}")

    # Evaluation
    model.eval()
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    apk_probs = defaultdict(list)
    apk_labels = {}
    apk_regions = defaultdict(list)
    with torch.no_grad():
        for x, y, apks, regions in val_loader:
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            for i, apk in enumerate(apks):
                apk_probs[apk].append(probs[i])
                apk_labels[apk] = y[i].item()
                apk_regions[apk].append((regions[i], probs[i]))

    apk_preds = {apk: int(np.mean(probs) > 0.5) for apk, probs in apk_probs.items()}
    apk_true = [apk_labels[apk] for apk in apk_preds]
    apk_pred = [apk_preds[apk] for apk in apk_preds]

    # Save model again after evaluation (optional second checkpoint)
    save_path_eval = os.path.join("path", "evaluated_fake_audio_model.pth")
    torch.save(model.state_dict(), save_path_eval)
    print(f"Evaluation model saved to {save_path_eval}")

    print("\n=== Classification Report ===")
    print(classification_report(apk_true, apk_pred, target_names=["Benign", "Malicious"]))
    print("Confusion Matrix:")
    print(confusion_matrix(apk_true, apk_pred))

    print("\n=== Misclassified APKs and Their Regions ===")
    for apk in apk_preds:
        if apk_preds[apk] != apk_labels[apk]:
            print(f"APK: {apk} | True: {apk_labels[apk]} | Pred: {apk_preds[apk]}")
            sorted_regions = sorted(apk_regions[apk], key=lambda x: x[1], reverse=True)
            for region, prob in sorted_regions:
                print(f"    Region: {region} | Prob: {prob:.4f}")

if __name__ == "__main__":
    train_and_evaluate("dataset/path")
