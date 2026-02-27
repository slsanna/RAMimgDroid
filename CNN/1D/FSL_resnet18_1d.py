import os
import re
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from loguru import logger
from collections import defaultdict, Counter
import warnings

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

# === ResNet1DTransfer ===
class ResNet1DTransfer(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet1DTransfer, self).__init__()
        base_model = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.classifier = nn.Identity()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x).squeeze(2)  # → [batch, 512, time]
        return x

# === Custom Dataset ===
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, selection_mode="stack", apk_list=None, class_filter=None):
        self.samples = []
        selection_map = {"stack": [r"\\[stack\\]"], "data_stack": [r"^/data/.*", r"\\[stack\\]"]}
        class_dirs = {0: 'benign_dumps', 1: 'dumps_dataset'}
        if class_filter is not None:
            class_dirs = {class_filter: class_dirs[class_filter]}

        for label, cdir in class_dirs.items():
            base_path = os.path.join(root_dir, cdir)
            apks = apk_list if apk_list else os.listdir(base_path)
            for apk in tqdm(apks, desc=f"Scanning {cdir}"):
                apk_path = os.path.join(base_path, apk)
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
                            match = selection_mode == "all" or any(re.search(p, desc) for p in selection_map.get(selection_mode, []))
                            if match:
                                img_path = os.path.join(img_dir, f"0x{addr}_dump_RGB_horizontal.png")
                                if os.path.exists(img_path):
                                    self.samples.append((img_path, label, apk, desc))
                except Exception as e:
                    logger.warning(f"[{apk}] skipped: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, apk, desc = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB").resize((1024, 224))
            image = transforms.ToTensor()(image)
            return image, label, f"{apk}|{os.path.basename(img_path)}"
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))

# === Main function ===
def cnn_then_classifiers(mem_regions="data_stack"):
    root_dir = "/mnt/malware_ram/Android"
    logger.remove()
    logger.add(f"cnn_FSL_resnet1d_APK_classifiers_apk_{mem_regions}.log", format="{message}")

    benign_paths = sorted(os.listdir(os.path.join(root_dir, "benign_dumps")))
    malware_paths = sorted(os.listdir(os.path.join(root_dir, "dumps_dataset")))
    benign_train = benign_paths[:1300]
    benign_val = benign_paths[1300:1500]
    malware_train = malware_paths[:1300]
    malware_val = malware_paths[1300:1500]

    train_benign = CustomImageDataset(root_dir, selection_mode=mem_regions, apk_list=benign_train, class_filter=0)
    train_malicious = CustomImageDataset(root_dir, selection_mode=mem_regions, apk_list=malware_train, class_filter=1)
    val_benign = CustomImageDataset(root_dir, selection_mode=mem_regions, apk_list=benign_val, class_filter=0)
    val_malicious = CustomImageDataset(root_dir, selection_mode=mem_regions, apk_list=malware_val, class_filter=1)

    train_dataset = ConcatDataset([train_benign, train_malicious])
    val_dataset = ConcatDataset([val_benign, val_malicious])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = ResNet1DTransfer().to(device)
    state_dict = torch.load("/home/ssanna/Desktop/malware_ram/Android/imgs/1d_cnn/resnet/resnet18-data_stack_resnet1d_RGB_best_validation_True.pth", map_location=device)
    allowed_keys = ["feature_extractor.", "pool."]
    filtered_dict = {k: v for k, v in state_dict.items() if any(k.startswith(prefix) for prefix in allowed_keys)}
    model.load_state_dict(filtered_dict, strict=False)
    model.eval()

    def extract_features(loader):
        features, labels, apk_tags = [], [], []
        with torch.no_grad():
            for x, y, tags in loader:
                x = x.to(device)
                feats = model(x)
                feats = feats.view(feats.size(0), -1).cpu().numpy()
                features.append(feats)
                labels.extend(y.numpy())
                apk_tags.extend(tags)
        return np.vstack(features), np.array(labels), apk_tags

    train_feats, train_lbls, _ = extract_features(train_loader)
    val_feats, val_lbls, val_apk_tags = extract_features(val_loader)

    apk_feature_map = defaultdict(lambda: {"features": [], "label": None, "paths": []})
    for feat, label, tag in zip(val_feats, val_lbls, val_apk_tags):
        apk, img_path = tag.split('|', 1)
        apk_feature_map[apk]["features"].append(feat)
        apk_feature_map[apk]["paths"].append(img_path)
        apk_feature_map[apk]["label"] = label

    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'SVM': SVC(probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'MLP': MLPClassifier(hidden_layer_sizes=(128,), max_iter=300)
    }

    for name, clf in classifiers.items():
        logger.info(f"\n=== Training {name} ===")
        clf.fit(train_feats, train_lbls)

        y_true, y_pred = [], []
        for apk, info in apk_feature_map.items():
            feats = np.vstack(info["features"])
            label = info["label"]
            probs = clf.predict_proba(feats)
            pred_avg = np.mean(probs, axis=0)
            pred = np.argmax(pred_avg)
            y_true.append(label)
            y_pred.append(pred)

            if pred != label:
                logger.info(f"✗ APK Misclassified: {apk} | True: {label} | Pred: {pred}")
                for path in info["paths"]:
                    logger.info(f"  - {path}")

        logger.info(f"\n{name} Classification Report:")
        logger.info(classification_report(y_true, y_pred, target_names=["Benign", "Malicious"]))
        logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_true, y_pred)))

if __name__ == "__main__":
    cnn_then_classifiers("data_stack")
