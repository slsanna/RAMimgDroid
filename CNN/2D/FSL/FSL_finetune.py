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
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings("ignore")
device = "cpu"

# === ResizeWithPadding ===
class ResizeWithPadding:
    def __init__(self, target_size=224):
        self.target_size = target_size

    def __call__(self, image):
        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        new_image = Image.new("RGB", (self.target_size, self.target_size))
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        new_image.paste(image, (paste_x, paste_y))
        return new_image

# === Transform ===
transform_no_augmentation = transforms.Compose([
    ResizeWithPadding(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === CustomDataset ===
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, apk_list=None,
                 class_filter=None, selection_mode="stack", custom_keywords=None):
        self.images = []
        self.transform = transform
        self.selection_mode = selection_mode
        self.custom_keywords = custom_keywords

        self.selection_map = {
            "stack": [r"\\[stack\\]"],
            "data_stack": [r"^/data/.*", r"\\[stack\\]"]
        }

        class_dirs = {0: 'benign_dumps', 1: 'dumps_dataset'}
        if class_filter is not None:
            class_dirs = {class_filter: class_dirs[class_filter]}

        for label, class_dir in class_dirs.items():
            apk_dirs = apk_list if apk_list is not None else [
                os.path.join(root_dir, class_dir, d) for d in os.listdir(os.path.join(root_dir, class_dir))
            ]
            for apk_dir in tqdm(apk_dirs, desc=f"Processing {class_dir}", unit="apk"):
                apk = os.path.basename(apk_dir)
                maps_path = os.path.join(apk_dir, 'maps.txt')
                images_dir = os.path.join(apk_dir, 'images', 'complete', 'RGB')
                if not os.path.exists(maps_path) or not os.path.exists(images_dir):
                    continue
                try:
                    selected_addresses = set()
                    with open(maps_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            address = parts[0].split('-')[0].strip()
                            desc = parts[-1] if len(parts) > 5 else ""
                            if self.selection_mode in [None, "all"]:
                                selected_addresses.add(address)
                            elif self.custom_keywords:
                                if any(re.search(pattern, desc) for pattern in self.custom_keywords):
                                    selected_addresses.add(address)
                            elif self.selection_mode in self.selection_map:
                                for pattern in self.selection_map[self.selection_mode]:
                                    if re.search(pattern, desc):
                                        selected_addresses.add(address)
                    for addr in selected_addresses:
                        image_filename = f"0x{addr}_dump_RGB.png"
                        image_path = os.path.join(images_dir, image_filename)
                        if os.path.exists(image_path):
                            self.images.append((image_path, label, apk))
                except Exception as e:
                    logger.info(f"Error processing {apk}: {e}")
                    continue

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label, apk = self.images[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            apk_tag = f"{apk}|{os.path.basename(img_path)}"
            return image, label, apk_tag
        except Exception:
            return self.__getitem__((idx + 1) % len(self.images))

# === Main function ===
def cnn_then_classifiers(mem_regions="data_stack"):
    root_dir = "/mnt/malware_ram/Android"

    logger.remove()
    logger.add(f"cnn_APK_classifiers_apk_{mem_regions}.log", format="{message}")

    benign_apks = sorted(os.listdir(os.path.join(root_dir, 'benign_dumps')))
    benign_apk_paths_train = [os.path.join(root_dir, 'benign_dumps', apk) for apk in benign_apks[:1300]]
    benign_apk_paths_val = [os.path.join(root_dir, 'benign_dumps', apk) for apk in benign_apks[1300:1500]]

    malicious_apks = sorted(os.listdir(os.path.join(root_dir, 'dumps_dataset')))
    malicious_apk_paths_train = [os.path.join(root_dir, 'dumps_dataset', apk) for apk in malicious_apks[:1300]]
    malicious_apk_paths_val = [os.path.join(root_dir, 'dumps_dataset', apk) for apk in malicious_apks[1300:1500]]

    train_benign = CustomDataset(root_dir, transform_no_augmentation, apk_list=benign_apk_paths_train, class_filter=0, selection_mode=mem_regions)
    train_malicious = CustomDataset(root_dir, transform_no_augmentation, apk_list=malicious_apk_paths_train, class_filter=1, selection_mode=mem_regions)

    val_benign = CustomDataset(root_dir, transform_no_augmentation, apk_list=benign_apk_paths_val, class_filter=0, selection_mode=mem_regions)
    val_malicious = CustomDataset(root_dir, transform_no_augmentation, apk_list=malicious_apk_paths_val, class_filter=1, selection_mode=mem_regions)

    logger.info(f"Benign validation samples: {len(val_benign)}")
    logger.info(f"Malicious validation samples: {len(val_malicious)}")

    train_dataset = ConcatDataset([train_benign, train_malicious])
    val_dataset = ConcatDataset([val_benign, val_malicious])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.BatchNorm1d(in_features), nn.Dropout(0.4), nn.Linear(in_features, 2))
    model.load_state_dict(torch.load("/home/ssanna/Desktop/malware_ram/Android/imgs/sections/memory_regions/data_stack/pre-trained/data_stack_resnet18_RGB_best_validation_True.pth", map_location=device))
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    def extract_features(loader):
        features, labels, apk_tags = [], [], []
        with torch.no_grad():
            for batch in loader:
                imgs, lbls, paths = batch
                imgs = imgs.to(device)
                feats = model(imgs)
                features.append(feats.cpu().numpy())
                labels.extend(lbls.numpy())
                apk_tags.extend(paths)
        return np.vstack(features), np.array(labels), apk_tags

    train_feats, train_lbls, _ = extract_features(train_loader)
    val_feats, val_lbls, val_apk_tags = extract_features(val_loader)

    apk_feature_map = defaultdict(lambda: {"features": [], "label": None, "paths": []})
    for feat, label, tag in zip(val_feats, val_lbls, val_apk_tags):
        apk_name, img_path = tag.split('|', 1)
        apk_feature_map[apk_name]["features"].append(feat)
        apk_feature_map[apk_name]["paths"].append(img_path)
        apk_feature_map[apk_name]["label"] = label

    from sklearn.model_selection import GridSearchCV

    param_grids = {
        "KNN": {"n_neighbors": [3, 5], "metric": ["euclidean"]},
        "SVM": {"C": [1], "gamma": [0.01], "kernel": ["rbf"]},
        "Random Forest": {"n_estimators": [100], "max_depth": [20], "class_weight": ["balanced"]},
        "Logistic Regression": {"C": [0.1], "solver": ["saga"], "max_iter": [3000]},
        "MLP": {"hidden_layer_sizes": [(256,), (512, 256)], "activation": ["relu"], "max_iter": [500], "early_stopping": [True]}
    }

    for name, param_grid in param_grids.items():
        logger.info(f"\n=== Grid Search for {name} ===")
        if name == "KNN":
            clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        elif name == "SVM":
            clf = GridSearchCV(SVC(probability=True), param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        elif name == "Random Forest":
            clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        elif name == "Logistic Regression":
            clf = GridSearchCV(LogisticRegression(), param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        elif name == "MLP":
            clf = GridSearchCV(MLPClassifier(), param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        else:
            continue

        clf.fit(train_feats, train_lbls)

        y_true, y_pred = [], []
        logger.info(f"\n--- Best {name} ---")
        logger.info(f"Best Parameters: {clf.best_params_}")

        for apk, info in apk_feature_map.items():
            feats = np.vstack(info["features"])
            label = info["label"]
            pred_probs = clf.predict_proba(feats)
            pred_avg = np.mean(pred_probs, axis=0)
            pred_label = np.argmax(pred_avg)

            y_true.append(label)
            y_pred.append(pred_label)

            if pred_label != label:
                logger.info(f"\n✗ APK Misclassified: {apk}")
                logger.info(f"  True Label: {label} | Predicted: {pred_label}")
                logger.info(f"  Misclassified memory regions:")
                for path in info["paths"]:
                    logger.info(f"    - {path}")

        logger.info(f"y_true distribution: {Counter(y_true)}")
        logger.info(f"y_pred distribution: {Counter(y_pred)}")
        logger.info("\n" + classification_report(y_true, y_pred, labels=[0, 1], target_names=["Benign", "Malicious"], zero_division=0))
        logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_true, y_pred, labels=[0, 1])))

cnn_then_classifiers("data_stack")
