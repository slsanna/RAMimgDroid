import os
import re
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from sklearn.ensemble import StackingClassifier, VotingClassifier
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from loguru import logger
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings("ignore")
device = "cpu"

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

transform_no_augmentation = transforms.Compose([
    ResizeWithPadding(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        class_dirs = {0: 'benign_dumps', 1: 'malware_dumps'}
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


# === Main function with Boosting and Ensemble Classifiers ===
def cnn_then_boost_classifiersOLD(mem_regions="data_stack"):

    root_dir = "dataset_path"

    logger.remove()
    logger.add(f"cnn_APK_ensemble_classifiers_{mem_regions}.log", format="{message}")

    benign_apks = sorted(os.listdir(os.path.join(root_dir, 'benign_dumps')))
    benign_apk_paths_train = [os.path.join(root_dir, 'benign_dumps', apk) for apk in benign_apks[:1300]]
    benign_apk_paths_val = [os.path.join(root_dir, 'benign_dumps', apk) for apk in benign_apks[1300:1500]]

    malicious_apks = sorted(os.listdir(os.path.join(root_dir, 'malware_dumps')))
    malicious_apk_paths_train = [os.path.join(root_dir, 'malware_dumps', apk) for apk in malicious_apks[:1300]]
    malicious_apk_paths_val = [os.path.join(root_dir, 'malware_dumps', apk) for apk in malicious_apks[1300:1500]]

    train_benign = CustomDataset(root_dir, transform_no_augmentation, apk_list=benign_apk_paths_train, class_filter=0, selection_mode=mem_regions)
    train_malicious = CustomDataset(root_dir, transform_no_augmentation, apk_list=malicious_apk_paths_train, class_filter=1, selection_mode=mem_regions)
    val_benign = CustomDataset(root_dir, transform_no_augmentation, apk_list=benign_apk_paths_val, class_filter=0, selection_mode=mem_regions)
    val_malicious = CustomDataset(root_dir, transform_no_augmentation, apk_list=malicious_apk_paths_val, class_filter=1, selection_mode=mem_regions)

    train_dataset = ConcatDataset([train_benign, train_malicious])
    val_dataset = ConcatDataset([val_benign, val_malicious])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.BatchNorm1d(in_features), nn.Dropout(0.4), nn.Linear(in_features, 2))
    model.load_state_dict(torch.load("path/to/pretrained", map_location=device))
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    def extract_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for batch in loader:
                imgs, lbls = batch[:2]
                imgs = imgs.to(device)
                feats = model(imgs)
                features.append(feats.cpu().numpy())
                labels.extend(lbls.numpy())
        return np.vstack(features), np.array(labels)

    train_feats, train_lbls = extract_features(train_loader)
    val_feats, val_lbls = extract_features(val_loader)

    base_models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True),
        "LogReg": LogisticRegression(max_iter=3000),
        "RF": RandomForestClassifier(n_estimators=200),
        "MLP": MLPClassifier(max_iter=500),
        "XGB": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0)
    }

    # Voting Classifier (soft)
    voting = VotingClassifier(
        estimators=[(k, v) for k, v in base_models.items()],
        voting='soft'
    )

    # Stacking Classifier
    stacking = StackingClassifier(
        estimators=[(k, v) for k, v in base_models.items()],
        final_estimator=LogisticRegression(max_iter=3000)
    )

    all_models = base_models.copy()
    all_models["Voting"] = voting
    all_models["Stacking"] = stacking

    for name, clf in all_models.items():
        clf.fit(train_feats, train_lbls)
        preds = clf.predict(val_feats)
        logger.info(f"\n--- {name} ---")
        logger.info("\n" + classification_report(val_lbls, preds, target_names=["Benign", "Malicious"]))
        logger.info("\nConfusion Matrix:\n" + str(confusion_matrix(val_lbls, preds)))

from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict, Counter

def cnn_then_ensemble_apk_level(mem_regions="data_stack"):
    root_dir = "dataset/path"

    logger.remove()
    logger.add(f"cnn_APK_ensemble_{mem_regions}.log", format="{message}")

    benign_apks = sorted(os.listdir(os.path.join(root_dir, 'benign_dumps')))
    benign_apk_paths_train = [os.path.join(root_dir, 'benign_dumps', apk) for apk in benign_apks[:1300]]
    benign_apk_paths_val = [os.path.join(root_dir, 'benign_dumps', apk) for apk in benign_apks[1300:1500]]

    malicious_apks = sorted(os.listdir(os.path.join(root_dir, 'malware_dumps')))
    malicious_apk_paths_train = [os.path.join(root_dir, 'malware_dumps', apk) for apk in malicious_apks[:1300]]
    malicious_apk_paths_val = [os.path.join(root_dir, 'malware_dumps', apk) for apk in malicious_apks[1300:1500]]

    train_benign = CustomDataset(root_dir, transform_no_augmentation, apk_list=benign_apk_paths_train, class_filter=0, selection_mode=mem_regions)
    train_malicious = CustomDataset(root_dir, transform_no_augmentation, apk_list=malicious_apk_paths_train, class_filter=1, selection_mode=mem_regions)
    val_benign = CustomDataset(root_dir, transform_no_augmentation, apk_list=benign_apk_paths_val, class_filter=0, selection_mode=mem_regions)
    val_malicious = CustomDataset(root_dir, transform_no_augmentation, apk_list=malicious_apk_paths_val, class_filter=1, selection_mode=mem_regions)

    train_dataset = ConcatDataset([train_benign, train_malicious])
    val_dataset = ConcatDataset([val_benign, val_malicious])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.BatchNorm1d(in_features), nn.Dropout(0.4), nn.Linear(in_features, 2))
    model.load_state_dict(torch.load("path/to/pretrained", map_location=device))
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    def extract_features_with_tags(loader):
        features, labels, apk_tags = [], [], []
        with torch.no_grad():
            for imgs, lbls, tags in loader:
                imgs = imgs.to(device)
                feats = model(imgs)
                features.append(feats.cpu().numpy())
                labels.extend(lbls.numpy())
                apk_tags.extend(tags)
        return np.vstack(features), np.array(labels), apk_tags

    train_feats, train_lbls, _ = extract_features_with_tags(train_loader)
    val_feats, val_lbls, val_apk_tags = extract_features_with_tags(val_loader)

    # === Define individual models with specified hyperparameters ===
    knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    svm = SVC(C=1, gamma=0.01, kernel='rbf', probability=True)
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight="balanced")
    lr = LogisticRegression(C=0.1, max_iter=3000, solver='saga')
    mlp = MLPClassifier(hidden_layer_sizes=(256,), activation='relu', max_iter=500, early_stopping=True)

    ensemble = VotingClassifier(
        estimators=[
            ('knn', knn),
            ('svm', svm),
            ('rf', rf),
            ('lr', lr),
            ('mlp', mlp)
        ],
        voting='soft'  # use 'soft' for probability averaging
    )

    # === Train ===
    ensemble.fit(train_feats, train_lbls)
    preds = ensemble.predict(val_feats)

    # === IMAGE-level metrics ===
    logger.info(f"\n--- ENSEMBLE (Image-Level) ---")
    logger.info("\n" + classification_report(val_lbls, preds, target_names=["Benign", "Malicious"]))
    logger.info("\nConfusion Matrix:\n" + str(confusion_matrix(val_lbls, preds)))

    # === APK-level aggregation ===
    apk_votes = defaultdict(list)
    apk_labels = {}

    for tag, pred, true in zip(val_apk_tags, preds, val_lbls):
        apk = tag.split("|")[0]
        apk_votes[apk].append(pred)
        apk_labels[apk] = true

    apk_preds = {}
    for apk, votes in apk_votes.items():
        majority_vote = Counter(votes).most_common(1)[0][0]
        apk_preds[apk] = majority_vote

    apk_true_labels = [apk_labels[apk] for apk in apk_preds]
    apk_pred_labels = [apk_preds[apk] for apk in apk_preds]

    logger.info(f"\n--- ENSEMBLE (APK-Level) ---")
    logger.info("\n" + classification_report(apk_true_labels, apk_pred_labels, target_names=["Benign", "Malicious"]))
    logger.info("\nConfusion Matrix:\n" + str(confusion_matrix(apk_true_labels, apk_pred_labels)))
    logger.info(f"\nTotal APKs evaluated: {len(apk_preds)}")

    # === Print misclassified APKs ===
    misclassified_apks = []
    for apk in apk_preds:
        true_label = apk_labels[apk]
        pred_label = apk_preds[apk]
        if true_label != pred_label:
            misclassified_apks.append(apk)
            logger.info(f"\n✗ APK Misclassified: {apk}")
            logger.info(f"  True Label: {true_label} | Predicted: {pred_label}")
            logger.info(f"  Misclassified memory regions:")
            for tag in val_apk_tags:
                tag_apk, image_path = tag.split("|", 1)
                if tag_apk == apk:
                    logger.info(f"    - {image_path}")


# Run it
cnn_then_ensemble_apk_level("data_stack")
