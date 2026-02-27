import os
import fnmatch
import sys
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, vit_b_16, ViT_B_16_Weights, inception_v3, Inception_V3_Weights

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import warnings
from loguru import logger


warnings.filterwarnings("ignore")

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# === MODEL BUILDER ===
def build_model(model_name="mobilenet_v2", num_classes=2, pretrained=True):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(pretrained=pretrained)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    
    elif model_name == "inception_v3":
        weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        model = inception_v3(weights=weights, aux_logits=False)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "vit_b_16":
        model = models.vit_b_16(pretrained=pretrained)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            model.classifier[0],
            nn.Hardswish(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=pretrained)
        in_features = model.classifier[0].in_features 
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "efficientnet_b0":
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
            state_dict = torch.hub.load_state_dict_from_url(
                weights.url, progress=True, check_hash=False  # Disable hash check
            )
            model = efficientnet_b0(weights=None)
            model.load_state_dict(state_dict)
        else:
            model = efficientnet_b0(weights=None)

        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )

    elif model_name in ["vgg11", "vgg13", "vgg16", "vgg19",
                    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]:
        vgg_models = {
            "vgg11": models.vgg11,
            "vgg13": models.vgg13,
            "vgg16": models.vgg16,
            "vgg19": models.vgg19,
            "vgg11_bn": models.vgg11_bn,
            "vgg13_bn": models.vgg13_bn,
            "vgg16_bn": models.vgg16_bn,
            "vgg19_bn": models.vgg19_bn,
        }
        model = vgg_models[model_name](pretrained=pretrained)
        in_features = model.classifier[6].in_features  # final fc layer
        model.classifier[6] = nn.Linear(in_features, num_classes)


    elif model_name == "didroid":
        didroid_path = os.path.abspath(os.path.join(os.getcwd(), '../../models'))
        sys.path.append(didroid_path)
        from didroid import DiDroidNet
        model = DiDroidNet(num_classes=num_classes)

    elif model_name == "dexray":
        dexray_path = os.path.abspath(os.path.join(os.getcwd(), '../../models'))
        sys.path.append(dexray_path)
        from dexray import DexRayNet 
        model = DexRayNet(input_size=128*128)

    elif model_name == "crgbmem":
        crgbmem_path = os.path.abspath(os.path.join(os.getcwd(), '../../models'))
        sys.path.append(crgbmem_path)
        from crgb import CRGBMemCNN 
        model = CRGBMemCNN()

    
    else:
        raise ValueError(f"Model {model_name} not supported")

    return model

# === ResizeWithPadding ===
class ResizeWithPadding:
    def __init__(self, target_size=224):
        self.target_size = target_size

    def __call__(self, image):
        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale) #forse per dexray serve 180x180
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

# === Custom Dataset ===
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, selected_indices=None, apk_list=None, class_filter=None, selection_mode="stack", custom_keywords=None, color_mode="RGB"):
        self.images = []
        self.transform = transform
        self.selection_mode = selection_mode
        self.custom_keywords = custom_keywords
        self.dexray_mode = False
        self.color_mode = color_mode
        self.use_grayscale = (color_mode != "RGB")

        self.selection_map = {
            "stack": [r"\[stack\]"],
            "memory_regions": [r"\[stack\]", r"\[vvar\]"],
            "stack_vdso" : [r"\[stack\]", r"\[vdso\]"],
            "base_odex-apk": [r"base\.odex", r"base\.apk"],
            "base_vdex-apk": [r"base\.vdex", r"base\.apk"],
            "base_odex-vdex": [r"base\.odex", r"base_vdex"],
            "base_odex": [r"base\.odex"],
            "base_vdex": [r"base\.vdex"],
            "base_apk": [r"base\.apk"],
            "complete_memory_regions": [r"\[heap\]", r"\[anon:libc_malloc\]", r"\[stack\]",
                r"\[vdso\]", r"\[vvar\]", r"\[vsyscall\]",
                r"\(deleted\)", r"u:object_r.*", r"data@resource-cache@" ],
            "base_apk_memory": [r"\[stack\]", r"\[vvar\]", r"base\.apk", r"base\.vdex", r"base\.odex"],
            "base_apk_stack": [r"\[stack\]", r"base\.apk", r"base\.vdex", r"base\.odex"],
            "base_apk": [
                r"base\.apk", r"base\.vdex", r"base\.odex"
            ],
            "all_data_paths": [r"^/data/.*"],
            "data_stack": [r"^/data/.*", r"\[stack\]"],
            "apk_full": [
                r"base\.apk", r"base\.vdex", r"base\.odex", r"/data/.*/lib/.+\.so$",
                r"\[anon:dalvik.*\]", r"\[anon:.*\.apk.*\]", r"\.db$", r"\.db-shm$", r"\.db-wal$",
                r"Cookies", r"Preferences", r"ua_preferences", r"data@resource-cache@", r"\.dex$",
                r"/data/.+\.odex$" 
            ],
            "apk_full_memory": [ r"\[stack\]", r"\[vvar\]",
                r"base\.apk", r"base\.vdex", r"base\.odex", r"/data/.*/lib/.+\.so$",
                r"\[anon:dalvik.*\]", r"\[anon:.*\.apk.*\]", r"\.db$", r"\.db-shm$", r"\.db-wal$",
                r"Cookies", r"Preferences", r"ua_preferences", r"data@resource-cache@", r"\.dex$",
                r"/data/.+\.odex$" 
            ],
            "apk_full_stack": [ r"\[stack\]", r"base\.apk", r"base\.vdex", r"base\.odex", r"/data/.*/lib/.+\.so$",
                r"\[anon:dalvik.*\]", r"\[anon:.*\.apk.*\]", r"\.db$", r"\.db-shm$", r"\.db-wal$",
                r"Cookies", r"Preferences", r"ua_preferences", r"data@resource-cache@", r"\.dex$",
                r"/data/.+\.odex$" ],
            "system": [
                r"framework\.jar", r"boot\.vdex", r"boot-framework\.vdex", r"base\.odex"
            ],
            "system_full": [
                r"dalvik-classes", r"framework\.jar", r"boot-framework\.art",
                r"boot-core-libart\.art", r"boot\.oat", r"boot-framework\.oat",
                r"boot\.vdex", r"boot-framework\.vdex", r"base\.odex"
            ],
            "libraries": [r"\.so", r"\[anon:lib.*"]
        }

        class_dirs = {0: 'benign_dumps', 1: 'dumps_dataset'}
        if class_filter is not None:
            class_dirs = {class_filter: class_dirs[class_filter]}

        temp_images = []

        for label, class_dir in class_dirs.items():
            class_dir_path = os.path.join(root_dir, class_dir)
            if not os.path.exists(class_dir_path):
                continue

            apk_dirs = apk_list if apk_list is not None else [os.path.join(class_dir_path, apk) for apk in os.listdir(class_dir_path)]
            for apk_dir in tqdm(apk_dirs, desc=f"Processing {class_dir}", unit="apk"):
                apk = os.path.basename(apk_dir)
                maps_path = os.path.join(apk_dir, 'maps.txt')
                images_dir = os.path.join(apk_dir, 'images', 'complete', 'RGB')

                if not os.path.exists(maps_path) or not os.path.exists(images_dir):
                    continue

                selected_regions = {}
                with open(maps_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        parts = line.split()
                        if not parts:
                            continue
                        address = parts[0].split('-')[0].strip()
                        desc = line.split(None, 5)[-1] if len(line.split(None, 5)) == 6 else ""

                        if self.selection_mode in [None, "all"]:
                            selected_regions[address] = desc
                        elif self.custom_keywords:
                            if any(re.search(pattern, desc) for pattern in self.custom_keywords):
                                selected_regions[address] = desc
                        elif self.selection_mode in self.selection_map:
                            for pattern in self.selection_map[self.selection_mode]:
                                if re.search(pattern, desc):
                                    selected_regions[address] = desc

                for addr, desc in selected_regions.items():
                    image_filename = f"0x{addr}_dump_RGB.png"
                    image_path = os.path.join(images_dir, image_filename)
                    if os.path.exists(image_path):
                        temp_images.append((image_path, label, apk, desc))

        self.images = [temp_images[i] for i in selected_indices if i < len(temp_images)] if selected_indices else temp_images

    def __len__(self):
        return len(self.images)

    def __getitemOLD__(self, idx):
        img_path, label, apk_name, region_desc = self.images[idx]
        try:
            if self.dexray_mode:
                image = Image.open(img_path).convert("L" if self.use_grayscale else "RGB")
                image = image.resize((128, 128))
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.tensor(image).view(1, -1)
                return image, torch.tensor(label, dtype=torch.float32), apk_name
            else:
                image = Image.open(img_path).convert("L" if self.use_grayscale else "RGB")
                if self.transform:
                    image = self.transform(image)
                return image, label, apk_name, region_desc
        except Exception:
            return self.__getitem__((idx + 1) % len(self.images))

    def __getitem__(self, idx):
        img_path, label, apk_name, region_desc = self.images[idx]
        try:
            if self.dexray_mode:
                image = Image.open(img_path).convert("L" if self.use_grayscale else "RGB")
                image = image.resize((128, 128))
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.tensor(image).view(1, -1)
                return image, torch.tensor(label, dtype=torch.float32), apk_name, region_desc  # Always return 4 items
            else:
                image = Image.open(img_path).convert("L" if self.use_grayscale else "RGB")
                if self.transform:
                    image = self.transform(image)
                return image, label, apk_name, region_desc  # Already returns 4
        except Exception:
            return self.__getitem__((idx + 1) % len(self.images))


"""def train_memory_regions(mem_regions, model_name, pretrained_val):
    # === Dataset Loading ===
    root_dir = "/mnt/malware_ram/Android"
    benign_train_indices = list(range(1300))
    malicious_train_indices = list(range(1300))
    benign_val_indices = list(range(1300, 1500))
    malicious_val_indices = list(range(1300, 1500))
    log_path = f"train_{model_name}_{mem_regions}_{pretrained_val}.log"
    logger.remove()
    logger.add(log_path, format="{message}")
    train_benign = CustomDataset(root_dir, transform_no_augmentation, benign_train_indices, class_filter=0, selection_mode=mem_regions)
    train_malicious = CustomDataset(root_dir, transform_no_augmentation, malicious_train_indices, class_filter=1, selection_mode=mem_regions)
    val_benign = CustomDataset(root_dir, transform_no_augmentation, benign_val_indices, class_filter=0, selection_mode=mem_regions)
    val_malicious = CustomDataset(root_dir, transform_no_augmentation, malicious_val_indices, class_filter=1, selection_mode=mem_regions)

    print(f"[DEBUG] train_benign: {len(train_benign)}")
    print(f"[DEBUG] train_malicious: {len(train_malicious)}")
    print(f"[DEBUG] val_benign: {len(val_benign)}")
    print(f"[DEBUG] val_malicious: {len(val_malicious)}")


    train_dataset = ConcatDataset([train_benign, train_malicious])
    val_dataset = ConcatDataset([val_benign, val_malicious])

    train_labels = [label for ds in [train_benign, train_malicious] for _, label in ds.images]
    class_counts = np.bincount(train_labels)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    #train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, sampler=sampler) #without weights
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # === Model Selection ===
    model = build_model(model_name=model_name, num_classes=2, pretrained=pretrained_val).to(device)

    # === Training Setup ===
    num_epochs = 100
    class_weights = torch.tensor([0.5, 0.5], device=device)
    if model_name == "dexray":
        criterion = nn.BCELoss()
    else:
        class_weights = torch.tensor([0.5, 0.5], device=device)
    criterion = nn.CrossEntropyLoss(class_weights)

    criterion = nn.CrossEntropyLoss(class_weights)
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=num_epochs)

    best_val_accuracy = 0.0
    best_train_accuracy = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'roc_auc': [], 'pr_auc': []}

    # === Training Loop ===
    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total * 100
        train_loss = running_loss / total

        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            torch.save(model.state_dict(), f"{mem_regions}_{model_name}_best_train_{pretrained_val}.pth")
            logger.info(f"\nModel saved at epoch {epoch+1} with TRAIN accuracy: {best_train_accuracy:.2f}%")

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_labels, all_probs = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        all_preds = (np.array(all_probs) > best_threshold).astype(int)
        val_accuracy = np.mean(all_preds == np.array(all_labels)) * 100
        auc = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['roc_auc'].append(auc)
        history['pr_auc'].append(pr_auc)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f"{mem_regions}_{model_name}_best_validation_{pretrained_val}.pth")
            logger.info(f"\nModel saved at epoch {epoch+1} with VALIDATION accuracy: {best_val_accuracy:.2f}%")

        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        logger.info(f"ROC AUC: {auc:.4f}, PR AUC: {pr_auc:.4f}")
        #logger.info("\nClassification Report:")
        #logger.info(classification_report(all_labels, all_preds, target_names=['Benign', 'Malicious']))
        logger.info("\n Classification Report: \n" +str(classification_report(all_labels, all_preds, target_names=['Benign', 'Malicious'])))
        #logger.info("Confusion Matrix:")
        #logger.info(confusion_matrix(all_labels, all_preds))
        logger.info("\n Confusion Matrix: \n" + str(confusion_matrix(all_labels, all_preds)))

    torch.save(model.state_dict(), f"{mem_regions}_{model_name}_last_{pretrained_val}.pth")

    # === Plot Training History ===
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
    plt.savefig(f"{mem_regions}_{model_name}_training_metrics_{pretrained_val}")
    #plt.savefig("training_metrics.png")
    plt.show()

train_memory_regions('data_stack', 'didroid', 'True')
"""

# === Training Function ===
def train_memory_regionsOLD(mem_regions, model_name, pretrained_val, color):
    root_dir = "/mnt/malware_ram/Android"
    benign_train_indices = list(range(1300))
    malicious_train_indices = list(range(1300))
    benign_val_indices = list(range(1300, 1500))
    malicious_val_indices = list(range(1300, 1500))
    num_epochs = 100

    save_dir = "/home/ssanna/Desktop/malware_ram/Android/imgs/sections/memory_regions/data_stack/pre-trained"
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, f"train_{model_name}_{color}_{mem_regions}_{pretrained_val}_misclassified-regions.log")
    logger.remove()
    logger.add(log_path, format="{message}")

    is_dexray = model_name == "dexray"
    train_benign = CustomDataset(root_dir, None if is_dexray else transform_no_augmentation, benign_train_indices, class_filter=0, selection_mode=mem_regions, color_mode=color)
    train_malicious = CustomDataset(root_dir, None if is_dexray else transform_no_augmentation, malicious_train_indices, class_filter=1, selection_mode=mem_regions, color_mode=color)
    val_benign = CustomDataset(root_dir, None if is_dexray else transform_no_augmentation, benign_val_indices, class_filter=0, selection_mode=mem_regions, color_mode=color)
    val_malicious = CustomDataset(root_dir, None if is_dexray else transform_no_augmentation, malicious_val_indices, class_filter=1, selection_mode=mem_regions, color_mode=color)

    if is_dexray:
        train_benign.dexray_mode = True
        train_malicious.dexray_mode = True
        val_benign.dexray_mode = True
        val_malicious.dexray_mode = True

    train_dataset = ConcatDataset([train_benign, train_malicious])
    val_dataset = ConcatDataset([val_benign, val_malicious])

    #train_labels = [label for ds in [train_benign, train_malicious] for _, label, _ in ds.images]
    train_labels = [label for ds in [train_benign, train_malicious] for _, label, _, _ in ds.images]

    class_counts = np.bincount(train_labels)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = build_model(model_name=model_name, num_classes=2, pretrained=pretrained_val).to(device)

    criterion = nn.BCELoss() if is_dexray else nn.CrossEntropyLoss(torch.tensor([0.5, 0.5], device=device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=100)

    best_val_accuracy = 0.0
    best_train_accuracy = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'roc_auc': [], 'pr_auc': []}
    
    
    
    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels, _, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if is_dexray:
                loss = criterion(outputs, labels)
                preds = (outputs > 0.5).long()
            else:
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_accuracy = correct / total * 100
        train_loss = running_loss / total

        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            torch.save(model.state_dict(), os.path.join(save_dir, f"{mem_regions}_{model_name}_{color}_best_train_{pretrained_val}.pth"))
            logger.info(f"\nModel saved at epoch {epoch+1} with TRAIN accuracy: {best_train_accuracy:.2f}%")
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_labels, all_probs = [], []
        with torch.no_grad():

            for batch in val_loader:
                if is_dexray:
                    images, labels, apk_names, region_descs = batch
                else:
                    images, labels, apk_names, region_descs = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if is_dexray:
                    loss = criterion(outputs, labels)
                    probs = outputs
                    preds = (outputs > 0.5).long().squeeze()
                else:
                    loss = criterion(outputs, labels)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    preds = (probs > 0.5).long()

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                val_loss += loss.item() * images.size(0)

                # === Misclassification Logging ===
                misclassified = set()
                for i in range(len(labels)):
                    actual = labels[i].item()
                    predicted = preds[i].item()
                    if actual != predicted:
                        key = (apk_names[i], region_descs[i], actual, predicted)
                        misclassified.add(key)

                for apk_name, region, actual, predicted in misclassified:
                        logger.info(f"Misclassified: APK={apk_name} | Region={region} | True={actual} | Predicted={predicted}")

            """for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if is_dexray:
                    loss = criterion(outputs, labels)
                    probs = outputs
                else:
                    loss = criterion(outputs, labels)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
            """
       
        val_loss /= len(val_loader.dataset)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        all_preds = (np.array(all_probs) > best_threshold).astype(int)
        val_accuracy = np.mean(all_preds == np.array(all_labels)) * 100
        auc = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['roc_auc'].append(auc)
        history['pr_auc'].append(pr_auc)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(save_dir, f"{mem_regions}_{model_name}_{color}_best_validation_{pretrained_val}.pth"))
            logger.info(f"\nModel saved at epoch {epoch+1} with VALIDATION accuracy: {best_val_accuracy:.2f}%")

        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        logger.info(f"ROC AUC: {auc:.4f}, PR AUC: {pr_auc:.4f}")
        #logger.info("\nClassification Report:")
        #logger.info(classification_report(all_labels, all_preds, target_names=['Benign', 'Malicious']))
        logger.info("\n Classification Report: \n" +str(classification_report(all_labels, all_preds, target_names=['Benign', 'Malicious'])))
        #logger.info("Confusion Matrix:")
        #logger.info(confusion_matrix(all_labels, all_preds))
        logger.info("\n Confusion Matrix: \n" + str(confusion_matrix(all_labels, all_preds)))

    torch.save(model.state_dict(), os.path.join(save_dir, f"{mem_regions}_{model_name}_{color}_last_{pretrained_val}.pth"))

    # === Plot Training History ===
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
    plt.savefig(f"{mem_regions}_{model_name}_{color}_training_metrics_{pretrained_val}")
    #plt.savefig("training_metrics.png")
    plt.show()

def train_memory_regions(mem_regions, model_name, pretrained_val, color, save_dir):
    import collections
    from glob import glob
    from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, classification_report, confusion_matrix
    from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

    root_dir = "/mnt/malware_ram/Android"
    benign_apk_paths = sorted(glob(f"{root_dir}/benign_dumps/*"))
    malicious_apk_paths = sorted(glob(f"{root_dir}/dumps_dataset/*"))

    benign_train_apks = benign_apk_paths[:1300]
    benign_val_apks = benign_apk_paths[1300:1500]
    malicious_train_apks = malicious_apk_paths[:1300]
    malicious_val_apks = malicious_apk_paths[1300:1500]

    log_path = os.path.join(save_dir, f"train_{model_name}_{color}_{mem_regions}_{pretrained_val}_misclassified-regions.log")
    logger.remove()
    logger.add(log_path, format="{message}")

    is_dexray = model_name == "dexray"

    # Dataset
    train_benign = CustomDataset(root_dir, None if is_dexray else transform_no_augmentation, apk_list=benign_train_apks, class_filter=0, selection_mode=mem_regions, color_mode=color)
    train_malicious = CustomDataset(root_dir, None if is_dexray else transform_no_augmentation, apk_list=malicious_train_apks, class_filter=1, selection_mode=mem_regions, color_mode=color)
    val_benign = CustomDataset(root_dir, None if is_dexray else transform_no_augmentation, apk_list=benign_val_apks, class_filter=0, selection_mode=mem_regions, color_mode=color)
    val_malicious = CustomDataset(root_dir, None if is_dexray else transform_no_augmentation, apk_list=malicious_val_apks, class_filter=1, selection_mode=mem_regions, color_mode=color)

    if is_dexray:
        for dset in [train_benign, train_malicious, val_benign, val_malicious]:
            dset.dexray_mode = True

    train_dataset = ConcatDataset([train_benign, train_malicious])
    val_dataset = ConcatDataset([val_benign, val_malicious])

    train_labels = [label for ds in [train_benign, train_malicious] for _, label, *_ in ds.images]
    class_counts = np.bincount(train_labels)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [weights[label] for label in train_labels]

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=WeightedRandomSampler(sample_weights, len(sample_weights)))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = build_model(model_name=model_name, num_classes=2, pretrained=pretrained_val).to(device)
    criterion = nn.BCELoss() if is_dexray else nn.CrossEntropyLoss(torch.tensor([0.5, 0.5], device=device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=10)

    best_val_accuracy = 0.0
    best_train_accuracy = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'roc_auc': [], 'pr_auc': []}
    num_epochs = 10
    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for images, labels, *_ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if is_dexray:
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels.float())
                preds = (outputs > 0.5).long()
            else:
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total * 100
        history['train_loss'].append(train_loss)

        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            torch.save(model.state_dict(), os.path.join(save_dir, f"{mem_regions}_{model_name}_{color}_best_train_{pretrained_val}.pth"))
            logger.info(f"\nModel saved at epoch {epoch+1} with TRAIN accuracy: {best_train_accuracy:.2f}%")
        

        # Validation phase
        model.eval()
        apk_probs = collections.defaultdict(list)
        apk_labels = {}
        apk_regions = collections.defaultdict(list)
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels, apk_names, region_descs in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if is_dexray:
                    outputs = outputs.squeeze()
                    probs = outputs
                else:
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                for i in range(len(labels)):
                    apk = apk_names[i]
                    prob = probs[i].item()
                    label = labels[i].item()
                    region = region_descs[i]
                    apk_probs[apk].append(prob)
                    apk_labels[apk] = label
                    apk_regions[apk].append((region, prob))
                    all_probs.append(prob)
                    all_labels.append(label)

        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        apk_preds = {apk: int(np.mean(probs) > best_threshold) for apk, probs in apk_probs.items()}
        apk_true = apk_labels
        pred_list = [apk_preds[a] for a in apk_true]
        true_list = list(apk_true.values())
        val_accuracy = np.mean(np.array(pred_list) == np.array(true_list)) * 100

        auc = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)

        val_loss = 0.0
        if is_dexray:
            val_loss = nn.BCELoss()(torch.tensor(all_probs), torch.tensor(all_labels).float()).item()
        else:
            val_loss = nn.CrossEntropyLoss()(torch.tensor([[1-p, p] for p in all_probs]), torch.tensor(all_labels)).item()
        
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['roc_auc'].append(roc_auc_score(true_list, [apk_preds[a] for a in apk_true]))
        history['pr_auc'].append(average_precision_score(true_list, [apk_preds[a] for a in apk_true]))

        # Misclassified APKs and their memory regions
        for apk in apk_true:
            if apk_preds[apk] != apk_true[apk]:
                print(f"[MISCLASSIFIED APK] {apk} | True={apk_true[apk]} | Pred={apk_preds[apk]}")
                sorted_regions = sorted(apk_regions[apk], key=lambda x: x[1], reverse=True)
                for region, prob in sorted_regions:
                    print(f"    Region: {region} | Prob: {prob:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(save_dir, f"{mem_regions}_{model_name}_{color}_best_validation_{pretrained_val}.pth"))
            logger.info(f"\nModel saved at epoch {epoch+1} with VALIDATION accuracy: {best_val_accuracy:.2f}%")

        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        logger.info(f"ROC AUC: {auc:.4f}, PR AUC: {pr_auc:.4f}")
        logger.info("\n Classification Report: \n" + str(classification_report(true_list, pred_list, target_names=['Benign', 'Malicious'])))
        logger.info("\n Confusion Matrix: \n" + str(confusion_matrix(true_list, pred_list)))

        # === Evaluate and Print Misclassified APKs with Regions ===
        apk_true = {apk: label for apk, label in apk_labels.items()}
        apk_misclassified = []

        for apk, pred in apk_preds.items():
            true = apk_true[apk]
            if pred != true:
                apk_misclassified.append(apk)
                logger.info(f"[MISCLASSIFIED APK] {apk} | True={true} | Predicted={pred}")
                regions = sorted(apk_regions[apk], key=lambda x: x[1], reverse=True)
                for region, prob in regions:
                    logger.info(f"    Region: {region} | Prob: {prob:.4f}")

        logger.info(f"\nTotal Misclassified APKs: {len(apk_misclassified)} / {len(apk_preds)}")

        
        torch.save(model.state_dict(), os.path.join(save_dir, f"{mem_regions}_{model_name}_{color}_last_{pretrained_val}.pth"))

        # === Plot Training History ===
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
        png_filename = f"{mem_regions}_{model_name}_{color}_training_metrics_{pretrained_val}.png"
        plt.savefig(os.path.join(save_dir, png_filename))
        #plt.savefig(f"{mem_regions}_{model_name}_{color}_training_metrics_{pretrained_val}")
        #plt.savefig("training_metrics.png")
        plt.show()




    # Final model save and plots can be added as needed


# === Run Training ===
#train_memory_regions('data_stack', 'resnet18', 'True', "RGB")
train_memory_regions('data_stack', 'dexray', 'false', 'L', '/home/ssanna/Desktop/malware_ram/Android/imgs/sections/memory_regions/data_stack/sota_CNN')