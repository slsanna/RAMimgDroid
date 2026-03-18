import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
import collections
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, roc_curve
#from complete_apks import build_model, CustomDataset, transform_no_augmentation  # Replace with your actual module or paste definitions here
from torchvision import transforms, models
import torch.nn as nn
import os
import fnmatch
import sys
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.optim as optim

from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, vit_b_16, ViT_B_16_Weights, inception_v3, Inception_V3_Weights
import collections
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import warnings
from loguru import logger
import random
import collections
from glob import glob
import os
import random
import collections
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, classification_report, confusion_matrix

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

        class_dirs = {0: 'benign_dumps', 1: 'malware_dumps'}
        if class_filter is not None:
            class_dirs = {class_filter: class_dirs[class_filter]}

        temp_images = []

        for label, class_dir in class_dirs.items():
            class_dir_path = os.path.join(root_dir, class_dir)
            if not os.path.exists(class_dir_path):
                continue

            #apk_dirs = apk_list if apk_list is not None else [os.path.join(class_dir_path, apk) for apk in os.listdir(class_dir_path)]
            if apk_list is not None:
                apk_dirs = apk_list
            else:
                apk_dirs = [os.path.join(class_dir_path, apk) for apk in os.listdir(class_dir_path)]

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

    def __getitem__(self, idx):
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


# === CONFIGURATION ===
mem_regions = "data_stack"
model_name = "resnet18"
color_mode = "RGB"
pretrained_val = True  # This flag only controls architecture init, not weight loading
model_path = f"pretrained/model/path"
root_dir = "dataset/path"
batch_size = 32

# === DEVICE SETUP ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD TEST APKS ===
benign_apk_paths = sorted(glob(f"{root_dir}/benign_dumps/*"))[1500:]
malicious_apk_paths = sorted(glob(f"{root_dir}/malware_dumps/*"))[1500:]

# === BUILD DATASETS ===
test_benign = CustomDataset(
    root_dir,
    transform=transform_no_augmentation,
    apk_list=benign_apk_paths,
    class_filter=0,
    selection_mode=mem_regions,
    color_mode=color_mode
)

test_malicious = CustomDataset(
    root_dir,
    transform=transform_no_augmentation,
    apk_list=malicious_apk_paths,
    class_filter=1,
    selection_mode=mem_regions,
    color_mode=color_mode
)

from torch.utils.data import ConcatDataset, DataLoader

test_dataset = ConcatDataset([test_benign, test_malicious])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# === BUILD & LOAD MODEL ===
model = build_model(model_name=model_name, num_classes=2, pretrained=False)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === INFERENCE ===
all_apk_probs = collections.defaultdict(list)
all_apk_labels = {}
all_apk_regions = collections.defaultdict(list)
all_labels, all_probs = [], []

with torch.no_grad():
    for images, labels, apk_names, region_descs in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = (probs > 0.5).long()

        for i in range(len(labels)):
            apk = apk_names[i]
            prob = probs[i].item()
            label = labels[i].item()
            region = region_descs[i]
            all_apk_probs[apk].append(prob)
            all_apk_labels[apk] = label
            all_apk_regions[apk].append((region, prob))
            all_labels.append(label)
            all_probs.append(prob)

# === THRESHOLD AND AGGREGATION ===
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
best_threshold = thresholds[np.argmax(tpr - fpr)]

apk_preds = {}
apk_true = {}
for apk, probs in all_apk_probs.items():
    avg_prob = np.mean(probs)
    pred = int(avg_prob > best_threshold)
    apk_preds[apk] = pred
    apk_true[apk] = all_apk_labels[apk]

apk_pred_list = [apk_preds[apk] for apk in apk_true]
apk_true_list = list(apk_true.values())

# === REPORT ===
accuracy = np.mean(np.array(apk_pred_list) == np.array(apk_true_list)) * 100
print(f"\n=== Evaluation on Test Set ===")
print(f"Accuracy: {accuracy:.2f}%")
print("Classification Report:")
print(classification_report(apk_true_list, apk_pred_list, target_names=["Benign", "Malicious"]))
print("Confusion Matrix:")
print(confusion_matrix(apk_true_list, apk_pred_list))
print("ROC AUC:", roc_auc_score(all_labels, all_probs))
print("PR AUC:", average_precision_score(all_labels, all_probs))

# === OPTIONAL: Misclassified APKs ===
print("\nMisclassified APKs with top suspicious regions:")
for apk in apk_preds:
    if apk_preds[apk] != apk_true[apk]:
        print(f"[MISCLASSIFIED] {apk} | True: {apk_true[apk]} | Predicted: {apk_preds[apk]}")
        sorted_regions = sorted(all_apk_regions[apk], key=lambda x: x[1], reverse=True)
        for region, prob in sorted_regions[:5]:
            print(f"    Region: {region} | Prob: {prob:.4f}")
