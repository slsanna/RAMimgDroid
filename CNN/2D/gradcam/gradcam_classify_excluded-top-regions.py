import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from loguru import logger
from torchvision import transforms, models
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import re
import torch.nn as nn


# === DATASET ===
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, apk_list=None, class_filter=None, selection_mode="stack", color_mode="RGB"):
        self.images = []
        self.transform = transform
        self.color_mode = color_mode
        self.use_grayscale = (color_mode != "RGB")

        self.selection_map = {
            "data_stack": [r"^/data/.*", r"\[stack\]"],
            "stack": [r"\[stack\]"],
            "memory_regions": [r"\[stack\]", r"\[vvar\]"]
            # Add more modes as needed
        }

        class_dirs = {0: 'benign_dumps', 1: 'dumps_dataset'}
        if class_filter is not None:
            class_dirs = {class_filter: class_dirs[class_filter]}

        for label, class_dir in class_dirs.items():
            class_dir_path = os.path.join(root_dir, class_dir)
            apk_dirs = apk_list if apk_list else [os.path.join(class_dir_path, apk) for apk in os.listdir(class_dir_path)]
            for apk_dir in tqdm(apk_dirs, desc=f"Processing {class_dir}", unit="apk"):
                apk = os.path.basename(apk_dir)
                maps_path = os.path.join(apk_dir, 'maps.txt')
                images_dir = os.path.join(apk_dir, 'images', 'complete', 'RGB')
                if not os.path.exists(maps_path) or not os.path.exists(images_dir):
                    continue
                selected_regions = {}
                with open(maps_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        address = parts[0].split('-')[0]
                        desc = line.split(None, 5)[-1] if len(parts) == 6 else ""
                        for pattern in self.selection_map.get(selection_mode, []):
                            if re.search(pattern, desc):
                                selected_regions[address] = desc
                for addr, desc in selected_regions.items():
                    image_file = f"0x{addr}_dump_RGB.png"
                    image_path = os.path.join(images_dir, image_file)
                    if os.path.exists(image_path):
                        self.images.append((image_path, label, apk, desc))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label, apk_name, region_desc = self.images[idx]
        try:
            image = Image.open(img_path).convert("L" if self.use_grayscale else "RGB")
            if self.transform:
                image = self.transform(image)
            return image, label, apk_name, region_desc
        except Exception:
            return self.__getitem__((idx + 1) % len(self.images))


# === TRANSFORMS ===
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === MODEL BUILDER ===
def build_model(model_name="resnet18", num_classes=2, pretrained=True):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
)

    else:
        raise ValueError(f"Model {model_name} not supported")
    return model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        self.model.zero_grad()
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-6)
        return cam.mean()

    def close(self):
        for handle in self.hook_handles:
            handle.remove()

def get_target_layer(model, model_name):
    if "resnet" in model_name:
        return model.layer4[-1]
    else:
        raise NotImplementedError(f"No Grad-CAM layer mapping for {model_name}")

from sklearn.metrics import confusion_matrix, classification_report

if __name__ == "__main__":
    # Config
    model_name = "resnet18"
    model_path = "/home/ssanna/Desktop/malware_ram/Android/imgs/sections/memory_regions/data_stack/resnet18_100epochs/data_stack_resnet18_RGB_best_validation_True.pth"
    root_dir = "/mnt/malware_ram/Android"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_path = os.path.join(f"gradcam_RGB_resnet18_True_excluded-top-regions.log")

    logger.remove()
    logger.add(log_path, format="{message}")

    # Load model
    model = build_model(model_name, num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Get APK list
    benign_val_apks = sorted([os.path.join(root_dir, "benign_dumps", apk) for apk in os.listdir(os.path.join(root_dir, "benign_dumps"))])[1500:]
    malicious_val_apks = sorted([os.path.join(root_dir, "dumps_dataset", apk) for apk in os.listdir(os.path.join(root_dir, "dumps_dataset"))])[1500:]
    apk_list = benign_val_apks + malicious_val_apks

    y_true = []
    y_pred = []

    for apk_path in apk_list:
        print(f"\n=== APK: {apk_path} ===")
        label = 0 if "benign_dumps" in apk_path else 1
        y_true.append(label)

        dataset = CustomDataset(
            root_dir=root_dir,
            transform=transform_no_augmentation,
            apk_list=[apk_path],
            class_filter=label,
            selection_mode="data_stack",
            color_mode="RGB"
        )

        region_scores = []
        region_predictions = []

        # Step 1: Get Grad-CAM scores
        for i in range(len(dataset)):
            img_tensor, lbl, apk_name, region_desc = dataset[i]
            image_path = dataset.images[i][0]
            image_pil = Image.open(image_path).convert("RGB")
            input_tensor = transform_no_augmentation(image_pil).unsqueeze(0).to(device)

            target_layer = get_target_layer(model, model_name)
            gradcam = GradCAM(model, target_layer)
            score = gradcam.generate(input_tensor, class_idx=lbl)
            gradcam.close()

            region_scores.append((i, region_desc, score))

        # Step 2: Filter out top-N most influential regions
        region_scores.sort(key=lambda x: x[2], reverse=True)
        top_n_to_exclude = 5
        excluded_indices = set([idx for idx, _, _ in region_scores[:top_n_to_exclude]])

        # Step 3: Re-run predictions without top regions
        for i in range(len(dataset)):
            if i in excluded_indices:
                continue  # Skip top regions

            img_tensor, lbl, apk_name, region_desc = dataset[i]
            input_tensor = img_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                pred_label = output.argmax(dim=1).item()
                region_predictions.append(pred_label)

        # Step 4: Aggregate APK-level prediction
        if region_predictions:
            majority_label = max(set(region_predictions), key=region_predictions.count)
        else:
            majority_label = -1  # fallback if everything excluded

        y_pred.append(majority_label)
        result = "CORRECT" if majority_label == label else "WRONG"
        print(f"Predicted Label: {majority_label} | True Label: {label} --> {result}")
        logger.info(f"APK: {apk_path}")
        logger.info(f"Predicted (w/o top-{top_n_to_exclude} regions): {majority_label} | True: {label} | Result: {result}")

    from sklearn.metrics import confusion_matrix, classification_report

    valid_indices = [i for i, p in enumerate(y_pred) if p != -1]
    filtered_y_true = [y_true[i] for i in valid_indices]
    filtered_y_pred = [y_pred[i] for i in valid_indices]

    # Step 5: Print Confusion Matrix
    print("\n=== Confusion Matrix (Using ONLY Most Influential Regions) ===")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Malicious"]))
