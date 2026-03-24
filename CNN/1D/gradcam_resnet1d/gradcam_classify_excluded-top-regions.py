import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from loguru import logger
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, classification_report

# === MODEL ===
class ResNet1DTransfer(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet1DTransfer, self).__init__()
        from torchvision import models
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])  # output: [B, 512, 7, T]
        self.pool = nn.AdaptiveAvgPool2d((1, None))  # → [B, 512, 1, T]
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
        x = self.feature_extractor(x)       # [B, 512, 7, T]
        x = self.pool(x).squeeze(2)         # [B, 512, T]
        return self.classifier(x)


# === DATASET ===
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, apk_list=None, class_filter=None, selection_mode="stack"):
        self.images = []
        self.transform = transform
        self.selection_map = {
            "data_stack": [r"^/data/.*", r"\[stack\]"],
            "stack": [r"\[stack\]"],
        }

        class_dirs = {0: 'benign_dumps', 1: 'dumps_dataset'}
        if class_filter is not None:
            class_dirs = {class_filter: class_dirs[class_filter]}

        for label, class_dir in class_dirs.items():
            class_path = os.path.join(root_dir, class_dir)
            apk_dirs = apk_list if apk_list else [os.path.join(class_path, apk) for apk in os.listdir(class_path)]
            for apk_dir in tqdm(apk_dirs, desc=f"Processing {class_dir}", unit="apk"):
                maps_path = os.path.join(apk_dir, 'maps.txt')
                images_dir = os.path.join(apk_dir, 'images', 'horizontal', 'RGB')
                if not os.path.exists(maps_path) or not os.path.exists(images_dir):
                    continue
                with open(maps_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        addr = parts[0].split('-')[0]
                        desc = line.split(None, 5)[-1] if len(parts) == 6 else ""
                        if any(re.search(p, desc) for p in self.selection_map.get(selection_mode, [])):
                            img_path = os.path.join(images_dir, f"0x{addr}_dump_RGB_horizontal.png")
                            if os.path.exists(img_path):
                                self.images.append((img_path, label, os.path.basename(apk_dir), desc))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label, apk_name, region_desc = self.images[idx]
        try:
            image = Image.open(img_path).convert("RGB").resize((1024, 224))
            if self.transform:
                image = self.transform(image)
            return image, label, apk_name, region_desc
        except Exception:
            return self.__getitem__((idx + 1) % len(self.images))


# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.ToTensor(),  # (C, H, W)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# === GradCAM ===
class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None, return_map=False):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        self.model.zero_grad()
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = torch.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-6)
        return cam if return_map else cam.mean()


# === MAIN ===
if __name__ == "__main__":
    model_path = "/home/ssanna/Desktop/malware_ram/Android/imgs/1d_cnn/resnet/resnet18-data_stack_resnet1d_RGB_best_validation_True.pth"
    root_dir = "/mnt/malware_ram/Android"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.remove()
    logger.add("resnet1d_exclude_top_regions.log", format="{message}")

    model = ResNet1DTransfer()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    benign_apks = sorted([os.path.join(root_dir, "benign_dumps", apk) for apk in os.listdir(os.path.join(root_dir, "benign_dumps"))])[1500:]
    malicious_apks = sorted([os.path.join(root_dir, "dumps_dataset", apk) for apk in os.listdir(os.path.join(root_dir, "dumps_dataset"))])[1500:]
    apk_list = benign_apks + malicious_apks

    y_true = []
    y_pred = []

    for apk_path in apk_list:
        label = 0 if "benign_dumps" in apk_path else 1
        y_true.append(label)
        print(f"\n=== APK: {apk_path} ===")

        dataset = CustomDataset(root_dir=root_dir, transform=transform,
                                apk_list=[apk_path], class_filter=label, selection_mode="data_stack")

        region_scores = []
        for i in range(len(dataset)):
            img_tensor, lbl, apk_name, region_desc = dataset[i]
            input_tensor = img_tensor.unsqueeze(0).to(device)
            gradcam = GradCAM1D(model, model.feature_extractor[-1])
            try:
                score = gradcam.generate(input_tensor, class_idx=lbl)
            except Exception as e:
                logger.warning(f"GradCAM failed on {apk_name} - {region_desc}: {e}")
                continue
            region_scores.append((i, region_desc, score))

        if not region_scores:
            logger.info(f"⚠️ No valid regions for APK: {apk_path}")
            y_pred.append(-1)
            continue

        # Exclude top-N
        top_n = 5
        region_scores.sort(key=lambda x: x[2], reverse=True)
        excluded = set(idx for idx, _, _ in region_scores[:top_n])

        region_predictions = []
        for i in range(len(dataset)):
            if i in excluded:
                continue
            img_tensor, lbl, _, _ = dataset[i]
            input_tensor = img_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_tensor)
                pred = out.argmax(dim=1).item()
                region_predictions.append(pred)

        if region_predictions:
            majority = max(set(region_predictions), key=region_predictions.count)
        else:
            majority = -1

        y_pred.append(majority)
        result = "CORRECT" if majority == label else "WRONG"
        logger.info(f"APK: {apk_path}")
        logger.info(f"Predicted (excluding top-{top_n} regions): {majority} | True: {label} | Result: {result}")
        print(f"Predicted Label: {majority} | True Label: {label} --> {result}")

    # === METRICS ===
    valid = [(p, t) for p, t in zip(y_pred, y_true) if p != -1]
    if valid:
        yp, yt = zip(*valid)
        print("\nConfusion Matrix:")
        print(confusion_matrix(yt, yp))
        print("\nClassification Report:")
        print(classification_report(yt, yp, target_names=["Benign", "Malicious"]))
    else:
        print("⚠️ No valid predictions.")
