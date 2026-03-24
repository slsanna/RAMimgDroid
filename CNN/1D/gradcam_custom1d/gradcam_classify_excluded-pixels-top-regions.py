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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import warnings
from loguru import logger
import collections

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Custom 1D CNN Model ===
class RGB1DCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=2):
        super(RGB1DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(32)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

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
            img = Image.open(img_path).convert("RGB").resize((1024, 3))
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.tensor(img).permute(2, 1, 0).mean(dim=2)
            return img, label, apk, desc
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))

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
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None, return_map=False):
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0).to(device)
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        self.model.zero_grad()
        output.backward(gradient=one_hot)
        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam if return_map else cam.mean()

def mask_top_features_1d(input_tensor, cam, num_features=50):
    cam_tensor = torch.tensor(cam)
    topk = torch.topk(cam_tensor, min(num_features, cam_tensor.numel())).indices
    masked = input_tensor.clone()
    masked[:, topk] = 0.0
    return masked

def test_rgb1dcnn_exclude_top_region(mem_regions, model_name, color, save_dir):
    root_dir = "dataset_path"
    model_path = os.path.join("path/to/1d_cnn/custom1d/1D_data_stack_rgb1dcnn_best_val.pth")
    log_path = os.path.join("exclude-pixels-top-regions.log")

    logger.remove()
    logger.add(log_path, format="{message}")

    benign_paths = sorted(os.listdir(os.path.join(root_dir, "benign_dumps")))[1500:]
    malware_paths = sorted(os.listdir(os.path.join(root_dir, "dumps_dataset")))[1500:]

    y_true, y_pred = [], []

    for apk in tqdm(benign_paths + malware_paths, desc="Evaluating APKs"):
        label = 0 if apk in benign_paths else 1
        y_true.append(label)

        dataset = CustomImageDataset(root_dir, selection_mode=mem_regions, apk_list=[apk], class_filter=label)
        if len(dataset) == 0:
            y_pred.append(-1)
            logger.warning(f"No samples found for APK: {apk}")
            continue

        model = RGB1DCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        region_scores = []
        for i in range(len(dataset)):
            x, lbl, apk_name, region_desc = dataset[i]
            target_layer = model.features[6]
            gradcam = GradCAM1D(model, target_layer)
            score = gradcam.generate(x, class_idx=lbl)
            region_scores.append((i, region_desc, score))

        if not region_scores:
            y_pred.append(-1)
            continue

        region_scores.sort(key=lambda x: x[2], reverse=True)
        top_region_index = region_scores[0][0]

        predictions = []
        for i in range(len(dataset)):
            x, lbl, apk_name, region_desc = dataset[i]
            input_tensor = x.clone().to(device)

            if i == top_region_index:
                gradcam = GradCAM1D(model, model.features[6])
                cam = gradcam.generate(input_tensor, class_idx=lbl, return_map=True)
                input_tensor = mask_top_features_1d(input_tensor, cam, num_features=50)

            with torch.no_grad():
                output = model(input_tensor.unsqueeze(0))
                pred = output.argmax(dim=1).item()
                predictions.append(pred)

        if predictions:
            majority_pred = max(set(predictions), key=predictions.count)
        else:
            majority_pred = -1

        y_pred.append(majority_pred)
        result = "CORRECT" if majority_pred == label else "WRONG"
        logger.info(f"APK: {apk} | True: {label} | Pred: {majority_pred} --> {result}")

    filtered = [(p, t) for p, t in zip(y_pred, y_true) if p != -1]
    if filtered:
        yp, yt = zip(*filtered)
        print("\n=== Confusion Matrix ===")
        print(confusion_matrix(yt, yp))
        print("\n=== Classification Report ===")
        print(classification_report(yt, yp, target_names=["Benign", "Malicious"]))
    else:
        print("No valid predictions to report.")

# === Run test with Grad-CAM masking ===
save_dir = "path/to/save/results/custom1d"
test_rgb1dcnn_exclude_top_region('data_stack', 'rgb1dcnn', 'RGB', save_dir)
