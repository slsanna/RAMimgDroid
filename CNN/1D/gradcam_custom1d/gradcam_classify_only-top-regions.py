import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Model ===
class RGB1DCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveMaxPool1d(32)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 32, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# === GradCAM for 1D ===
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

    def generate(self, input_tensor, class_idx=None):
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
        cam = (weights * self.activations).sum(dim=1).squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam

# === Dataset ===
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, selection_mode="data_stack", apk_list=None, class_filter=None):
        self.samples = []
        import re
        self.selection_map = {
            "data_stack": [r"^/data/.*", r"\[stack\]"],
            "stack": [r"\[stack\]"],
            "memory_regions": [r"\[stack\]", r"\[vvar\]"]
        }
        class_dirs = {0: 'benign_dumps', 1: 'dumps_dataset'}
        if class_filter is not None:
            class_dirs = {class_filter: class_dirs[class_filter]}
        for label, folder in class_dirs.items():
            folder_path = os.path.join(root_dir, folder)
            apks = apk_list if apk_list else os.listdir(folder_path)
            for apk in apks:
                apk_path = os.path.join(folder_path, apk)
                maps_file = os.path.join(apk_path, "maps.txt")
                img_dir = os.path.join(apk_path, "images", "horizontal", "RGB")
                if not os.path.exists(maps_file) or not os.path.exists(img_dir):
                    continue
                with open(maps_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 6:
                            continue
                        addr = parts[0].split("-")[0]
                        desc = line.split(None, 5)[-1]
                        if any(re.search(p, desc) for p in self.selection_map.get(selection_mode, [])):
                            img_path = os.path.join(img_dir, f"0x{addr}_dump_RGB_horizontal.png")
                            if os.path.exists(img_path):
                                self.samples.append((img_path, label, apk, desc))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, apk, desc = self.samples[idx]
        img = Image.open(img_path).convert("RGB").resize((1024, 3))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 1, 0).mean(dim=2)  # [C, L]
        return img, label, apk, desc

# === Main Evaluation ===
if __name__ == "__main__":
    root_dir = "path/dataset"
    model_path = "path/to/custom1d/1D_data_stack_rgb1dcnn_best_val.pth"

    benign_apks = sorted(os.listdir(os.path.join(root_dir, "benign_dumps")))[1500:]
    malware_apks = sorted(os.listdir(os.path.join(root_dir, "dumps_dataset")))[1500:]
    apk_list = [(apk, 0) for apk in benign_apks] + [(apk, 1) for apk in malware_apks]

    model = RGB1DCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    for apk, label in tqdm(apk_list, desc="Processing APKs"):
        y_true.append(label)
        dataset = CustomImageDataset(root_dir, apk_list=[apk], selection_mode="data_stack", class_filter=label)
        if len(dataset) == 0:
            y_pred.append(-1)
            continue

        region_scores = []
        for i in range(len(dataset)):
            x, _, _, _ = dataset[i]
            gradcam = GradCAM1D(model, model.features[-3])
            score = gradcam.generate(x.to(device), class_idx=label)
            region_scores.append((i, score))

        region_scores.sort(key=lambda x: x[1], reverse=True)
        top_n = 5
        included_indices = [i for i, _ in region_scores[:top_n]]

        region_predictions = []
        for i in included_indices:
            x, _, _, _ = dataset[i]
            with torch.no_grad():
                pred = model(x.unsqueeze(0).to(device)).argmax(dim=1).item()
                region_predictions.append(pred)

        if region_predictions:
            final_pred = max(set(region_predictions), key=region_predictions.count)
        else:
            final_pred = -1

        y_pred.append(final_pred)

    # Results
    valid = [(p, t) for p, t in zip(y_pred, y_true) if p != -1]
    if valid:
        yp, yt = zip(*valid)
        print("Confusion Matrix:\n", confusion_matrix(yt, yp))
        print("\nClassification Report:\n", classification_report(yt, yp, target_names=["Benign", "Malicious"]))
    else:
        print("No valid predictions.")
