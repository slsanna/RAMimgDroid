import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm
from torch.utils.data import Dataset
import re
import torch.nn as nn
from loguru import logger

# === Transforms ===
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

# === Dataset ===
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
        }

        class_dirs = {0: 'benign_dumps', 1: 'malware_dumps'}
        if class_filter is not None:
            class_dirs = {class_filter: class_dirs[class_filter]}

        for label, class_dir in class_dirs.items():
            class_dir_path = os.path.join(root_dir, class_dir)
            apk_dirs = apk_list if apk_list else [os.path.join(class_dir_path, apk) for apk in os.listdir(class_dir_path)]
            for apk_dir in tqdm(apk_dirs, desc=f"Processing {class_dir}", unit="apk"):
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
                        self.images.append((image_path, label, os.path.basename(apk_dir), desc))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label, apk_name, region_desc = self.images[idx]
        image = Image.open(img_path).convert("L" if self.use_grayscale else "RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, apk_name, region_desc

# === Model ===
def build_model(model_name="resnet18", num_classes=2, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

def get_target_layer(model, model_name):
    if "resnet" in model_name:
        return model.layer4[-1]
    raise NotImplementedError(f"No Grad-CAM mapping for {model_name}")

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
        return cam.squeeze().cpu().numpy()

    def close(self):
        for h in self.hook_handles:
            h.remove()


def get_image_size(data_length):
    if data_length < 10240:
        width = 32
    elif 10240 <= data_length <= 10240 * 3:
        width = 64
    elif 10240 * 3 <= data_length <= 10240 * 6:
        width = 128
    elif 10240 * 6 <= data_length <= 10240 * 10:
        width = 256
    elif 10240 * 10 <= data_length <= 10240 * 20:
        width = 384
    elif 10240 * 20 <= data_length <= 10240 * 50:
        width = 512
    elif 10240 * 50 <= data_length <= 10240 * 100:
        width = 768
    else:
        width = 1024
    height = (data_length + 2) // 3 // width + 1
    return width, height

def map_resized_to_original(row, col, orig_w, orig_h, target_size=224):
    scale = target_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    if pad_x <= col < pad_x + new_w and pad_y <= row < pad_y + new_h:
        mapped_x = int((col - pad_x) * orig_w / new_w)
        mapped_y = int((row - pad_y) * orig_h / new_h)
        return mapped_y, mapped_x
    return None

def pixel_to_byte_offsets(row, col, width):
    pixel_index = row * width + col
    return pixel_index * 3, pixel_index * 3 + 1, pixel_index * 3 + 2

def extract_pixel_bytes_from_binary(bin_path, coords, orig_w, orig_h):
    with open(bin_path, 'rb') as f:
        binary_data = f.read()
    width, height = get_image_size(len(binary_data))
    results = []
    for row, col in coords:
        mapped = map_resized_to_original(row, col, orig_w, orig_h)
        if mapped is None:
            results.append(((row, col), b'', '[padding area]'))
            continue
        mapped_row, mapped_col = mapped
        b_start, _, b_end = pixel_to_byte_offsets(mapped_row, mapped_col, width)

        if b_end >= len(binary_data):
            byte_seq = binary_data[b_start:] if b_start < len(binary_data) else b''
        else:
            byte_seq = binary_data[b_start:b_end + 1]

        full_line = byte_seq
        if b_start < len(binary_data):
            line_start = binary_data.rfind(b'\n', 0, b_start) + 1
            line_end = binary_data.find(b'\n', b_end)
            if line_end == -1:
                line_end = len(binary_data)
            full_line = binary_data[line_start:line_end]

        try:
            decoded = full_line.decode('utf-8', errors='replace')
        except:
            decoded = '[decode error]'

        results.append(((mapped_row, mapped_col), byte_seq, decoded))
    return results

if __name__ == "__main__":
    apk_path = "apk/path"

    model_name = "resnet18"
    model_path = "pretrained/path"
    root_dir = "dataset/path"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.remove()
    logger.add("top_pixels_per_top_region.log", format="{message}")

    model = build_model(model_name, num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"\n=== APK: {apk_path} ===")
    label = 0 if "benign_dumps" in apk_path else 1

    dataset = CustomDataset(
        root_dir=root_dir,
        transform=transform_no_augmentation,
        apk_list=[apk_path],
        class_filter=label,
        selection_mode="data_stack",
        color_mode="RGB"
    )

    region_scores = []

    for i in range(len(dataset)):
        img_tensor, lbl, apk_name, region_desc = dataset[i]
        input_tensor = img_tensor.unsqueeze(0).to(device)
        gradcam = GradCAM(model, get_target_layer(model, model_name))
        cam = gradcam.generate(input_tensor, class_idx=lbl)
        gradcam.close()
        region_scores.append((i, region_desc, cam, img_tensor.shape[2], img_tensor.shape[1]))

    if not region_scores:
        exit()

    region_scores.sort(key=lambda x: np.mean(x[2]), reverse=True)

    for region_rank in range(len(region_scores)):
        region_index, region_desc, cam, orig_h, orig_w = region_scores[region_rank]

        heatmap = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
        heatmap = F.interpolate(heatmap, size=(224, 224), mode='bilinear', align_corners=False)
        heatmap_np = heatmap.squeeze().cpu().numpy()

        max_val = heatmap_np.max()
        max_coords = np.argwhere(heatmap_np == max_val)
        max_coords_list = [tuple(coord) for coord in max_coords]

        print(f"Region {region_rank+1}: {region_desc}")
        print(f"Most influential pixel(s) (row, col) with score {max_val:.4f}:")
        for coord in max_coords_list:
            print(f" - {coord}")

        maps_path = os.path.join(apk_path, 'maps.txt')
        region_addr = dataset.images[region_index][0].split('/')[-1].split('_')[0].replace('0x', '')
        binary_file_path = os.path.join(apk_path, 'r--', f'0x{region_addr}_dump.data')

        print(f"[DEBUG] Using binary file: {binary_file_path}")

        if os.path.exists(binary_file_path):
            results = extract_pixel_bytes_from_binary(binary_file_path, max_coords_list, orig_w, orig_h)
            for (mapped_row, mapped_col), byte_seq, decoded in results:
                print(f"Mapped Pixel ({mapped_row},{mapped_col}): {byte_seq} → '{decoded}'")
        else:
            print(f"Binary file not found: {binary_file_path}")