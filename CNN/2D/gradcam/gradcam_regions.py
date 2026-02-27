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

# === Transform ===
transform_no_augmentation = transforms.Compose([
    ResizeWithPadding(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


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
        return cam.mean()  # Return a single importance score

    def close(self):
        for handle in self.hook_handles:
            handle.remove()


def get_target_layer(model, model_name):
    if "resnet" in model_name:
        return model.layer4[-1]
    elif "mobilenet_v2" in model_name:
        return model.features[-1]
    elif "mobilenet_v3" in model_name:
        return model.features[-1]
    elif "efficientnet_b0" in model_name:
        return model.features[-1]
    elif "convnext" in model_name:
        return model.features[-1]
    else:
        raise NotImplementedError(f"No Grad-CAM layer mapping for {model_name}")


if __name__ == "__main__":
    # Config
    model_name = "resnet18"
    model_path = "/home/ssanna/Desktop/malware_ram/Android/imgs/sections/memory_regions/data_stack/resnet18_100epochs/data_stack_resnet18_RGB_best_validation_True.pth"  # <- Change this
    root_dir = "/mnt/malware_ram/Android"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_path = os.path.join(f"gradcam_RGB_resnet18_True_misclassified-regions.log")
    logger.remove()
    logger.add(log_path, format="{message}")
    # Load model
    model = build_model(model_name, num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # List all test APKs
    benign_val_apks = sorted([os.path.join(root_dir, "benign_dumps", apk) for apk in os.listdir(os.path.join(root_dir, "benign_dumps"))])[1300:1500]
    malicious_val_apks = sorted([os.path.join(root_dir, "dumps_dataset", apk) for apk in os.listdir(os.path.join(root_dir, "dumps_dataset"))])[1300:1500]
    apk_list = benign_val_apks + malicious_val_apks

    # Loop over each test APK
    for apk_path in apk_list:
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
            image_path = dataset.images[i][0]
            image_pil = Image.open(image_path).convert("RGB")
            input_tensor = transform_no_augmentation(image_pil).unsqueeze(0).to(device)

            target_layer = get_target_layer(model, model_name)
            gradcam = GradCAM(model, target_layer)
            score = gradcam.generate(input_tensor, class_idx=lbl)
            gradcam.close()

            region_scores.append((region_desc, score))

        # Sort and print most influential memory areas
        region_scores.sort(key=lambda x: x[1], reverse=True)
        logger.info("Most influential memory regions:")
        for region, score in region_scores[:5]:
            logger.info(f"  Region: {region} | Score: {score:.4f}")
