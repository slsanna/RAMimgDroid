import os
import re
from tqdm import tqdm
from PIL import Image
from collections import defaultdict, Counter

root_dataset = "path/dataset"
classes_dataset = {
    'benign_dumps': 'goodware',
    'malware_dumps': 'malware'
}
# Your selection_map
selection_map = {
    "stack": [r"\[stack\]"],
    "memory_regions": [r"\[stack\]", r"\[vvar\]"],
    "complete_memory_regions": [r"\[heap\]", r"\[anon:libc_malloc\]", r"\[stack\]",
        r"\[vdso\]", r"\[vvar\]", r"\[vsyscall\]",
        r"\(deleted\)", r"u:object_r.*", r"data@resource-cache@" ],
    "base_apk_memory": [r"\[stack\]", r"\[vvar\]", r"base\.apk", r"base\.vdex", r"base\.odex"],
    "base_apk_stack": [r"\[stack\]", r"base\.apk", r"base\.vdex", r"base\.odex"],
    "base_apk": [r"base\.apk", r"base\.vdex", r"base\.odex"],
    "apk_full": [
        r"base\.apk", r"base\.vdex", r"base\.odex", r"/data/.*/lib/.+\.so$",
        r"\[anon:dalvik.*\]", r"\[anon:.*\.apk.*\]", r"\.db$", r"\.db-shm$", r"\.db-wal$",
        r"Cookies", r"Preferences", r"ua_preferences", r"data@resource-cache@", r"\.dex$",
        r"/data/.+\.odex$"
    ],
    "apk_full_memory": [r"\[stack\]", r"\[vvar\]",
        r"base\.apk", r"base\.vdex", r"base\.odex", r"/data/.*/lib/.+\.so$",
        r"\[anon:dalvik.*\]", r"\[anon:.*\.apk.*\]", r"\.db$", r"\.db-shm$", r"\.db-wal$",
        r"Cookies", r"Preferences", r"ua_preferences", r"data@resource-cache@", r"\.dex$",
        r"/data/.+\.odex$"
    ],
    "apk_full_stack": [r"\[stack\]", r"base\.apk", r"base\.vdex", r"base\.odex", r"/data/.*/lib/.+\.so$",
        r"\[anon:dalvik.*\]", r"\[anon:.*\.apk.*\]", r"\.db$", r"\.db-shm$", r"\.db-wal$",
        r"Cookies", r"Preferences", r"ua_preferences", r"data@resource-cache@", r"\.dex$",
        r"/data/.+\.odex$"],
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

# Master structure: {selection_mode -> {pattern -> {class -> Counter of (w, h)}}}
all_stats = defaultdict(lambda: defaultdict(lambda: {'goodware': Counter(), 'malware': Counter()}))

for class_folder, class_label in classes_dataset.items():
    class_dir_path = os.path.join(root_dataset, class_folder)

    for apk in tqdm(os.listdir(class_dir_path), desc=f"Processing {class_label}", unit="apk"):
        apk_dir = os.path.join(class_dir_path, apk)
        maps_path = os.path.join(apk_dir, 'maps.txt')
        images_dir = os.path.join(apk_dir, 'images', 'complete', 'RGB')

        if not os.path.exists(maps_path) or not os.path.exists(images_dir):
            continue

        try:
            with open(maps_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading maps.txt in {apk_dir}: {e}")
            continue

        for selection_mode, pattern_list in selection_map.items():
            compiled_patterns = [re.compile(p) for p in pattern_list]

            for line in lines:
                parts = line.strip().split()
                if not parts or '-' not in parts[0]:
                    continue

                address = parts[0].split('-')[0].strip()
                desc = parts[-1] if len(parts) > 5 else ""

                for pattern in compiled_patterns:
                    if re.search(pattern, desc):
                        image_filename = f"0x{address}_dump_RGB.png"
                        image_path = os.path.join(images_dir, image_filename)

                        if os.path.exists(image_path):
                            try:
                                with Image.open(image_path) as img:
                                    width, height = img.size
                                    all_stats[selection_mode][pattern.pattern][class_label][(width, height)] += 1
                            except Exception as e:
                                print(f"Failed to open image {image_path}: {e}")
                        break

# --- Print results ---
for selection_mode, patterns_data in all_stats.items():
    print(f"\n=== Selection Mode: {selection_mode} ===")
    for pattern, classes_data in patterns_data.items():
        print(f"  Pattern: {pattern}")
        for class_label, counter in classes_data.items():
            print(f"    {class_label.capitalize()}:")
            if not counter:
                print("      No data")
            else:
                for (w, h), count in counter.items():
                    print(f"      {w}x{h}: {count} images")
