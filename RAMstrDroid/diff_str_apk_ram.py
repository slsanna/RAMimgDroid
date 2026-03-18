import subprocess
import os
from tqdm import tqdm

ram_dataset = "path/dataset"
classes = ["benign", "malware"]

for apk_class in classes:
    class_path = os.path.join(ram_dataset, apk_class)
    for apk in tqdm(os.listdir(class_path), desc=f"Processing {apk_class}", unit="apk"):
        strings_ram_path = os.path.join(class_path, apk, 'strings', 'strings.txt')
        if ".apk" in apk:
            apkname = apk.split(".apk")[0]
        else:
            apkname = apk.split("_")[0]
        apk_class = "malicious"
        apk_path = os.path.join("path/to/strings_apk", apk_class, apkname)
        out_dir = os.path.join("path/to/diff_strings", apk_class, apkname)
        out = subprocess.run(f'diff -c {strings_ram_path} {apk_path} > {out_dir}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)