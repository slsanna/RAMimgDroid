import os
import subprocess
import argparse
from tqdm import tqdm

# Argument parsing
parser = argparse.ArgumentParser(description="Process APKs into horizontal images.")
parser.add_argument("ds_path", help="Path to the dataset directory containing APK folders")
parser.add_argument("--mode", default="RGB", help="Image mode (default: RGB)")
args = parser.parse_args()

ds_path = args.ds_path
mode = "RGB"
#enumerate(tqdm(os.listdir(ds_path), desc="Processing APKs")):
for i, apk in enumerate(tqdm(os.listdir(ds_path), desc="Processing APKs")):
    print(apk)
    if "images2D" in os.listdir(os.path.join(ds_path, apk)):
        dump_dir_apk = os.path.join(ds_path, apk, "r--")
        if "images1D" not in os.listdir(os.path.join(ds_path, apk)) and len(os.listdir(dump_dir_apk)) != 0:
            os.makedirs(os.path.join(ds_path, apk, "images1D"))
        out_horizontal_dir = os.path.join(ds_path, apk, "images1D")
        if mode not in os.listdir(out_horizontal_dir) or len(os.listdir((os.path.join(out_horizontal_dir, mode)))) != len(os.listdir(dump_dir_apk)):
            print(f"Converting to horizontal images {apk} ...")
            out = subprocess.run(
                ["python3", "/home/ssanna/Desktop/malware_ram/Android/imgs/1d_img_conversion.py", dump_dir_apk, "--output_dir", out_horizontal_dir, "--mode", mode],
                capture_output=True,
                text=False
            )
        else:
            print(f"Already processed {apk}, skipping conversion.")