import os
import subprocess
from tqdm import tqdm

ram_dataset = "path/dataset"
classes = ["benign_dumps", "dumps_dataset"]

for apk_class in classes:
    class_path = os.path.join(ram_dataset, apk_class)
    for apk in tqdm(os.listdir(class_path), desc=f"Processing {apk_class}", unit="apk"):
        apk.split(".apk")[0]
        
        apk_path = "path/to/benign_apk"
        save_strings_path = "/path/save/strings"
        
        bin_complete = os.path.join(apk_path, apkname)
        apkname = apk.split(".apk")[0]
        out_str = os.path.join(save_strings_path, apkname)

        if os.path.exists(out_str):
            with open(out_str) as f:
                strings_content = f.readlines()
            out = subprocess.run(f'cat {out_str}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if len(out.stderr) > 1:
                bin2strings = subprocess.run(f'strings {bin_complete} > {out_str}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            else:
                print("Already processed, skip \n")
        
        else:
            bin2strings = subprocess.run(f'strings {bin_complete} > {out_str}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        

