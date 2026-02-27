import os
import subprocess
from tqdm import tqdm
import sys

class_ds = sys.argv[1]
ds_path = os.path.join("/mnt/malware_ram/", class_ds)
count = 0

for i, apk in enumerate(tqdm(os.listdir(ds_path), desc="Processing APKs")):
    if i != 1339:
        print("Now processing ", apk)
        bin_complete = os.path.join(ds_path, apk, "r--")
        complete_str_path = os.path.join(ds_path, apk, 'strings')
        out_str = os.path.join(complete_str_path, "strings.txt")
        if not os.path.exists(complete_str_path):
            os.makedirs(complete_str_path)

        if os.path.exists(out_str):
            with open(out_str) as f:
                strings_content = f.readlines()
            out = subprocess.run(f'cat {out_str}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if len(out.stderr) > 1:
                bin2strings = subprocess.run(f'strings {bin_complete}//* > {out_str}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            else:
                print("Already processed, skip \n")
        
        else:
            bin2strings = subprocess.run(f'strings {bin_complete}//* > {out_str}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    

