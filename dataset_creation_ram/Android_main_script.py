import os
import subprocess
from Android_genymotion import run, interact_emulator
import shutil
from tqdm import tqdm
import time
import json
import sys

def get_common_elements(list1, list2=None):
    if list2 is None or not list2:  # Check if the second list is missing or empty
        list2 = list1  # Take all elements from list1 (no filtering)
    
    # Find intersection
    intersection = list(set(list1) & set(list2))
    return intersection

type_dataset = sys.argv[1] #benign or malware


genymotion_path = "path/to/genymotion"
emulator_device_name = "Device Name"
dataset_app = "path/to/apk/"
skip_apk = "filename/of/apk/to/be/skipped/for/problems/reasons"
dumps_ds = "path/to/already/saved/dumps"
for i, apk_name in enumerate(tqdm(dataset_app, desc="Processing APKs")):
	
	
	#if i < 1000 and apk_name in list_apk and not any (apk_name in filename for filename in dumpsds) and not (apk_name in skip_apk): #check if not analysed (check dir name)
	if apk_name in dataset_app and not any (apk_name in filename for filename in dumps_ds) and not (apk_name in skip_apk):
		print("ANALYZING ........ ", apk_name)
		try:
			print("Start... stop and factoryreset \n")
			power_command = "stop"
			interact_emulator(emulator_device_name, power_command)
			time.sleep(5)
			power_command = "factoryreset"
			interact_emulator(emulator_device_name, power_command)
			run(apk_name, type_dataset, apk_name, dumps_ds)
		except Exception as e:
			print(e)
			with open(skip_apk, "a", encoding='utf-8') as f:
				f.write(apk_name)
				f.write("\n")
			print("Something went wrong, sending stop and factory reset \n")
			power_command = "stop"
			interact_emulator(emulator_device_name, power_command)
			time.sleep(5)
			power_command = "factoryreset"
			interact_emulator(emulator_device_name, power_command)
			local_apk_path = dataset_app + os.sep + apk_name
			os.remove(local_apk_path)
			continue
