import os
import subprocess
import shutil
from Android_ram_extraction import dump_ram_apk
import time

def interact_emulator(emulator_device_name, input_power_command):    
    # Get the list of emulators and find the target emulator by name
    list_ed = subprocess.run(f'gmtool admin list | grep "{emulator_device_name}"', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if list_ed.returncode != 0 or not list_ed.stdout:
        print(f"Error finding emulator: {list_ed.stderr}")
        return
    
    print("List command output:", list_ed.stdout)
    
    try:
        target_uuid = list_ed.stdout.split("|")[2].strip()
        target_state = list_ed.stdout.split("|")[0].strip()
    except IndexError:
        print("Error extracting UUID or state. Ensure the output format is correct.")
        return
    
    # Define the power command based on input
    if input_power_command == "start":
        power_command = "start"
        state_mode = "On"
    elif input_power_command == "factoryreset":
        power_command = "factoryreset"
        state_mode = "Off"  # The emulator should be stopped before factory reset
    else:
        power_command = "stop"
        state_mode = "Off"
    
    # Send the appropriate power command if needed
    if state_mode not in target_state or input_power_command == "factoryreset":
        print(f"Sending {power_command} command to emulator with UUID {target_uuid}")
        
        if power_command == "factoryreset":
            # Ensure the emulator is stopped before performing factory reset
            stop_emulator = subprocess.run(f"gmtool admin stop {target_uuid}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            time.sleep(5)  # Short delay to ensure the emulator has stopped
            
            # Send the factory reset command
            factory_reset = subprocess.run(f"gmtool admin factoryreset {target_uuid}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if factory_reset.returncode == 0:
                print(f"Factory reset successful for emulator {emulator_device_name}")
            else:
                print(f"Factory reset failed for emulator {emulator_device_name}: {factory_reset.stderr}")
        else:
            # Send start or stop command as per input
            os.system(f"gmtool admin {power_command} {target_uuid}")
            
    else:
        print(f"Emulator {emulator_device_name} is already in the desired state: {state_mode}")
		

def get_apkname(apk_path):
	print("APKPATH ", apk_path)
	manifest_info = subprocess.run(f'aapt d badging "{apk_path}"', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	pkg_name = manifest_info.stdout.split("package: name='")[1].split("' versionCode=")[0]
	print(pkg_name)
	
	return pkg_name

def is_device_ready():
	while True:
		boot_completed = subprocess.run(f'adb shell getprop sys.boot_completed', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		if boot_completed.stdout.strip() == "1":
			print("Emulator boot completed\n")
			break
		else:
			print("Waiting booting ... \n")
			time.sleep(10)
               
     


def run(apk_name, benign, remote_apk, remote_saved_dumps):
     
	emulator_device_name = "Device Name"
	dataset_app = "apk_path"
	frida_server_local_path = "path/to/frida-server"
	frida_server_android_path = "/data/local/tmp/frida-server"
	dump_path = "path/to/save/dumps/"

	power_command = "stop"
	interact_emulator(emulator_device_name, power_command)
	time.sleep(5)
	power_command = "factoryreset"
	interact_emulator(emulator_device_name, power_command)
	
	print("Running Analysis for APK ----- ", apk_name)
	local_apk_path = dataset_app + os.sep + apk_name
	
	apk_name = apk_name.split(".apk")[0]
	
	power_command = "start"
	interact_emulator(emulator_device_name, power_command)
	
	print("Installing Frida Android \n")
	push_frida = subprocess.check_call(f'adb push {frida_server_local_path} {frida_server_android_path}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	print(push_frida)
	chmod_frida = subprocess.run(f'adb shell "su -c \'chmod +x {frida_server_android_path}\'"', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	
	print("Installing uiautomator Android \n")
	uiautomator_server_path = "path/to/app-uiautomator.apk"
	install_uiautomator = subprocess.run(f'adb install {uiautomator_server_path}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	run_uiautomator = subprocess.run(f'adb shell monkey -p com.github.uiautomator -c android.intent.category.LAUNCHER 1', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	time.sleep(10)
	
	pkg_name = get_apkname(local_apk_path)
	print(pkg_name)
	time.sleep(5)
	#push and start uiautomator apk server
	dump_ram_apk(pkg_name, local_apk_path, dump_path)
	time.sleep(10)
	power_command = "stop"
	interact_emulator(emulator_device_name, power_command)
	time.sleep(10)
	power_command = "factoryreset"
	interact_emulator(emulator_device_name, power_command)
	time.sleep(10)
	power_command = "factoryreset"
	interact_emulator(emulator_device_name, power_command)
	saved_ram = dump_path + os.sep + os.listdir(dump_path)[0]
	
	time.sleep(2)
		
