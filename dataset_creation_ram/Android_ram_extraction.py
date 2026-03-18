import os
import subprocess
import shutil
import re
import uiautomator2 as u2
import time
import unicodedata
from set_default import set_default_app

fridump_path = "/home/kali/fridump/fridump.py"
bin2img_tool = "/home/kali/binary-to-image/binary2image.py"


def run_frida_server_adb(frida_path):
	cmd_frida_ps = "adb shell ps -A | grep 'frida-server' | awk '{print $2}'"
	frida_pid_out = subprocess.run(cmd_frida_ps, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	frida_pid = frida_pid_out.stdout.strip()
	if frida_pid:
		print(f"Frida server is running on the device with PID: {frida_pid}")
	else:
		#frida-server not running
		cmd_run_frida = ["adb","shell","su 0 sh -c '/data/local/tmp/frida-server &'"]
		run_frida = subprocess.Popen(cmd_run_frida, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
		cmd_frida_ps = "adb shell ps -A | grep 'frida-server' | awk '{print $2}'"
		frida_pid_out = subprocess.run(cmd_frida_ps, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		frida_pid = frida_pid_out.stdout.strip()
		print(f"Started Frida server & running on the device with PID: {frida_pid}")

	
def grant_permissions():
	device = u2.connect()
	device.reset_uiautomator()
	time.sleep(2)
	while True:
		window_detected=False
		
		if device.xpath(f"//*[contains(@text, 'Allow')]").exists:
			if device.xpath(f"//*[contains(@text, 'WHILE USING THE APP')]").exists:
				device(text="WHILE USING THE APP").click()
				window_detected=True
				time.sleep(1)
			elif device.xpath(f"//*[contains(@text, 'ALLOW')]").exists:
				device(text="ALLOW").click()
				window_detected=True
				time.sleep(1)	
			elif device.xpath(f"//*[contains(@text, 'ALWAYS')]").exists:
				device(text="ALWAYS").click()
				window_detected=True
				time.sleep(1)
			elif device.xpath(f"//*[contains(@text, 'always')]").exists:
				device(text="always").click()
				window_detected=True
				time.sleep(1)
			elif device.xpath(f"//*[contains(@text, 'Allow')]").exists:
				device(text="Allow").click()
				window_detected=True
				time.sleep(1)
		elif device.xpath(f"//*[contains(@text, 'full screen')]").exists:
			device(text="GOT IT").click()
			window_detected=True
			time.sleep(1)
		elif device.xpath(f"//*[contains(@text, 'Activating this admin app will allow')]").exists:
			device(text="Activate this device admin app").click()
			window_detected=True
			time.sleep(1)
		elif device.xpath(f"//*[contains(@text, 't responding')]").exists:
			device(text="Close app").click()
			window_detected=True
			time.sleep(1)
		elif device.xpath(f"//*[contains(@text, 'Choose what to allow')]").exists:
			device(text="CONTINUE").click()
			window_detected=True
			time.sleep(1)
		elif device.xpath(f"//*[contains(@text, 'older version')]").exists:
			device(text="OK").click()
			window_detected=True
			time.sleep(1)
		elif device.xpath(f"//*[contains(@text, 'run in background')]").exists:
			device(text="ALLOW").click()
			window_detected=True
			time.sleep(1)
		elif device(textMatches=r".*as your default .* app\?").exists:
			print("Setting the app as the default one \n")
			set_default_app(device)
			window_detected=True
			time.sleep(1)
		elif device.xpath(f"//*[contains(@text, 'keeps stopping')]").exists:
			print("Keeps stopping \n")
			window_detected=False
			time.sleep(1)
		elif device.xpath(f"//*[contains(@text, 'Permission')]").exists:
			device(text="OK").click()
			window_detected = True
			time.sleep(2)
		if not window_detected:
			print("No more windows detected to handle. Exiting loop.")
			break


def install_run_apk(path_apk, pkg_name):
	print(f"Installing apk {path_apk}") 
	install_apk = subprocess.run(f'adb install {path_apk}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	print(f"Running apk {pkg_name}")
	time.sleep(10)
	run_apk = subprocess.run(f'adb shell monkey -p {pkg_name} -c android.intent.category.LAUNCHER 1', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	time.sleep(5)
	#automatic interaction to grant permissions
	grant_permissions()
	time.sleep(5)

		
def automatic_interactions(pkg_name, random_clicks=False):
	 auto_click = subprocess.run(f'adb shell monkey -p {pkg_name} -v 500', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		
def check_process_adb(pkg_name):
	time.sleep(10)
	cmd_pid_pkgname = f"adb shell ps -A | grep '{pkg_name}' | awk '{{print $2}}'"
	process_pid = subprocess.run(cmd_pid_pkgname, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	pid = process_pid.stdout.split("\n")[0]
	print(f"{pkg_name} running with PID {pid}")
	#cat proc/pid/maps | grep with \[stack\] and with dex"""
	return pid
	
def get_memory_address(pid, save_dump_path):
	print(f"Checking relevant addresses for {pid}")
	relevant_addresses = []
	maps_file = subprocess.run(f"adb shell 'su -c \"cat /proc/{pid}/maps > /sdcard/Documents/maps_{pid}.txt\"' && adb pull /sdcard/Documents/maps_{pid}.txt {save_dump_path}/maps.txt", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
	stack_addresses = subprocess.run(f'adb shell \'su -c "cat /proc/{pid}/maps | grep \\"\\[stack\\]\\""\'', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	relevant_addresses.append("0x"+stack_addresses.stdout.split("-")[0])
	heap_addresses = subprocess.run(f'adb shell \'su -c "cat /proc/{pid}/maps | grep \\"\\[heap\\]\\""\'' , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	relevant_addresses.append("0x"+heap_addresses.stdout.split("-")[0])
	dex_addresses = subprocess.run(f'adb shell \'su -c "cat /proc/{pid}/maps | grep \\".dex\\""\'', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	list_dex_addresses = dex_addresses.stdout.splitlines()
	so_addresses = subprocess.run(f'adb shell \'su -c "cat /proc/{pid}/maps | grep \\"\\.so\\""\'', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	list_so_addresses = so_addresses.stdout.splitlines()
	addresses = list_dex_addresses + list_so_addresses
	for address in addresses:
		relevant_addresses.append("0x" + address.split("-")[0])
	print(relevant_addresses)
	return relevant_addresses

def check_process_frida(pid):
	print("Connecting and getting process name from Frida \n")
	get_pid_pname = subprocess.run(f'frida-ps -U | grep {pid}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	print(get_pid_pname.stdout)
	print(get_pid_pname.stderr)
	pattern = r"^\d+\s+(.+?)\s*$"
	match = re.match(pattern, get_pid_pname.stdout)
	process_name = match.group(1)
	print(process_name)
	return process_name

def dump_process(process_name, fridump_path, save_dump_path):
	print(f"Dumping memory of process {process_name}")
	perm = "r--"
	out_dir_dump = save_dump_path + os.sep + perm
	if perm not in os.listdir(os.path.dirname(out_dir_dump)):
		os.makedirs(out_dir_dump)
		run_fridump = subprocess.run(['python3', fridump_path, "-U", process_name, "--perms", perm, "--directory", out_dir_dump])
	return out_dir_dump

def prune_dump(save_dump_path, relevant_addresses, path_relevant_dump="", cancel=False):
	print(f"Saving relevant dumps")
	for file_dump in os.listdir(save_dump_path):
		file_name = file_dump.split("\\")[-1].split("_")[0]
		if not cancel and file_name in relevant_addresses:
			destination_path = os.path.join(path_relevant_dump, file_dump)
			source_path = os.path.join(save_dump_path, file_dump)
			print(destination_path, source_path)
			try:
				shutil.copy2(source_path, destination_path)
				print(f"Copied file: {source_path} to {destination_path}")
			except OSError as e:
				print(f"Error copying file {source_path} to {destination_path}: {e}")
		elif cancel and file_name not in a:
			file_path = os.path.join(save_dump_path, file_name)
			try:
				os.remove(file_path)
				print(f"Deleted file: {file_path}")
			except OSError as e:
				print(f"Error deleting file {file_path}: {e}")

def convert_dump_to_png(src_dump2img, img_dump_path, bin2img_tool, color='RGB'):
	print("Converting dumps into images \n")
	bin2img = subprocess.run(f'python {bin2img_tool} {src_dump2img} {img_dump_path} --analysis_type {color}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	print(bin2img.stdout)
	print(bin2img.stderr)
	print("Saved images from dumps \n")

def sanitize_string(input_string):
	# Define a pattern of allowed characters (e.g., alphanumeric, spaces, and a few safe symbols)
	allowed_characters = re.compile(r'[^a-zA-Z0-9\s_\-.]')
	# Replace all characters not matching the allowed pattern with an empty string
	sanitized = allowed_characters.sub('', input_string)
	if "$" in sanitized:
		sanitized = sanitized.replace("$", "")
	elif "\n" in sanitized:
		sanitized = sanitized.replace("\n", "")
	# Return the cleaned string
	return sanitized	

#def dump_ram_apk(pkg_name, localds_path_apk, fridump_path, dump_path, bin2img_tool):
def dump_ram_apk(pkg_name, apk_path, dump_path):
	#run frida on android
	frida_android_path = "/data/local/tmp/"
	run_frida_server_adb(frida_android_path)
	#install and run the target app
	install_run_apk(apk_path, pkg_name)
	#get the pid of the running target app
	pid_pkg = None
	while pid_pkg is None:
		pid_pkg = check_process_adb(pkg_name)
		
		time.sleep(2)
	#get process name of running target app
	process_name = check_process_frida(pid_pkg)
	#make some automatic interactions
	random_clicks = False
	automatic_interactions(random_clicks, pkg_name)
	#define directory name to save the process
	apk_filename = os.path.basename(apk_path)
	dir_name_process = apk_filename + "_" + pkg_name + "_" + process_name.replace(" ", "-") + "_" + pid_pkg
	dir_name_process = sanitize_string(dir_name_process)
	#define the path where to save dumps for the target process
	save_dump_path = dump_path + os.sep + dir_name_process
	#save_dump_path = save_dump_path.replace("'", "'\\''")
	print("@@@@@@@@@@@@@@@@@@@@@@@@@@@ ", save_dump_path)
	with open("path/save/dump/logfile.txt", "w") as f:
		f.write(save_dump_path)
	#create a dump directory for the process if not present
	if dir_name_process not in os.listdir(os.path.dirname(save_dump_path)):
		os.makedirs(save_dump_path)
	#get the interesting addresses to be examined in the memory of the target process
	relevant_addresses = get_memory_address(pid_pkg, save_dump_path)
	
	#dump the memory at the interesting addresses of the target process
	out_dir_dump = dump_process(process_name, fridump_path, save_dump_path)
	#define if want to copy the target addresses dumps in another directory without deleting the whole dump
	copy = True
	#create a directory to save the dump of the relevant addresses and if copy False delete the others
	dir_name_save_reladdr = "relevant_addresses"
	if copy:
		#define a path where to save the dumps of the target address
		path_relevant_dump = save_dump_path + os.sep + dir_name_save_reladdr
		#create directory if not present
		if dir_name_save_reladdr not in os.listdir(os.path.dirname(path_relevant_dump)):
			os.makedirs(path_relevant_dump)
		src_dump2img = path_relevant_dump
	else:
		#if not set to copy, the other files have been deleted and the directory has default fridump name (src_dump2img)
		path_relevant_dump = ""
		src_dump2img = save_dump_path + "/r--/"
	#select only interesting dumps
	prune_dump(out_dir_dump, relevant_addresses, path_relevant_dump)
	#set the path where to save the images of the dumps
	dir_name_save_img = "images"
	img_dump_path = save_dump_path + os.sep + dir_name_save_img
	if dir_name_save_img not in os.listdir(os.path.dirname(img_dump_path)):
		os.makedirs(img_dump_path)
	#set the color, default set to RGB but can be changed to grayscale
	color = "RGB"
	#convert the dump to png images
	convert_dump_to_png(src_dump2img, img_dump_path, bin2img_tool, color)


