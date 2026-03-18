import uiautomator2 as u2
import unicodedata
#https://github.com/openatx/android-uiautomator-server/releases/tag/2.4.0

# Function to remove non-printable Unicode characters
def clean_text(text):
	return ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')

def set_default_app(device):
	
	# Step 1: Identify the app name from the generalized dialog title
	title_element = device(textMatches=r".*as your default .* app\?")
	print(title_element)
	if title_element.exists:
		# Extract the app name and clean it
		dialog_text = title_element.get_text()
		cleaned_text = clean_text(dialog_text)
		app_name = cleaned_text.split("Set ")[1].split(" as your default")[0]
		print(f"App name extracted and cleaned: {app_name}")
		
		# Step 2: Look for the app name in the list of options
		app_option_found = False
		app_options = device(className="android.widget.LinearLayout")
		for app_option in app_options:
			app_name_element = app_option.child(className="android.widget.TextView")
			if app_name_element.exists:
				app_name_text = clean_text(app_name_element.get_text())
				if app_name_text == app_name:
					print(f"Matching app option found: {app_name_text}")
					# Find and click the radio button by simulating a click on the parent layout
					radio_button = app_option.child(className="android.widget.RadioButton")
					if radio_button.exists:
						# Try clicking on the radio button's parent layout if direct clicking doesn't work
						app_option.click()  # Click on the layout containing the radio button
						print(f"Clicked on the layout containing the radio button for: {app_name_text}")
						app_option_found = True
						break
		if not app_option_found:
			print(f"App '{app_name}' was not found in the list.")
			exit(1)
			
		# Step 3: Click on "SET AS DEFAULT" if it's enabled
		set_default_button = device(resourceId="android:id/button1")
		if set_default_button.exists and set_default_button.info['enabled']:
			set_default_button.click()
			print("Clicked 'SET AS DEFAULT'.")
		else:
			print("'SET AS DEFAULT' button is not enabled.")
	else:
		print("The dialog is not displayed.")

