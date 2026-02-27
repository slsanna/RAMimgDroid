import os
import json
import joblib
import pandas as pd
from tqdm import tqdm
from lime.lime_text import LimeTextExplainer
from sklearn.model_selection import train_test_split

# ---------------------------
# Configurations
# ---------------------------
MODEL_PATH = "stack_strings.pkl"
ROOT_DIR = "/mnt/malware_ram/Android"
CLASS_DIRS = ['benign_dumps', 'dumps_dataset']
CLASS_NAMES = ['Benign', 'Malicious']
OUTPUT_JSON = "lime_explanations_stack.json"
OUTPUT_CSV = "lime_explanations_stack.csv"

# ---------------------------
# Dataset Loader
# ---------------------------
def load_dataset(root_dir, class_dirs):
    texts = []
    labels = []

    for label, class_dir in enumerate(class_dirs):
        class_dir_path = os.path.join(root_dir, class_dir)
        if not os.path.exists(class_dir_path):
            continue

        for apk in tqdm(os.listdir(class_dir_path), desc=f"Processing {class_dir} APKs", unit="apk"):
            class_path = os.path.join(class_dir_path, apk, 'strings')
            str_path = os.path.join(class_path, 'strings_stack.txt')

            if not os.path.exists(str_path):
                continue

            with open(str_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                combined_text = " ".join(line.strip() for line in lines if line.strip())
                texts.append(combined_text)
                labels.append(label)

    return texts, labels

# ---------------------------
# Load Model and Data
# ---------------------------
print("📦 Loading model...")
model = joblib.load(MODEL_PATH)

print("📂 Loading dataset...")
texts, labels = load_dataset(ROOT_DIR, CLASS_DIRS)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)

# ---------------------------
# LIME Explainability
# ---------------------------
print("🔍 Generating LIME explanations...")
explainer = LimeTextExplainer(class_names=CLASS_NAMES)
explanations = []

for i, sample_text in enumerate(X_test):
    true_label = y_test[i]
    predicted_label = y_pred[i]

    exp = explainer.explain_instance(sample_text, model.predict_proba, num_features=5)

    explanation = {
        'index': i,
        'true_label': CLASS_NAMES[true_label],
        'predicted_label': CLASS_NAMES[predicted_label],
        'features': exp.as_list()
    }
    explanations.append(explanation)

    # Optional: Print to console
    print(f"Sample #{i} — True: {explanation['true_label']} | Predicted: {explanation['predicted_label']}")
    for feat, val in explanation['features']:
        print(f"  {feat}: {val:.4f}")
    print("-" * 50)

# ---------------------------
# Save Outputs
# ---------------------------
print(f"💾 Saving LIME explanations to {OUTPUT_JSON} and {OUTPUT_CSV}...")

with open(OUTPUT_JSON, "w") as f:
    json.dump(explanations, f, indent=2)

rows = []
for item in explanations:
    row = {
        'index': item['index'],
        'true_label': item['true_label'],
        'predicted_label': item['predicted_label']
    }
    for feat, score in item['features']:
        row[feat] = score
    rows.append(row)

#df_exp = pd.DataFrame(rows)
#df_exp.to_csv(OUTPUT_CSV, index=False)

print("✅ LIME explanation script completed.")
