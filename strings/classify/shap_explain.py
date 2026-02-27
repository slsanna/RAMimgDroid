import os
import json
import joblib
import shap
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ---------------------------
# Configurations
# ---------------------------
MODEL_PATH = "stack_strings.pkl"
ROOT_DIR = "/mnt/malware_ram/Android"
CLASS_DIRS = ['benign_dumps', 'dumps_dataset']
CLASS_NAMES = ['Benign', 'Malicious']
OUTPUT_JSON = "shap_explanations_stack.json"
OUTPUT_CSV = "shap_explanations_stack.csv"

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
tfidf = model.named_steps['tfidf']
clf = model.named_steps['clf']

print("📂 Loading dataset...")
texts, labels = load_dataset(ROOT_DIR, CLASS_DIRS)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
y_pred = model.predict(X_test)

# ---------------------------
# SHAP Explainability
# ---------------------------
print("🔍 Generating SHAP explanations...")
explainer = shap.Explainer(clf, X_train_tfidf, feature_names=tfidf.get_feature_names_out())
shap_values = explainer(X_test_tfidf)

explanations = []
for i in range(len(X_test)):
    row = shap_values[i]
    top_indices = np.argsort(-np.abs(row.values))[:5]

    explanation = {
        'index': i,
        'true_label': CLASS_NAMES[y_test[i]],
        'predicted_label': CLASS_NAMES[y_pred[i]],
        'features': [(row.feature_names[idx], float(row.values[idx])) for idx in top_indices]
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
print(f"💾 Saving SHAP explanations to {OUTPUT_JSON} and {OUTPUT_CSV}...")

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

print("✅ SHAP explanation script completed.")
