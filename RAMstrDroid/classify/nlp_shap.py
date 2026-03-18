from tqdm import tqdm
import os
import json
import shap
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------
# Dataset Loader
# ---------------------------
def load_dataset(root_dir, class_dirs, class_filter=None):
    texts = []
    labels = []

    for label, class_dir in enumerate(class_dirs):
        class_dir_path = os.path.join(root_dir, class_dir)
        if not os.path.exists(class_dir_path):
            continue

        for apk in tqdm(os.listdir(class_dir_path), desc=f"Processing {class_dir} APKs", unit="apk"):
            class_path = os.path.join(class_dir_path, apk, 'strings')
            str_path = os.path.join(class_path, 'strings.txt')

            if not os.path.exists(str_path):
                continue

            with open(str_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                combined_text = " ".join(line.strip() for line in lines if line.strip())
                texts.append(combined_text)
                labels.append(label)

    return texts, labels

# ---------------------------
# Load and Split Data
# ---------------------------
root_dir = "dataset_path"
class_dirs = ['benign_dumps', 'malware_dumps']
class_names = ['Benign', 'Malicious']

texts, labels = load_dataset(root_dir, class_dirs)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# ---------------------------
# Train Model
# ---------------------------
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=500)
clf.fit(X_train_vec, y_train)

# ---------------------------
# Evaluate Model
# ---------------------------
y_pred = clf.predict(X_test_vec)
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))

# ---------------------------
# SHAP Explainability - All Samples
# ---------------------------
print("\n Generating SHAP explanations for all test samples...\n")

explainer = shap.LinearExplainer(clf, X_train_vec, feature_perturbation="interventional")
feature_names = vectorizer.get_feature_names_out()

shap_values = explainer.shap_values(X_test_vec)

explanations = []

for i in range(X_test_vec.shape[0]):
    true_label = y_test[i]
    predicted_label = y_pred[i]

    sample_expl = {
        'index': i,
        'true_label': class_names[true_label],
        'predicted_label': class_names[predicted_label],
        'features': []
    }

    # Get non-zero shap values for top features
    row_values = shap_values[predicted_label][i].toarray().flatten()
    top_indices = row_values.argsort()[::-1][:10]

    for idx in top_indices:
        word = feature_names[idx]
        value = row_values[idx]
        if abs(value) > 0:
            sample_expl['features'].append((word, float(value)))

    explanations.append(sample_expl)

    print(f"Sample #{i}")
    print(f"True Label: {class_names[true_label]} | Predicted: {class_names[predicted_label]}")
    for feat, score in sample_expl['features']:
        print(f"  {feat}: {score:.4f}")
    print("-" * 50)

# ---------------------------
# Save to JSON and CSV
# ---------------------------
with open("shap_explanations.json", "w") as f:
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

df_exp = pd.DataFrame(rows)
df_exp.to_csv("shap_explanations.csv", index=False)

