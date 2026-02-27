from tqdm import tqdm
import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lime.lime_text import LimeTextExplainer

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
root_dir = "/mnt/malware_ram/Android"
class_dirs = ['benign_dumps', 'dumps_dataset']
class_names = ['Benign', 'Malicious']

texts, labels = load_dataset(root_dir, class_dirs)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# ---------------------------
# Train Model
# ---------------------------
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', LogisticRegression(max_iter=500))
])

model.fit(X_train, y_train)

# ---------------------------
# Evaluate Model
# ---------------------------
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))

# ---------------------------
# LIME Explainability - All Samples
# ---------------------------
print("\n🔍 Generating LIME explanations for all test samples...\n")
explainer = LimeTextExplainer(class_names=class_names)
explanations = []

for i, sample_text in enumerate(X_test):
    true_label = y_test[i]
    predicted_label = model.predict([sample_text])[0]

    exp = explainer.explain_instance(sample_text, model.predict_proba, num_features=5)

    explanation_data = {
        'index': i,
        'true_label': class_names[true_label],
        'predicted_label': class_names[predicted_label],
        'features': exp.as_list()
    }
    explanations.append(explanation_data)

    # Print explanation to terminal
    print(f"Sample #{i}")
    print(f"True Label: {class_names[true_label]} | Predicted: {class_names[predicted_label]}")
    for feat, score in exp.as_list():
        print(f"  {feat}: {score:.4f}")
    print("-" * 50)

# ---------------------------
# Save to JSON and CSV
# ---------------------------
with open("lime_explanations.json", "w") as f:
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
#df_exp.to_csv("lime_explanations.csv", index=False)


