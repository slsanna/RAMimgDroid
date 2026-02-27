from tqdm import tqdm
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_dataset(root_dir, class_dirs, class_filter=None):
    texts = []
    labels = []

    for label, class_dir in enumerate(class_dirs):

        class_dir_path = os.path.join(root_dir, class_dir)
        if not os.path.exists(class_dir_path):
            continue

        for apk in tqdm(os.listdir(class_dir_path), desc=f"Processing {class_dir} APKs", unit="apk"):
            if apk != "deed5f52de0c3318c44c3312ee2e636a236d1d13f28c64f8242713b3e7b8a4f2.apk_vbkoxh.cswnpr_angry_birds_seasons_2882":
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


root_dir = "/mnt/malware_ram/Android"
class_dirs = ['benign_dumps', 'dumps_dataset']

texts, labels = load_dataset(root_dir, class_dirs)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Define pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', LogisticRegression(max_iter=500))
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
