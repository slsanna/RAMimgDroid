from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import torch
import os

# Use CybersecurityBERT
model_name = "CybersecurityBERT/cybersecurity-bert"  # Change to a real model if needed

# Auto-select device: GPU if available, else CPU
device = 0 if torch.cuda.is_available() else -1

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Optional: Use half precision if on GPU for faster inference
if device == 0:
    model = model.half()

# Create classifier pipeline with batching and truncation
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    device=device
)

# Function to chunk long text
def chunk_text(text, chunk_size=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return [tokenizer.decode(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]

root_dir = "dataset/path"
class_dirs = ['benign_dumps', 'malware/dumps']

for label, class_dir in enumerate(class_dirs):
    class_dir_path = os.path.join(root_dir, class_dir)
    if not os.path.exists(class_dir_path):
        continue

    for apk in tqdm(os.listdir(class_dir_path), desc=f"Processing {class_dir} APKs", unit="apk"):
    

        str_path = os.path.join(class_path, 'strings.txt')
        if not os.path.exists(str_path):
            continue

        with open(str_path, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        full_text = " ".join(lines)
        chunks = chunk_text(full_text)

        # Batch classify all chunks (MUCH faster)
        if not chunks:
            continue

        predictions = classifier(chunks, batch_size=8)
        top_result = max(predictions, key=lambda x: x['score'])

        print(f"Class: {class_dir}")
        print(f"APK: {apk}")
        print(f"Prediction: {top_result['label']}, Confidence: {top_result['score']:.2f}")
        print("-" * 50)
