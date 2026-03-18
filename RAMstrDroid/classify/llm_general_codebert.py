from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from tqdm import tqdm
import os

# Load a pre-trained model (can be changed to one fine-tuned for security)
model_name = "microsoft/graphcodebert-base"  # or a security-focused model if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Force CPU usage by setting device=-1
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True, device=-1)

# Function to chunk long text
def chunk_text(text, chunk_size=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return [tokenizer.decode(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]

root_dir = "/mnt/malware_ram"
class_dirs = ['benign_dumps', 'dataset_dumps']
        
for label, class_dir in enumerate(class_dirs):
    class_dir_path = os.path.join(root_dir, class_dir)
    if not os.path.exists(class_dir_path):
        continue

    for apk in tqdm(os.listdir(class_dir_path), desc=f"Processing {class_dir} APKs", unit="apk"):
    
        str_path = os.path.join(class_path, 'strings.txt')
        if not os.path.exists(str_path):
            continue

        with open(str_path) as f:
            lines = f.readlines()
        
        full_text = " ".join(lines)
        chunks = chunk_text(full_text)

        # Collect predictions from each chunk
        predictions = []
        for chunk in chunks:
            result = classifier(chunk)[0]
            predictions.append(result)

        # Aggregate predictions (e.g., majority voting or max confidence)
        top_result = max(predictions, key=lambda x: x['score'])

        print(f"Class: {class_dir}")
        print(f"APK: {apk}")
        print(f"Prediction: {top_result['label']}, Confidence: {top_result['score']:.2f}")
        print("-" * 50)
