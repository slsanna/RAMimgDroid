import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
import torch.nn as nn
from gensim.models import Word2Vec
from tqdm import tqdm
import pickle


def create_dataset(benign_dir, malicious_dir, save_dir):
    texts = []
    labels = []
    # Read text data and labels
    for directory, label in [(benign_dir, 0), (malicious_dir, 1)]:
        for filename in tqdm(os.listdir(directory), desc=f"Processing {'benign' if label == 0 else 'malicious'} files"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                texts.append(file.read())
            labels.append(label)
    with open(save_dir+"labels.pkl", "wb") as f:
        pickle.dump(labels, f)
    with open(save_dir+"text.pkl", "wb") as f:
        pickle.dump(texts, f)

def load_dataset(save_dir):
    with open(save_dir+"labels.pkl", "rb") as f:
        labels = pickle.load(f)
    with open(save_dir+"texts.pkl", "rb") as f:
        texts = pickle.load(f)
    return labels, texts

def tokenize(save_dir, texts):
    print("Tokenizing \n")
    # Tokenize and create Word2Vec model
    tokenized_texts = [text.split() for text in texts]  # Simple whitespace tokenization
    word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
    with open(save_dir+"word2vec_model.pkl", "wb") as f:
        pickle.dump(word2vec_model, f)
    with open(save_dir+"tokens.pkl", "wb") as f:
        pickle.dump(tokenized_texts, f)
    return word2vec_model, tokenized_texts

def build_vocabulary(save_dir, word2vec_model):
    print("Build vocabulary \n")
    # Build vocabulary mapping (word to index) and inverse mapping
    vocab = {word: idx for idx, word in enumerate(word2vec_model.wv.index_to_key)}
    vocab_size = len(vocab)
    with open(save_dir+"vocabulary.pkl", "wb") as f:
        pickle.dump(vocab, f)
    return vocab

def embed_matrix(vocab_size, word2vec_model, save_dir):
    print("Create embedding matrix \n")
    # Create an embedding matrix
    embedding_matrix = np.zeros((vocab_size, 100))
    for i, word in enumerate(word2vec_model.wv.index_to_key):
        embedding_vector = word2vec_model.wv[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    with open(save_dir+"embedding_matrix.pkl", "wb") as f:
        pickle.dump(embedding_matrix, f)
    return embedding_matrix

def create_vectors(tokenized_texts, vocab, save_dir):
    print("Create vectors \n")
    # Convert texts to fixed size padded vector sequences
    max_len = 100  # or any other fixed length you want
    vectorized_data = [[vocab[word] for word in doc if word in vocab] for doc in tokenized_texts]
    padded_data = np.zeros((len(vectorized_data), max_len), dtype=int)
    for i, doc in enumerate(vectorized_data):
        padded_data[i, :len(doc)] = doc[:max_len]
    with open(save_dir+"padded_data.pkl", "rb") as f:
        pickle.dump(padded_data, f)
    return padded_data

def create_tensors(padded_data, labels, save_dir):
    print("Creating tensors \n")
    data = torch.tensor(padded_data, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)
    with open(save_dir+"tensors_data.pkl", "wb") as f:
        pickle.dump(data, f)
    with open(save_dir+"tensors_label.pkl", "wb") as f:
        pickle.dump(labels, f)
    return data, labels


# Define a PyTorch model that uses the Word2Vec embeddings
class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = nn.LSTM(100, 128, batch_first=True)
        self.linear = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        output = self.linear(hidden.squeeze(0))
        return output

def training(data, labels, save_dir):
    # Model and training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_count = int(0.8 * len(data))
    test_count = len(data) - train_count
    train_data, test_data = random_split(TensorDataset(data, labels), [train_count, test_count])
    model = TextClassifier().to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters())
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Training loop
    num_epochs = 10
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    torch.save(model.state_dict(), save_dir+"model.pth")
    with open(save_dir+"test_loader.pkl", "wb") as f:
        pickle.dump(test_loader, f)
    return model, test_loader


save_dir = "/home/ssanna/Desktop/malware_ram/Android/"

# Load your data
benign_dir = "/mnt/malware_ram/Android/Benign_App_Memdumps"
malicious_dir = "/mnt/malware_ram/Android/Malicious_App_Memdumps"

if "labels.pkl" in os.listdir(save_dir) and "texts.pkl" in os.listdir(save_dir):
    print("Loading dataset \n")
    labels, texts = load_dataset(save_dir)
else:
    create_dataset(benign_dir, malicious_dir, save_dir)

if "word2vec_model.pkl" in os.listdir(save_dir) and "tokens.pkl" in os.listdir(save_dir):
    print("Loading tokens \n")
    with open(save_dir+"word2vec_model.pkl", "rb") as f:
        word2vec_model = pickle.load(f)
    with open(save_dir+"tokens.pkl", "rb") as f:
        tokenized_texts = pickle.load(f)
else:
    word2vec_model, tokenized_texts = tokenize(save_dir, texts)

if "vocabulary.pkl" in os.listdir(save_dir):
    print("Loading vocabulary \n")
    with open("vocabulary.pkl", "rb") as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
else:
    vocab = build_vocabulary(save_dir, word2vec_model)
    vocab_size = len(vocab)

if "embedding_matrix.pkl" in os.listdir(save_dir):
    print("Loading embedding matrix \n")
    with open(save_dir+"embedding_matrix.pkl", "rb") as f:
        embedding_matrix = pickle.load(f)
else:
    embedding_matrix = embed_matrix(vocab_size, word2vec_model, save_dir)


if "padded_data.pkl" in os.listdir(save_dir):
    print("Loading padded data \n")
    with open(save_dir+"padded_data.pkl", "rb") as f:
        padded_data = pickle.load(f)
else:
    padded_data = create_vectors(tokenized_texts, vocab, save_dir)


if "data_tensors.pkl" in os.listdir(save_dir) and "labels_tensors.pkl" in os.listdir(save_dir):
    with open(save_dir+"data_tensors.pkl", "rb") as f:
        data = pickle.load(f)
    with open(save_dir+"labels_tensors.pkl", "rb") as f:
        labels = pickle.load(f)
else:
    data, labels = create_tensors(padded_data, labels, save_dir)


if "model.pth" in os.listdir(save_dir) and "test_loader.pkl" in os.listdir(save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassifier().to(device)
    model.load_state_dict(torch.load(save_dir+"model.pth"))
    with open(save_dir+"test_loader.pkl", "rb") as f:
       test_loader =  pickle.load(f)

else:
    model, test_loader = training(data, labels, save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing \n")
    # Testing
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += targets.size(0)
            correct += (predicted.squeeze() == targets).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')
