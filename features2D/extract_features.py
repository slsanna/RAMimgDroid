import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy
from sklearn.model_selection import train_test_split
import cv2

# Feature extraction function
def extract_features(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # Color Histograms (RGB + HSV)
    rgb_hist = np.concatenate([np.histogram(img_np[:, :, i], bins=32, range=(0, 256))[0] for i in range(3)])
    hsv_img = rgb2hsv(img_np / 255.0)
    hsv_hist = np.concatenate([np.histogram(hsv_img[:, :, i], bins=32, range=(0, 1))[0] for i in range(3)])

    # Color Moments (mean, std, skewness per channel)
    color_moments = []
    for i in range(3):
        channel = img_np[:, :, i].flatten()
        color_moments += [np.mean(channel), np.std(channel), skew(channel)]

    # GLCM Texture Features
    gray = rgb2gray(img_np)
    gray_uint8 = (gray * 255).astype('uint8')
    glcm = graycomatrix(gray_uint8, distances=[1], angles=[0], symmetric=True, normed=True)
    glcm_features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ]

    # Local Binary Pattern (LBP)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 11), range=(0, 10), density=True)

    # HOG (Shape) - resize only for HOG to ensure consistent feature length
    hog_resized = cv2.resize((gray * 255).astype('uint8'), (128, 128))
    hog_features, _ = hog(hog_resized, orientations=9, pixels_per_cell=(16, 16),
                          cells_per_block=(2, 2), visualize=True, feature_vector=True)

    # Edge Detection (Sobel)
    edges = sobel(gray)
    edge_stats = [np.mean(edges), np.std(edges)]

    # Hu Moments (Shape Descriptors)
    moments = cv2.moments(gray_uint8)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Statistical Features
    stat_features = [
        np.mean(gray), np.std(gray), skew(gray.flatten()), kurtosis(gray.flatten()),
        entropy(np.histogram(gray, bins=256, range=(0, 1), density=True)[0] + 1e-10)
    ]

    # Frequency Domain (FFT)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1)
    freq_features = magnitude_spectrum.flatten()[:100]

    # Combine all features
    features = {
        'rgb_hist': rgb_hist.tolist(),
        'hsv_hist': hsv_hist.tolist(),
        'color_moments': color_moments,
        'glcm_features': glcm_features,
        'lbp_hist': lbp_hist.tolist(),
        'hog_features': hog_features.tolist(),
        'edge_stats': edge_stats,
        'hu_moments': hu_moments.tolist(),
        'stat_features': stat_features,
        'freq_features': freq_features.tolist()
    }

    return features

# CSV loading function
def load_feature_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['label', 'file_path'])
    y = df['label']
    return X, y, df

# Dataset preparation
root_dir = "/mnt/malware_ram/Android"
classes = {'benign_dumps': 0, 'dumps_dataset': 1}
output_csv = 'image_features_dataset.csv'

# Load existing data if any
if os.path.exists(output_csv):
    existing_df = pd.read_csv(output_csv)
    processed_files = set(existing_df['file_path'].tolist())
    expected_columns = existing_df.shape[1]
else:
    existing_df = pd.DataFrame()
    processed_files = set()
    expected_columns = None

data = []

for class_name, label in classes.items():
    class_dir = os.path.join(root_dir, class_name)
    if not os.path.exists(class_dir): continue
    for apk in tqdm(os.listdir(class_dir), desc=f"Extracting features from {class_name}"):
        class_path = os.path.join(class_dir, apk, 'images', 'complete', 'RGB')
        if not os.path.exists(class_path): continue
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if img_path in processed_files:
                continue
            try:
                features = extract_features(img_path)
                row = {**features, 'label': label, 'file_path': img_path}
                row_df = pd.DataFrame([{k: v for k, v in row.items()}])

                # Define headers if new file
                if existing_df.empty and expected_columns is None:
                    # Use dictionary-based DataFrame, column headers will be automatically inferred
                    expected_columns = row_df.shape[1]
                    # Columns are inferred from the dictionary keys; no manual assignment needed
                    row_df.to_csv(output_csv, index=False)
                    existing_df = row_df
                    processed_files.add(img_path)
                else:
                    if row_df.shape[1] == expected_columns:
                        row_df.columns = existing_df.columns
                        row_df.to_csv(output_csv, mode='a', header=False, index=False)
                        processed_files.add(img_path)
                    else:
                        print(f"Skipping {img_path}: column mismatch {row_df.shape[1]} vs {expected_columns}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

print("Feature extraction completed and saved.")
