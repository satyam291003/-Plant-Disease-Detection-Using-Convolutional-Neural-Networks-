# preprocess_data.py
import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(data_dir, img_size=(128, 128)):
    images = []
    labels = []
    label_map = {}  # To store label mapping

    # Loop through all subfolders (diseases and healthy folders)
    for idx, folder in enumerate(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        label_map[idx] = folder  # Map index to class name (e.g., 0: 'Tomato_healthy')
        
        # Loop through each image in the folder
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)  # Resize image to 128x128
                images.append(img)
                labels.append(idx)  # Use folder index as label
    
    # Convert lists to NumPy arrays
    X = np.array(images, dtype='float32') / 255.0  # Normalize pixel values
    y = to_categorical(np.array(labels))  # Convert labels to one-hot encoding
    
    return X, y, label_map

# Example usage
if __name__ == "__main__":
    data_dir = "E:/leafproject/PlantVillage"  # Path to your dataset
    X, y, label_map = load_and_preprocess_data(data_dir)
    
    # Save preprocessed data
    np.save("data/X.npy", X)
    np.save("data/y.npy", y)
    
    # Save label map
    with open("data/label_map.txt", "w") as f:
        for idx, label in label_map.items():
            f.write(f"{idx}:{label}\n")
    
    print("Data Preprocessing Complete!")
    print(f"Classes: {label_map}")
