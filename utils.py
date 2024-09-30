import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

def load_dataset(image_dir, labels_csv, img_size=(64, 64)):
    # Load CSV file with labels
    df = pd.read_csv(labels_csv)
    
    images = []
    labels = []
    
    for index, row in df.iterrows():
        img_path = os.path.join(image_dir, row['image_filename'])
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img) / 255.0  # Normalize the image
        images.append(img_array)
        labels.append(row['label'])

    # Convert to numpy arrays
    X = np.array(images)
    y = to_categorical(np.array(labels))  # One-hot encode the labels

    return X, y
