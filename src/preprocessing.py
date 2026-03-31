import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def preprocess_data(path_to_data = "data/Images", img_size = 64, n_class = 10, n_samples_in_class = 100):
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / path_to_data

    images = []
    labels = []

    for breed in os.listdir(data_dir)[:n_class]:
        breed_path = os.path.join(data_dir, breed)

        if not os.path.isdir(breed_path):
            continue

        for img_name in os.listdir(breed_path)[:n_samples_in_class]:
            img_path = os.path.join(breed_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (img_size, img_size))

            images.append(img)
            labels.append(breed)

    X = np.array(images)
    y = np.array(labels)

    X = X.reshape(X.shape[0], -1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y



