import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data" / "Images"

images = []
labels = []

IMG_SIZE = 64

for breed in os.listdir(data_dir)[:10]:
    breed_path = os.path.join(data_dir, breed)

    if not os.path.isdir(breed_path):
        continue

    for img_name in os.listdir(breed_path)[:100]:
        img_path = os.path.join(breed_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        images.append(img)
        labels.append(breed)

X = np.array(images)
y = np.array(labels)

X = X.reshape(X.shape[0], -1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)