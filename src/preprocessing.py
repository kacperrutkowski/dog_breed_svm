import os
import cv2
import numpy as np
from pathlib import Path

import xml.etree.ElementTree as ET

def get_bounding_box(xml_path):
    tree = ET.parse(xml_path) #przerabia xml na strukturę drzewa
    root = tree.getroot() #bierze najwyższy element w xml

    bbox = root.find(".//bndbox") #znajduje pierwszy tag bndbox w drzewie; // odpowiada za szukanie rekurencyjne

    if bbox is None: #jeśli nie ma bboxa zwracamy None
        return None, None, None, None

    def get_value(tag):
        bbox_inside = bbox.find(tag)
        return int(bbox.find(tag).text) if bbox_inside is not None else None # jeśli któraś współrzędna nie jest zwracana zwraca None

    xmin = get_value("xmin")
    ymin = get_value("ymin")
    xmax = get_value("xmax")
    ymax = get_value("ymax")

    return xmin, ymin, xmax, ymax

def crop_image(img_path, xml_path, pad = 0):
    img = cv2.imread(img_path)

    xmin, ymin, xmax, ymax = get_bounding_box(xml_path)

    def check_nones(boundry, value_if_none):
        if boundry is None:
            boundry = value_if_none
        return boundry

    xmin = check_nones(xmin, 0)
    ymin = check_nones(ymin, 0)
    xmax = check_nones(xmax, img.shape[1])
    ymax = check_nones(ymax, img.shape[0])

    #crop
    img = img[
        max(0,ymin - pad):min(img.shape[0], ymax + pad),
        max(0, xmin - pad):min(img.shape[1], xmax + pad)
    ]

    return img

def preprocess_data(path_to_data = "data", img_size = 64, n_class = 10, n_samples_in_class = 100, crop = False, pad = 0, flatten = True):
    BASE_DIR = Path(__file__).resolve().parent.parent
    images_dir = BASE_DIR / path_to_data / "Images"
    annotations_dir = BASE_DIR / path_to_data / "Annotation"

    images = []
    labels = []

    for breed in os.listdir(images_dir)[:n_class]:
        breed_path = os.path.join(images_dir, breed)

        if not os.path.isdir(breed_path):
            continue

        for img_name in os.listdir(breed_path)[:n_samples_in_class]:
            img_path = os.path.join(breed_path, img_name)

            xml_name = img_name.replace(".jpg", "")
            xml_path = os.path.join(annotations_dir, breed, xml_name)

            if not os.path.exists(xml_path):
                continue

            if crop:
                img = crop_image(img_path, xml_path, pad)
            else:
                img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (img_size, img_size))

            images.append(img)
            labels.append(breed)

    X = np.array(images)
    y = np.array(labels)

    #flatten
    if flatten:
        X = X.reshape(X.shape[0], -1)

    return X, y



BASE_DIR = Path(__file__).resolve().parent.parent
images_dir = BASE_DIR / "data" / "Images" /'n02085620-Chihuahua'



