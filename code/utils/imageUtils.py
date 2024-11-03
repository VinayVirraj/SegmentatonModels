import os
import cv2
import numpy as np
from torch.utils.data import random_split
from configure import training_configs


def preprocess_images(img, is_mask=False):
    resize_shape = training_configs["img_resize_shape"]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not is_mask:
        img_resized = cv2.resize(img_gray, (resize_shape, resize_shape), interpolation=cv2.INTER_CUBIC)
        blur = cv2.bilateralFilter(img_resized,9,50,50)
        hist_eq = cv2.equalizeHist(blur)
        return hist_eq/255.0
    else:
        img_resized = cv2.resize(img_gray, (resize_shape, resize_shape), interpolation=cv2.INTER_NEAREST)
        return img_resized

def load_images(data_dir, is_mask=False):
    image_list = []
    files = sorted(os.listdir(data_dir))

    for file in files:
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path)
        image_list.append(preprocess_images(img, is_mask))

    if is_mask:
        binary_masks = []
        for mask in image_list:
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            binary_masks.append(binary_mask / 255.0)
            
        return np.array(binary_masks)

    return np.array(image_list)


def data_split(data, train_ratio, val_ratio):
    train_size = int(train_ratio * len(data))
    val_size = int(val_ratio * len(data))
    test_size = len(data) - train_size - val_size

    train_data, val_data, test_data = random_split(data, [train_size, val_size, test_size])
    return train_data, val_data, test_data
