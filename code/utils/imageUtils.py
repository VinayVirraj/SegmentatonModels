import os
import cv2
import numpy as np
from torch.utils.data import random_split
from configure import training_configs


def load_images(data_dir, is_mask=False):
    image_list = []
    files = sorted(os.listdir(data_dir))
    resize_shape = training_configs["img_resize_shape"]

    for file in files:
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (resize_shape, resize_shape))
        image_list.append(img)

    if is_mask:
        binary_masks = []
        for mask in image_list:
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)
            binary_masks.append(binary_mask / 255.0)
        return np.array(binary_masks)

    return np.array(image_list)


def data_split(data, train_ratio, val_ratio):
    train_size = int(train_ratio * len(data))
    val_size = int(val_ratio * len(data))
    test_size = len(data) - train_size - val_size

    train_data, val_data, test_data = random_split(data, [train_size, val_size, test_size])
    return train_data, val_data, test_data
