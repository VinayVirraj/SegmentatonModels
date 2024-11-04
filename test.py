import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

images_path = "E:\D\capstone\SegmentatonModels\data\images"



files = sorted(os.listdir(images_path))

x = []
y = []
for file in files:
    img_path = os.path.join(images_path, file)
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized_c = cv2.resize(img_gray, (256, 256), interpolation=cv2.INTER_CUBIC)
    blur3 = cv2.bilateralFilter(img_resized_c,9,50,50)
    hist = cv2.equalizeHist(blur3)
    images = [img_gray, blur3, hist]
    titles = ["Original Image", "bilinear transform", "hist eq"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    # Show the images
    plt.show()