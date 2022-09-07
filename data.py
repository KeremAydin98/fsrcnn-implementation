import os
import cv2
import numpy as np
import config

"""
Loading the images
"""
def load_image():
    # Data path
    data_path = config.data_path
    # Get file paths
    img_files = os.listdir(data_path)
    # Png extension
    extension = config.extension
    # Load the images
    images = [cv2.imread(data_path + f) for f in img_files if f.endswith(extension)]
    # Resizing the images
    images = [cv2.resize(image, config.target_shape) for image in images]
    # Turn them into numpy arrays
    images = np.array(images)

    return images
