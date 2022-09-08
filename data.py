import os
import cv2
import numpy as np
import config

"""
Loading the images
"""

class T91_dataset:

    def __init__(self,batch_size):

        self.batch_size = batch_size
        self.data_path = config.DATA_PATH
        self.extension = config.EXTENSION

        # Get file paths
        img_files = os.listdir(self.data_path)

        # Load the images
        images = [cv2.imread(self.data_path + f) for f in img_files if f.endswith(self.extension)]

        train_test = 0.7
        val_test = 0.66

        # Train images
        train_images = images[:int(len(images) * train_test)]

        val_test_images = images[int(len(images) * train_test):]

        # Validation images
        val_images = val_test_images[:int(len(val_test_images) * val_test)]

        # Test images
        test_images = val_test_images[int(len(val_test_images) * val_test):]

        all_images = [train_images, val_images, test_images]

        for ind, type in enumerate(all_images):

            if ind in [0,1]:

                for image in type:

                    tf.image.random_crop(image, size=config.TARGET_SHAPE)
