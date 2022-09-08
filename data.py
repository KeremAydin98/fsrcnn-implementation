import os
import cv2
import config
import albumentations as A


class T91_dataset:

    def __init__(self,batch_size, type):

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

        if type == "train":

            self.dataset = train_images

        elif type == "validation":

            self.dataset = val_images

        else:

            self.dataset = test_images

        if type in ["train", "val"]:

            self.transform = A.compose([
                    A.RandomCrop(width=config.HR_TARGET_SHAPE[0], height=config.HR_TARGET_SHAPE[1]),
                    A.Downscale(scale_min=0.6, scale_max=0.9, always_apply=True),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=270)
                ])

        else:

            self.tranform = A.compose([
                    A.RandomCrop(width=config.HR_TARGET_SHAPE[0], height=config.HR_TARGET_SHAPE[1])
                ])

    def __getitem__(self, item):
