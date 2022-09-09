import os
import config
import albumentations as A
import numpy as np
from PIL import Image
import tensorflow as tf

class T91Dataset(tf.keras.utils.Sequence):

    def __init__(self,batch_size, which_type, color_channels):

        self.batch_size = batch_size
        self.data_path = config.DATA_PATH
        self.extension = config.EXTENSION
        self.color_channels = color_channels
        self.which_type = which_type

        # Get file paths
        img_files = os.listdir(self.data_path)

        # Load the images
        images = [np.array(Image.open(os.path.join(self.data_path, f))) for f in img_files if f.endswith(self.extension)]

        train_test = 0.7
        val_test = 0.66

        # Train images
        train_images = images[:int(len(images) * train_test)]

        val_test_images = images[int(len(images) * train_test):]

        # Validation images
        val_images = val_test_images[:int(len(val_test_images) * val_test)]

        # Test images
        test_images = val_test_images[int(len(val_test_images) * val_test):]

        if self.which_type == "train":

            self.dataset = train_images

        elif self.which_type == "validation":

            self.dataset = val_images

        else:

            self.dataset = test_images

        if self.which_type in ["train", "val"]:

            self.transform = A.Compose([
                    A.RandomCrop(width=config.HR_TARGET_SHAPE[0], height=config.HR_TARGET_SHAPE[1]),
                    A.Downscale(scale_min=0.6, scale_max=0.9, always_apply=True),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=270)
                ])

        else:

            self.transform = A.Compose([
                    A.RandomCrop(width=config.HR_TARGET_SHAPE[0], height=config.HR_TARGET_SHAPE[1])
                ])

        self.to_float = A.ToFloat(max_value=255)

    def __len__(self):

        return len(self.dataset) // self.batch_size

    def __getitem__(self, item):

        index = item * self.batch_size

        batch_images = self.dataset[index:index + self.batch_size]

        batch_hr_images = np.zeros((self.batch_size,) + config.HR_TARGET_SHAPE + (self.color_channels,))
        batch_lr_images = np.zeros((self.batch_size,) + config.LR_TARGET_SHAPE + (self.color_channels,))

        for i, image_fn in enumerate(batch_images):

            hr_image_transform = self.transform(image=np.array(image_fn))["image"]

            hr_image_transform_pil = Image.fromarray(hr_image_transform)
            lr_image_transform_pil = hr_image_transform_pil.resize(
                config.LR_TARGET_SHAPE, resample=config.DOWNSAMPLE_MODE
            )

            lr_image_transform = np.array(lr_image_transform_pil)

            batch_hr_images[i] = self.to_float(image=hr_image_transform)["image"]
            batch_lr_images[i] = self.to_float(image=lr_image_transform)["image"]

        return (batch_lr_images, batch_hr_images)
