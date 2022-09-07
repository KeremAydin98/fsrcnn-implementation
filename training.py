import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Data path
data_path = "./data/T91/"
# Get file paths
img_files = os.listdir(data_path)
# Png extension
extension = ".png"
# Load the images
images = [cv2.imread(data_path + f) for f in img_files if f.endswith(extension)]
# Turn them into numpy arrays
images = np.array(images,dtype=object)

# Display one image
cv2.imshow("Example image:", images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()






