from PIL import Image

# Data path
DATA_PATH = "./data/T91/"

# Png extension
EXTENSION = ".png"

# From the smallest image in the dataset
HR_TARGET_SHAPE = (78,78)

# Rescaling factor
RESCALING_FACTOR = 4

# Colored images
COLOR_CHANNELS = 3

LR_TARGET_SHAPE = (HR_TARGET_SHAPE[0] // 4, HR_TARGET_SHAPE[1] // 4)

# Configurations
d = 56
s = 12
m = 4

# Bicubic Downsample
DOWNSAMPLE_MODE = Image.BICUBIC