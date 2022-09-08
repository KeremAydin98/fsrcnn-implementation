from PIL import Image

# Data path
DATA_PATH = "./data/DIV2K/"

# Png extension
EXTENSION = ".png"

# From the smallest image in the dataset
HR_TARGET_SHAPE = (76,76)

# Rescaling factor
RESCALING_FACTOR = 4

# Colored images
COLOR_CHANNELS = 3

LR_TARGET_SHAPE = (HR_TARGET_SHAPE[0] // RESCALING_FACTOR, HR_TARGET_SHAPE[1] // RESCALING_FACTOR)

# Configurations
d = 56
s = 12
m = 4

# Bicubic Downsample
DOWNSAMPLE_MODE = Image.BICUBIC