# fsrcnn-implementation
Code implementation of the paper "Accelerating the Super-Resolution Convolutional Neural Network"

The link of the paper: 

https://arxiv.org/abs/1608.00367

## 1. Gather general information

Super Resolution Convolutional Network(SRCNN) used to be state of the art resolution enhancer model with its' superior performance to the previous hand-crafted models in speed and restoration quality. On the other hand, the model had a very high computation cost, therefore the training took a long time. The authors of the paper examined the model in order to find out which parts are slowing down the model and how to fix it. They found out three key reasons for high computations:

- Bicubic Interpolation at the beginning
- Mapping on interpolated images
- Large filter sizes

Authors proposed some changes to speed up the model. First, they removed the interpolation at the beginning and started with feature extraction phase. Then a shrinking operation has been added before the mapping, thus greatly decreased computation cost of mapping since mapping is done on the low resolution images instead of the high resolution images after mapping the image was expanded. Lastly, they realized with smaller filter sizes and more layers can both speed up and improve the performance of the mapping layers.

The data that was used in the paper is called T91:

[https://www.kaggle.com/datasets/ll01dm/t91-image-dataset](https://www.kaggle.com/datasets/ll01dm/t91-image-dataset)

## 2. Examine Model Architecture

- FSRCNN can be decomposed into five parts – feature extraction,
shrinking, mapping, expanding and deconvolution.
- The first four parts are convolution
layers, while the last one is a deconvolution layer.

<p align="center">
  <img src="https://user-images.githubusercontent.com/77073029/189313395-b36730c2-c2b9-49ac-a889-69d8249e2ac0.png">
</p>

### 1. Feature extraction:

- FSRCNN performs feature extraction on the original LR image without interpolation.
- By doing convolution with the first set of filters, each patch of the input (1-pixel overlapping) is represented as a high-dimensional feature vector.
- Therefore, they adopted a smaller filter size f1 = 5 with little information loss. For the number of channels, they followed SRCNN
 to set c1 = 1. Then they only needed to determine the filter number n1. From another perspective, n1 can be regarded as the number of LR feature dimension, denoted as d 
– Finally, the first layer can be represented as Conv(5; d; 1).
    
### 2. Shrinking:

- Here s is the second sensitive variable that determines the level of shrinking, and the second layer can be represented as Conv(1; s; d). This strategy greatly reduces the number of parameters.

### 3. Mapping:

- The non-linear mapping step is the most important part that affects
    the SR performance, and the most influencing factors are the width (i.e., the number
    of filters in a layer) and depth (i.e., the number of layers) of the mapping layer.
- First, as a trade-off
    between the performance and network scale, they adopted a medium filter size f3 = 3.
    Then, to maintain the same good performance as SRCNN, they used multiple 33 layers
    to replace a single wide one.
- To be consistent, all mapping layers contain the same number of filters n3 = s. Then the
    non-linear mapping part can be represented as m(number of mapping layers)  x Conv(3; s; s).
    
### 4. Expanding:

- They added
    an expanding layer after the mapping part to expand the HR feature dimension. To
    maintain consistency with the shrinking layer, they also adopted 1 x 1 filters, the number
    of which is the same as that for the LR feature extraction layer. As opposed to the
    shrinking layer Conv(1; s; d), the expanding layer is Conv(1; d; s).
    
### 5. Deconvolution:

- The last part is a deconvolution layer, which up-samples and aggregates
    the previous features with a set of deconvolution filters.
- For convolution, the filter is
    convolved with the image with a stride k, and the output is 1 / k times of the input.
    Contrarily, if we exchange the position of the input and output, the output will be k
    times of the input. They took advantage of this property to set
    the stride k = n, which is the desired upscaling factor. Then the output is directly the
    reconstructed HR image.
- Lastly, we can represent the deconvolution layer as DeConv(9; 1; d).

## 3. Training Details

- Before training, there are two data augmentations. They augmented the data in two ways. 1) Scaling: each image is
downscaled with the factor 0.9, 0,8, 0.7 and 0.6. 2) Rotation: each image is rotated with
the degree of 90, 180 and 270. Therefore, 19 times more images
for training.
- We took only high-resolution (HR) images from DIV2K, both train and validation parts, 900 images in total. Then combine train and validation parts into a single folder and create the custom train-val-test split based on image ids: ids 1–700 for train, ids 701–800 for validation, ids 801–900 for the test. Low-resolution (LR) images are generated from high-resolution images by bicubic downsampling.
- We will do up-sampling with transposed convolution.
- Zero padding is used in all layers.
- All parameters are optimized
using stochastic gradient descent with the standard backpropagation.
- Weights for convolutional layers are initialized “with the method designed for PReLU in [23].” And it’s a reference to another paper, that explains He normal initializer.

## Conclusion

I have concluded training, however I could not get successful results with the paper's model. Adam optimizer greatly increased the performance in contrast to paper's optimizer SGD. Three callbacks were used: model checkpoint, learning rate scheduler time and early stopping. There is still hyperparameter tuning that needs to be done to improve results.

<p align="center">
  <img src="https://user-images.githubusercontent.com/77073029/189302519-02affc96-d2d1-4bec-ae20-a7a2c1f2b10c.png">
</p>

# Thanks to Olga Chernytska

Huge thanks to OlgaChernytska for her great blog explanation for code implementations in Medium and her clean code implementation in Github. They both benefitted me.
Her blog:

https://medium.com/towards-data-science/learn-to-reproduce-papers-beginners-guide-2b4bff8fcca0

Her GitHub repository:

https://github.com/OlgaChernytska/Super-Resolution-with-FSRCNN
