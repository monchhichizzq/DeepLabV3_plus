


#### Errors:

1. different size error:

        Invalid argument: padded_shape[0]=212 is not divisible by block_shape[0]=36
  
    Solution:
    
        set the same size of image for training and validation

2. number of samples % batchsize = 0
    
        Invalid argument: slice index 17 of dimension 0 out of bounds.
        

#### Reference

1. Atrous Convolution

    https://towardsdatascience.com/understanding-2d-dilated-convolution-operation-with-examples-in-numpy-and-tensorflow-with-d376b3972b25

2. Deeplabv3+

    https://towardsdatascience.com/review-deeplabv3-atrous-convolution-semantic-segmentation-6d818bfd1d74
    
    https://sh-tsang.medium.com/review-deeplabv3-atrous-separable-convolution-semantic-segmentation-a625f6e83b90