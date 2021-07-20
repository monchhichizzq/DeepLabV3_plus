


#### Errors:

1. different size error:

        Invalid argument: padded_shape[0]=212 is not divisible by block_shape[0]=36
  
    Solution:
    
        set the same size of image for training and validation


2. number of samples % batchsize = 0
    
        Invalid argument: slice index 17 of dimension 0 out of bounds.

    