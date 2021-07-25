


#### Results:

#### Performance on Cityscapes (19 classes, 1024 x 2048)

Training: 768x768 random crop  
validation: 1024x2048

**pytorch version**

|  Model          | Batch Size  | FLOPs  | train/val OS   |  mIoU        |  overall accuracy | mean accuracy | FreqW accuracy |time (ms)    | 
| :--------        | :-------------: | :----:   | :-----------: | :--------: | :--------: |  :----:   | :--------: | :----:   |
| DeepLabV3Plus-MobileNet   | 16      |  135G      |  16/16   |  0.721  | 0.952  | 0.800  | 0.913  | 38.06  |   [Download](https://www.dropbox.com/s/753ojyvsh3vdjol/best_deeplabv3plus_mobilenet_cityscapes_os16.pth?dl=0) | [Download](https://share.weiyun.com/aSKjdpbL) 
| DeepLabV3Plus-ResNet50   | 16      |  N/A      |  16/16   |  0.763  | 0.957  | 0.840  | 0.921  | 22.02  |   [Download](https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb/view?usp=sharing) | [Comming Soon]()
| DeepLabV3Plus-ResNet101   | 16      |  N/A      |  16/16   |  0.762  | 0.959  | 0.838  | 0.924  | 53.82  |   [Download](https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb/view?usp=sharing) | [Comming Soon]()

**tensorflow version**

|  Model          | Batch Size  | FLOPs  | train/val OS   |  mIoU    |  overall accuracy | mean accuracy | FreqW accuracy |time (ms)    | 
| :--------        | :-------------: | :----:   | :-----------: | :--------: | :--------: |  :----:   | :--------: | :----:   |
| DeepLabV3Plus-ResNet18   | 8      |        |  16/16   |  0.648  | 0.9421  |   |  | 304.37  |   [Download](https://www.dropbox.com/s/753ojyvsh3vdjol/best_deeplabv3plus_mobilenet_cityscapes_os16.pth?dl=0) | [Download](https://share.weiyun.com/aSKjdpbL) 
| DeepLabV3Plus-ResNet18   | 4      |        |  8/8  |  0.649  | 0.945  | |   | 352.53  |   [Download](https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb/view?usp=sharing) | [Comming Soon]()


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