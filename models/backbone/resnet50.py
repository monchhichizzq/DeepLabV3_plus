# -*- coding: utf-8 -*-
# @Time    : 2021/7/20 7:05
# @Author  : Zeqi@@
# @FileName: resnet50.py
# @Software: PyCharm

from tensorflow.keras import layers
from tensorflow.keras.models import Model

class ResNet50():
    def __init__(self, use_bn=True, use_bias=False, 
                include_top=True, classes=1000, **kwargs):
        self.add_bn = use_bn
        self.add_bias = use_bias
        self.include_top = include_top
        self.classes = classes
        self.remove_maxpool = False

    def stack1(self, x, filters, blocks, stride=1, dilated_rate=1, name=None):
        x = self.block1(x, filters, stride=stride, 
                        dilated_rate=dilated_rate, name=name + '_block1')
        for i in range(2, blocks + 1):
            x = self.block1(x, filters, dilated_rate=dilated_rate, 
                            conv_shortcut=False, name=name + '_block' + str(i))
        return x

    def block1(self, x, filters, kernel_size=3, stride=1, dilated_rate=1, conv_shortcut=True, name=None):
        '''
        Bottleneck: 
        '''
        
        bn_axis=3
        # bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        inputs = x
        # conv_layer 1
        x = layers.Conv2D(filters, 1, strides=stride, padding='same',
                          use_bias=self.add_bias, name=name + '_1_conv')(inputs)
        if self.add_bn:
            x = layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_1_conv_bn')(x)
        x = layers.Activation('relu', name=name + '_1_relu')(x)
        # conv_layer 2 - padding='SAME'
        x = layers.Conv2D(
            filters, 3, dilation_rate=dilated_rate, padding='same',
            use_bias=self.add_bias, name=name + '_2_conv')(x)
        if self.add_bn:
            x = layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_2_conv_bn')(x)
        x = layers.Activation('relu', name=name + '_2_relu')(x)
        # conv_layer 3
        x = layers.Conv2D(4 * filters, 1, padding='same',
                          use_bias=self.add_bias, name=name + '_3_conv')(x)
        if self.add_bn:
            x = layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_3_conv_bn')(x)
        # conv_layer shortcut
        if conv_shortcut:
            shortcut = layers.Conv2D(4 * filters, 1, strides=stride, padding='same',
                                    use_bias=self.add_bias, name=name + '_0_conv')(inputs)
            if self.add_bn:
                shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, 
                            name=name + '_0_conv_bn')(shortcut)
        else:
            shortcut = inputs
    
        x = layers.Add(name=name + '_add')([x, shortcut])
        x = layers.Activation('relu', name=name + '_out')(x)
        return x

    def build(self, input_tensor):

        '''
        1. For the task of image classification, the spatial resolution of the final feature maps 
        is usually 32 times smaller than the input image resolution and thus output stride = 32.
        2. For the task of semantic segmentation, it is too small.
        3. One can adopt output stride = 16 (or 8) for denser feature extraction by removing the
         striding in the last one (or two) block(s) and applying the atrous convolution correspondingly.
        4. Additionally, DeepLabv3 augments the Atrous Spatial Pyramid Pooling module, which probes 
        convolutional features at multiple scales by applying atrous convolution with different rates, 
        with the image-level features.
        '''

        bn_axis=3
        net = {}
        # x = layers.ZeroPadding2D(
        #     padding=((3, 3), (3, 3)), name='conv1_pad')(input_tensor) # not support 

        # 416x416x3 -> 208x208x64 downsample 2x
        x = layers.Conv2D(64, 7, strides=2, padding='SAME', 
                          use_bias=self.add_bias, name='conv1_conv')(input_tensor)

        if self.add_bn:
            x = layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name='conv1_conv_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)
 
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x) # not support 

        # 208x208x64 -> 104x104x64 downsample 4x
        if self.remove_maxpool: 
            x = layers.AveragePooling2D(2, strides=2, name='avgpool1_pool')(x)
        else:
            x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x) # not support 

        # 104x104x64 -> 104x104x256
        x = self.stack1(x, 64, 3, stride=1, dilated_rate=1, name='conv2')
        net['low_level'] = x
        # 104x104x256 -> 52x52x512 atrous False downsample 8x 
        # output stride = 8, aspp_pooling dilated_rate = 12, 24, 36
        x = self.stack1(x, 128, 4, stride=2, dilated_rate=1, name='conv3')
        # 52x52x512 -> 26x26x1024  atrous True downsample 16x 
        # output stride = 16, aspp_pooling dilated_rate = 6, 12, 18
        x = self.stack1(x, 256, 6, stride=1, dilated_rate=2, name='conv4')
        # 26x26x1024 -> 13x13x2048 atrous True downsample 32x
        x = self.stack1(x, 512, 3, stride=1, dilated_rate=4, name='conv5')
        net['out'] = x

        # if self.include_top:
        #     x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        #     # imagenet_utils.validate_activation(classifier_activation, weights)
        #     x = layers.Dense(self.classes, activation='softmax',
        #                     name='probs')(x)
        # else:
        #     if pooling == 'avg':
        #         x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        #     elif pooling == 'max':
        #         x = layers.GlobalMaxPooling2D(name='max_pool')(x)

        # # Create model.
        # model = Model(img_input, x, name='resnet_50')

        return net




# reference tf2 https://github.com/lattice-ai/DeepLabV3-Plus/blob/master/deeplabv3plus/model/deeplabv3_plus.py
# https://github.com/rishizek/tensorflow-deeplab-v3-plus
# 膨胀卷积在保持参数个数不变的情况下增大了卷积核的感受野