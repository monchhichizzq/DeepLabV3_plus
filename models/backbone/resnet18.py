# -*- coding: utf-8 -*-
# @Time    : 2021/7/20 7:05
# @Author  : Zeqi@@
# @FileName: resnet18.py
# @Software: PyCharm


import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import DepthwiseConv2D, BatchNormalization, ReLU
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import MaxPooling2D

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('Dilated-ResNet18')


class ResNet18():
    def __init__(self, use_bn=True, use_bias=False, **kwargs):
        self.use_bn = use_bn
        self.use_bias = use_bias
        self.layers_dims = [2, 2, 2, 2]

    def build_basic_block(self, inputs, filter_num, blocks, stride, dilated_rate, module_name):
        # The first block stride of each layer may be non-1
        x = self.Basic_Block(inputs, filter_num, stride, dilated_rate, block_name='{}_{}'.format(module_name, 0))

        for i in range(1, blocks):
            x = self.Basic_Block(x, filter_num, stride=1, dilated_rate=dilated_rate, block_name='{}_{}'.format(module_name, i))

        return x

    def Basic_Block(self, inputs, filter_num, stride=1, dilated_rate=1, block_name=None):
        conv_name_1 = 'block_' + block_name + '_conv_1'
        conv_name_2 = 'block_' + block_name + '_conv_2'
        skip_connection = 'block_' + block_name + '_skip_connection'

        # Part 1
        x = Conv2D(filter_num, (3, 3), strides=stride, padding='same', use_bias=self.use_bias,
                    kernel_initializer='he_normal', dilation_rate=dilated_rate, name=conv_name_1)(inputs)
        if self.use_bn:
            x = BatchNormalization(name=conv_name_1 + '_bn')(x)
        x = ReLU(name=conv_name_1 + '_relu')(x)

        # Part 2
        x = Conv2D(filter_num, (3, 3), strides=1, padding='same', use_bias=self.use_bias,
                    kernel_initializer='he_normal', name=conv_name_2)(x)
        if self.use_bn:
            x = BatchNormalization(name=conv_name_2 + '_bn')(x)

        #  skip
        if stride != 1 or dilated_rate != 1:
            residual = Conv2D(filter_num, (1, 1), strides=stride, use_bias=self.use_bias,
                                kernel_initializer='he_normal', dilation_rate=dilated_rate, name=skip_connection)(inputs)
        else:
            residual = inputs

        # Add
        x = Add(name='block_' + block_name + '_residual_add')([x, residual])
        out = ReLU(name='block_' + block_name + '_residual_add_relu')(x)

        return out

    def ConvBn_Block(self, inputs, filters, kernel_size, stride, block_name):
        conv_name = 'block_convbn' + block_name + '_conv'
        x = Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=self.use_bias,
                    kernel_initializer='he_normal', name=conv_name)(inputs)
        if self.use_bn:
            x = BatchNormalization(name=conv_name + '_bn')(x)
        out = ReLU(name=conv_name + '_relu')(x)
        return out

    def build(self, input_tensor):
        '''
        
        '''
        net = {}

        # Block 1
        x = input_tensor

        # Initial
        # 416, 416, 3 -> 208, 208, 3   downscale 2x
        x = self.ConvBn_Block(x, filters=64, kernel_size=(7, 7), stride=(2, 2), block_name='0')
        # 208, 208, 64 -> 104, 104, 64 downscale 4x
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # Basic blocks
        # 104, 104, 64 -> 104, 104, 64 
        x = self.build_basic_block(x, filter_num=64, blocks=self.layers_dims[0], stride=1, dilated_rate=1, module_name='module_0')
        net['low_level'] = x
        # 104, 104, 64 -> 52, 52, 128 downscale 8x
        x = self.build_basic_block(x, filter_num=128, blocks=self.layers_dims[1], stride=2, dilated_rate=1, module_name='module_1')
        # 52, 52, 128 -> 52, 52, 256  downscale 16x
        x = self.build_basic_block(x, filter_num=256, blocks=self.layers_dims[2], stride=1, dilated_rate=2, module_name='module_2')
        # 52, 52, 256 -> 52, 52, 512  downscale 32x
        x = self.build_basic_block(x, filter_num=512, blocks=self.layers_dims[3], stride=1, dilated_rate=4,
                                                module_name='module_3')

        net['out'] = x
        return net


'''
Torch original
  if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

'''