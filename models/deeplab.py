# -*- coding: utf-8 -*-
# @Time    : 2021/7/20 7:02
# @Author  : Zeqi@@
# @FileName: deeplab.py
# @Software: PyCharm

from os.path import dirname, realpath, sep, pardir
# sep: \
# pardir: ..
abs_path = dirname(realpath(__file__)) + sep + pardir
print(abs_path)
import sys
sys.path.append(abs_path)

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.layers import (ReLU, Add, BatchNormalization, Concatenate,
                                      Conv2D, DepthwiseConv2D, Dropout, UpSampling2D,
                                      GlobalAveragePooling2D, Input, Lambda, Reshape,
                                      Softmax, ZeroPadding2D, AveragePooling2D)
from tensorflow.keras.models import Model
# from tensorflow.keras.utils.data_utils import get_file

from models.backbone.resnet18 import ResNet18
from models.backbone.resnet50 import ResNet50

class DeepLabV3Plus:
    def __init__(self, backbone, add_bn, use_bias, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabV3Plus, self).__init__()
        '''
        backbone_out = ['low_level', 'out']
        project = conv2d + bn + relu
        ASPP = dilated_conv2d + bn + relu
        
        '''
        self.backbone = backbone
        self.add_bn = add_bn
        self.use_bias = use_bias
        self.num_classes = num_classes
        self.aspp_dilate = aspp_dilate

    def aspp_conv(self, x, filters, kernel_size, dilated_rate, idx):
        '''
            conv2d + bn + relu
        '''
        conv_name =  'aspp_conv_{}'.format(idx)
        bn_name =  'aspp_conv_{}_bn'.format(idx)
        act_name = 'aspp_conv_{}_relu'.format(idx)
        x = Conv2D(filters, (kernel_size, kernel_size), padding='same', kernel_initializer='he_normal', 
                    dilation_rate=dilated_rate, use_bias=self.use_bias, name=conv_name)(x)
        if self.add_bn:
            x = BatchNormalization(name=bn_name)(x)
        x = ReLU(name=act_name)(x)
        return x

    def assp_pooling(self, x):
        size_before = tf.keras.backend.int_shape(x)
        pool_size = size_before[1]

        # Atrous Spatial Pyramid Pooling
        # ---------------------------------------------------------------#
        # Image pooling
        #   全部求平均后，再利用expand_dims扩充维度
        #   52,52,512 -> 1,1,512 -> 1,1,512
        # ---------------------------------------------------------------#
        # b4 = GlobalAveragePooling2D()(x)
        # b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
        # b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
        b4 = x
        b4 = AveragePooling2D(pool_size=(pool_size, pool_size))(b4)
        b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
        b4 = BatchNormalization(name='image_pooling_bn', epsilon=1e-5)(b4)
        b4 = ReLU(name='image_pooling_relu')(b4)
        # 1,1,256 -> 52,52,256
        # b4 = Lambda(lambda x: tf.image.resize(x, size_before[1:3]))(b4)
        b4 = UpSampling2D(size=(pool_size, pool_size), interpolation='bilinear')(b4)
        return b4

    def aspp(self, x, atrous_rates):
        '''    
            conv2d + bn + relu
            dilated_conv2d 12 + bn + relu
            dilated_conv2d 24 + bn + relu
            dilated_conv2d 36 + bn + relu
            aspp pooling

        '''
        b0 = self.aspp_conv(x, 256, kernel_size=1, dilated_rate=1, idx='a0')
        rate1, rate2, rate3 = tuple(atrous_rates) # 6, 12, 18 - 12, 24, 36
        b1 = self.aspp_conv(x, 256, kernel_size=3, dilated_rate=rate1, idx='a1')
        b2 = self.aspp_conv(x, 256, kernel_size=3, dilated_rate=rate2, idx='a2')
        b3 = self.aspp_conv(x, 256, kernel_size=3, dilated_rate=rate3, idx='a3')
        b4 = self.assp_pooling(x)

        # 52, 52, 256 + 52, 52, 256 + 52, 52, 256 + 52, 52, 256 + 52, 52, 256 -> 52, 52, 1280
        x = Concatenate()([b4, b0, b1, b2, b3])

        # 52x52x1280 -> 52x52x256
        x = self.aspp_conv(x, 256, kernel_size=1, dilated_rate=1, idx='concate_project')
        x = Dropout(0.1)(x)
        return x


    def classifier(self, x, filters, kernel_size):
        conv_name =  'head_cl'
        bn_name =  'head_cl_bn'
        act_name = 'head_cl_relu'
        x = Conv2D(filters, (kernel_size, kernel_size), padding='same', kernel_initializer='he_normal', 
                   use_bias=self.use_bias, name=conv_name)(x)
        if self.add_bn:
            x = BatchNormalization(name=bn_name)(x)
        x = ReLU(name=act_name)(x)
        x = Conv2D(self.num_classes, (1, 1), padding='same', kernel_initializer='he_normal', 
                   use_bias=self.use_bias, name='head_final')(x)
        return x

    def build(self, input_tensor):
        print('Backbone: {}'.format(self.backbone))
        if self.backbone == 'resnet18':
            net_outs = ResNet18(use_bn=self.add_bn, use_bias=self.use_bias).build(input_tensor=input_tensor)
        if self.backbone == 'resnet50':
            net_outs = ResNet50(use_bn=self.add_bn, use_bias=self.use_bias).build(input_tensor=input_tensor)

        # layer 1 low-level outputs 104x104x48
        low_level_feature = self.aspp_conv(net_outs['low_level'], 48, kernel_size=1, dilated_rate=1, idx='project')

        # layer 4 outs 52x52x512
        output_feature = self.aspp(net_outs['out'], atrous_rates=self.aspp_dilate)
        # 52x52x256 -> 104x104x256
        output_feature = UpSampling2D(size=(2, 2), interpolation='bilinear')(output_feature)

        # 104,104,256 + 104,104,48 -> 104,104,304
        x = Concatenate()([output_feature, low_level_feature])

        # 104,104,304 -> 104,104,num_classes
        x = self.classifier(x, filters=256, kernel_size=3)
        
        # 104, 104, num_classes -> 416, 416, num_classes
        x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

        # x = Reshape((-1, self.num_classes))(x)
        x = Softmax()(x)

        model = Model(input_tensor, x, name='deeplabv3plus')
        return model



