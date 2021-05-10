# -*- coding: utf-8 -*-
# @Time    : 2021/5/9 20:13
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

from models.mobilenetv2 import mobilenetV2


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = ReLU(name=prefix+'_relu')(x)

    # 首先使用3x3的深度可分离卷积
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_bn', epsilon=epsilon)(x)
    if depth_activation:
        x = ReLU(name=prefix + '_depthwise_relu')(x)

    # 利用1x1卷积进行通道数调整
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_bn', epsilon=epsilon)(x)
    if depth_activation:
        x = ReLU(name=prefix + '_pointwise_relu')(x)

    return x


def Deeplabv3(input_shape=(416, 416, 3), classes=21, alpha=1.):

    img_input = Input(shape=input_shape)

    # Atrous MobileNetV2 Output
    # x         52, 52, 320
    # skip1     104, 104, 24
    x, skip1 = mobilenetV2(img_input, alpha)
    size_before = tf.keras.backend.int_shape(x)
    pool_size = size_before[1]

    # Atrous Spatial Pyramid Pooling
    # ---------------------------------------------------------------#
    # Image pooling
    #   全部求平均后，再利用expand_dims扩充维度
    #   52,52,320 -> 1,1,320 -> 1,1,320
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
    #
    # ---------------------------------------------------------------#
    #   1x1 Conv
    # ---------------------------------------------------------------#
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_bn', epsilon=1e-5)(b0)
    b0 = ReLU(name='aspp0_activation')(b0)

    # ---------------------------------------------------------------#
    #   rate值与OS相关，SepConv_BN为先3x3膨胀卷积，再1x1卷积，进行压缩
    #   其膨胀率就是rate值
    # ---------------------------------------------------------------#
    b1 = SepConv_BN(x, 256, 'aspp1', rate=6, depth_activation=True, epsilon=1e-5)
    b2 = SepConv_BN(x, 256, 'aspp2', rate=12, depth_activation=True, epsilon=1e-5)
    b3 = SepConv_BN(x, 256, 'aspp3', rate=18, depth_activation=True, epsilon=1e-5)

    # 52, 52, 256 + 52, 52, 256 + 52, 52, 256 + 52, 52, 256 + 52, 52, 256 -> 52, 52, 1280
    x = Concatenate()([b4, b0, b1, b2, b3])

    # 1x1 Conv channel reduction
    # 52, 52, 512 -> 52,52,256
    x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_bn', epsilon=1e-5)(x)
    x = ReLU(name='concat_projection_relu')(x)
    x = Dropout(0.1)(x)

    # skip1.shape[1:3] 为 104,104
    # 52,52,256 -> 104,104,256
    # x = Lambda(lambda xx: tf.image.resize(x, skip1.shape[1:3]))(x)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)

    # 104,104,24 -> 104,104,48
    dec_skip1 = Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(name='feature_projection0_bn', epsilon=1e-5)(dec_skip1)
    dec_skip1 = ReLU(name='feature_projection0_relu',)(dec_skip1)

    # 104,104,256 + 104,104,48 -> 104,104,304
    x = Concatenate()([x, dec_skip1])
    # 104,104,304 -> 104,104,256 -> 104,104,256
    x = SepConv_BN(x, 256, 'decoder_conv0', depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1', depth_activation=True, epsilon=1e-5)

    # 104,104,256 -> 104,104,2 -> 416,416,2
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Conv2D(classes, (1, 1), padding='same')(x)
    # x = Lambda(lambda xx: tf.image.resize(xx, size_before3[1:3]))(x)
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    x = Reshape((-1, classes))(x)
    x = Softmax()(x)

    model = Model(img_input, x, name='deeplabv3plus')
    return model

if __name__ == '__main__':
    # (512, 1024) cityscape
    model = Deeplabv3(input_shape=(512, 1024, 3), classes=21, alpha=1.)
    model.summary()
    model.save('atrous_mbv2_deeplabv3plus.h5')