# -*- coding: utf-8 -*-
# @Time    : 2021/5/9 20:12
# @Author  : Zeqi@@
# @FileName: mobilenetv2.py
# @Software: PyCharm

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (ReLU, Activation, Add, BatchNormalization, Concatenate,
                                    Conv2D, DepthwiseConv2D, Dropout, GlobalAveragePooling2D,
                                     Input, Lambda, ZeroPadding2D)



def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    '''
    Inverted bottleneck
    :param inputs:
    :param expansion:
    :param stride:
    :param alpha:
    :param filters:
    :param block_id:
    :param skip_connection:
    :param rate:
    :return:
    '''

    # in_channels = inputs.shape[-1].value  # inputs._keras_shape[-1]
    in_channels = inputs.shape[-1]
    pointwise_filters = _make_divisible(int(filters * alpha), 8)
    prefix = 'expanded_conv_{}_'.format(block_id)

    x = inputs
    # Expand the channel numbers by using 1x1 convolution, normally 6 times
    if block_id:
        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_bn')(x)
        x = ReLU(max_value=6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Using 3x3 depthwise convolution to extract the features of the network
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_bn')(x)
    x = ReLU(max_value=6, name=prefix + 'depthwise_relu')(x)

    # Reduce the channel numbers by using 1x1 convolution, back to the input filter number
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_bn')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    return x


def mobilenetV2(inputs, alpha=1):
    first_block_filters = _make_divisible(32 * alpha, 8)
    # 416,416,3 -> 208,208,32 downsample x2
    # 通常情况(torch)下使用 zeropadding+conv2d(pad='valid), tensorflow中可以直接使用conv2d(pad='same'), 效果相同吗？
    x = Conv2D(first_block_filters,
               kernel_size=3,
               strides=(2, 2), padding='same',
               use_bias=False, name='Conv')(inputs)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_bn')(x)
    x = ReLU(max_value=6, name='Conv_Relu6')(x)
    x = Dropout(0.2)(x)

    # 208,208,32 -> 208,208,16
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, skip_connection=False)

    # 208,208,16 -> 104,104,24 downsample x4
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, skip_connection=True)
    x = Dropout(0.3)(x)
    skip1 = x # downsample x4

    # 104,104,24 -> 52,52,32 downsample x8
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, skip_connection=True)
    x = Dropout(0.5)(x)

    # ---------------------------------------------------------------#
    # 52,52,32 -> 52,52,64 downsample x8 -> downsample x16, dilation_rate=2
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=6, skip_connection=False)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=7, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=8, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=9, skip_connection=True)
    x = Dropout(0.5)(x)

    # 52,52,64 -> 52,52,96
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=10, skip_connection=False)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=11, skip_connection=True)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=12, skip_connection=True)
    x = Dropout(0.5)(x)

    # 52,52,96 -> 52,52,160
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                            expansion=6, block_id=13, skip_connection=False)
    # downsample x8 -> downsample x32, dilation_rate=4
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=14, skip_connection=True)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=15, skip_connection=True)

    # 52,52,160 -> 52,52,320
    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=16, skip_connection=False)
    x = Dropout(0.5)(x)

    # x32 downsampling
    return x, skip1

if __name__ == '__main__':
    inputs = Input(shape=(416, 416, 3))
    out, skip1 = mobilenetV2(inputs, alpha=1)
    MBV2 = Model(inputs=inputs, outputs=out, name='atrous_mobilenetv2')
    MBV2.summary()