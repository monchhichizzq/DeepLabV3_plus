# -*- coding: utf-8 -*-
# @Time    : 2021/7/20 7:02
# @Author  : Zeqi@@
# @FileName: nets.py
# @Software: PyCharm

from os.path import dirname, realpath, sep, pardir
# sep: \
# pardir: ..
abs_path = dirname(realpath(__file__)) + sep + pardir
print(abs_path)
import sys
sys.path.append(abs_path)

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from models.backbone.resnet18 import ResNet18
from models.deeplab import DeepLabV3Plus



def dilated_resnet18(input_tensor, use_bn, use_bias):
    net_outs = ResNet18(use_bn, use_bias).build(input_tensor=input_tensor)
    model = Model(input_tensor, net_outs['out'], name='dilated_resnet18')
    return model

def deeplabv3plus_resnet18(input_tensor, use_bn, use_bias, num_classes, backbone='resnet18'):
    deeplab = DeepLabV3Plus(backbone, use_bn, use_bias, num_classes, aspp_dilate=[12, 24, 36])
    model = deeplab.build(input_tensor)
    return model

def deeplabv3plus_resnet50(input_tensor, use_bn, use_bias, num_classes, backbone='resnet50'):
    deeplab = DeepLabV3Plus(backbone, use_bn, use_bias, num_classes, aspp_dilate=[12, 24, 36])
    model = deeplab.build(input_tensor)
    return model


if __name__ == '__main__':
    inputs = Input(shape=(416, 416, 3))
    # model = dilated_resnet18(inputs, use_bn=True, use_bias=False)
    # model.summary()

    # model = deeplabv3plus_resnet18(inputs, backbone='resnet18', use_bn=True, use_bias=False, num_classes=21)
    # model.summary()
    # model.save('deeplabv3+_resnet18.h5')

    model = deeplabv3plus_resnet50(inputs, use_bn=True, use_bias=False, 
                                   num_classes=19, backbone='resnet50')
    # model.save('deeplabv3+_resnet50.h5')
    model.summary()

