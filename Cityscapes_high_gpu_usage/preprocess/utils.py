# -*- coding: utf-8 -*-
# @Time    : 2021/5/11 14:26
# @Author  : Zeqi@@
# @FileName: utils.py
# @Software: PyCharm

import cv2
import random
import numpy as np

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def input_standardization(images, mean, std):
    return (images-mean)/std

def preprocess(img, target, flip_rate, scale, input_size, num_classes):
    # Flip
    if random.random() < flip_rate:
        img = np.fliplr(img)
        target = np.fliplr(target)

    if scale != 1:
        resized_height = int(input_size[0])
        resized_width = int(input_size[1])
        image = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
    else:
        image = img
    orig_image = image.copy()
    # print('Image {}, min {}, max {}'.format(np.shape(image), np.min(image), np.max(image)))
    image = input_standardization(image / 255., mean, std)
    target_reshape= np.reshape(np.array(target), [np.prod(input_size), num_classes])
    # target_batch = np.reshape(target_batch, [-1])
    # print('Image batch: {}, label_batch: {}, target_batch: {}'.format(np.shape(image_batch), np.shape(label_batch), np.shape(target_reshape_batch)))

    return image, target_reshape