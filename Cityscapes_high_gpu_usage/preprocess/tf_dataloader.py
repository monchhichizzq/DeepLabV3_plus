# -*- coding: utf-8 -*-
# @Time    : 2020/11/10 17:28
# @Author  : Zeqi@@
# @FileName: tf_dataloader.py
# @Software: PyCharm

from os.path import dirname, realpath, sep, pardir
abs_path = dirname(realpath(__file__)) + sep + pardir
print(abs_path)
import sys
sys.path.append(abs_path)

import time
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from preprocess.utils import preprocess


def create_onehot_encoding(label, n_class):
    h, w   = np.shape(label)
    target = np.zeros((n_class, h, w))
    for c in range(n_class):

        target[c][label==c] = 1
    target = np.transpose(target, (1, 2, 0))
    return target


def read_img(address_line, scale, input_size, num_classes, is_train):
    image_name = address_line[0]
    label_name = address_line[1]

    image_path = bytes.decode(image_name.numpy())
    label_path = bytes.decode(label_name.numpy())

    # load image and gt
    img = cv2.imread(image_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    # tf.print('img: {}, label: {}'.format(img.shape, label.shape))
    target = create_onehot_encoding(label, num_classes)

    flip_rate = 0.5 if is_train else 0
    image, target_reshape = preprocess(img, target, flip_rate, scale, input_size, num_classes)
    return image, target_reshape

@tf.autograph.experimental.do_not_convert
def preprocess_data(address_line, scale, input_size, num_classes, is_train):
    result_tensors = tf.py_function(read_img, [address_line, scale, input_size, num_classes, is_train], [tf.float32, tf.float32])
    result_tensors[0].set_shape((input_size[0], input_size[1], 3))
    result_tensors[1].set_shape((np.prod(input_size), num_classes))
    return result_tensors

def build_input(filenames_list, batch_size, scale, input_size, num_classes, is_train, num_parallel=4):
    ### buffer_size in prefetch and num_parallel_calls in `interleave`, `map` could be auto-optimized by tf.data.experimental.AUTOTUNE

    dataset = tf.data.Dataset.from_tensor_slices(filenames_list)
    # dataset = dataset.interleave( num_parallel_calls=Y) # interleave  并行I/O
    # dataset = dataset.map(lambda x: tf.py_function(read_img, [x, scale, input_size, num_classes, is_train], [tf.float32, tf.float32]), num_parallel_calls=num_parallel)
    dataset = dataset.map(lambda x: preprocess_data(x, scale, input_size, num_classes, is_train), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print(dataset)
    # 这里大概率是因为tf.dataset中使用了tf.py_function导致无法自动推导出张良的形状，所以需要自己手动设置形状。
    # num_parallel_calls 多线程运算preprocess

    if is_train:
        dataset = dataset.shuffle(10)  # 将数据打乱，数值越大，混乱程度越大
    dataset = dataset.batch(batch_size, drop_remainder=False)
    for data in dataset:
        img, label = data
        print(np.shape(img), np.shape(label), batch_size)
    # dataset = dataset.cache()
    # dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)   # prefetch 训练同时加载数据
    # dataset = dataset.prefetch(buffer_size=4)

    # iterator = dataset.make_xx_iterator()
    # iterator = dataset.make_one_shot_iterator().get_next()
    return dataset


if __name__ == '__main__':
    # hyperparameter
    scale = 1
    batch_size = 16
    num_classes = 34
    height, width = 1024, 2048
    input_size = (int(height / scale), int(width / scale))

    root_dir = "G:\Datasets\cityscapes"
    txt_dir = '../train_test_inputs'
    train_file = os.path.join(txt_dir, "Cityscape_eigen_train_files.txt")
    val_file = os.path.join(txt_dir, "Cityscape_eigen_val_files.txt")

    with open(val_file, 'r') as f:
        filenames = f.readlines()
    print(filenames)

    # filename reading time: 1 ms
    filenames_list = []
    start = time.time()
    for filename in filenames:
        image_name = filename.split()[0]
        label_name = filename.split()[1]
        label_path = label_name.replace("color", "labelIds")
        filenames_list.append([image_name, label_path])
    end = time.time()
    print('time: %.2f ms'%((end-start)*1000))

    train_ds = build_input(filenames_list, batch_size, scale, input_size, num_classes, is_train=True, num_parallel=8)
    val_ds = build_input(filenames_list, batch_size, scale, input_size, num_classes, is_train=False, num_parallel=8)


