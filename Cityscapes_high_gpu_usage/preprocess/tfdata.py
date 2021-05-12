# -*- coding: utf-8 -*-
# @Time    : 2021/5/11 13:21
# @Author  : Zeqi@@
# @FileName: tfdata.py
# @Software: PyCharm

import time
import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from utils import preprocess

def create_onehot_encoding(label, n_class):
    h, w   = np.shape(label)
    target = np.zeros((n_class, h, w))
    for c in range(n_class):

        target[c][label==c] = 1
    target = np.transpose(target, (1, 2, 0))
    return target


def read_img(address_line, scale, num_classes, is_train):
    image_name = address_line[0]
    label_name = address_line[1]

    image_path = bytes.decode(image_name.numpy())
    label_path = bytes.decode(label_name.numpy())

    # load image and gt
    img = cv2.imread(image_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    # tf.print('img: {}, label: {}'.format(img.shape, label.shape))
    target = create_onehot_encoding(label, num_classes)

    input_size = [img.shape[0], img.shape[1]]

    flip_rate = 0.5 if is_train else 0
    image, target_reshape = preprocess(img, target, flip_rate, scale, input_size, num_classes)
    return image, target_reshape

def tfread_img(address_line, scale, num_classes, is_train):
    image_name = address_line[0]
    label_name = address_line[1]

    image_path = bytes.decode(image_name.numpy())
    label_path = bytes.decode(label_name.numpy())

    # load image and gt
    # image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img_height, img_width, c = img.shape
    tf.image.resize(img, [img_height, img_width])
    # Gt
    label = tf.io.read_file(label_path)
    label = tf.image.decode_jpeg(label, channels=1)
    # tf.print('img: {}, label: {}'.format(img.shape, label.shape))
    target = tf.one_hot(label, num_classes)
    img = tf.cast(img, dtype=tf.float32)

    return img, target

if __name__ == '__main__':
    # hyperparameter
    scale = 1
    num_classes = 34

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

    dataset = tf.data.Dataset.from_tensor_slices(filenames_list)
    # # , num_parallel_calls=2
    # dataset = dataset.map(lambda x: tf.py_function(read_img, [x, scale, num_classes, True], [tf.float32, tf.float32]), num_parallel_calls=1)
    # interleave


    # Parallel mapping effect (read image only)
    # Non parallel time: 31618.19 ms
    # num_parallel_calls=2 time: 16295.13 ms 2x
    # num_parallel_calls=4 time: 9527.95 ms 3x
    # num_parallel_calls=8 time: 7226.71 ms 4.3x
    # num_parallel_calls=16 time: 6388.91 ms 5x
    # num_parallel_calls=32 time: 7292.46 ms 4 x

    # Parallel mapping effect (read image and one-hot)
    # Non parallel time: 70450.21 ms
    # num_parallel_calls=2 time: 38650.37 ms 1.8x
    # num_parallel_calls=4 time: 27879.88 ms 2.5x
    # num_parallel_calls=8 time: 23840.37 ms 3x
    # num_parallel_calls=16 time: 22840.63 ms 3.08x
    # num_parallel_calls=32 time: 23586.34 ms 3x

    # Parallel mapping effect (read image and one-hot tensorflow) # not good because not using gpu
    # Non parallel time: 66419.92 ms
    # num_parallel_calls=2 time: 47533.18 ms 1.8x
    # num_parallel_calls=4 time: 48050.27 ms 2.5x
    # num_parallel_calls=8 time: 48008.25 ms 3x
    # num_parallel_calls=16 time: 49120.98 ms 3.08x
    # num_parallel_calls=32 time: 54399.80 ms 3x

    # Parallel mapping effect (preprocess)
    # Generator time: 967673.608 ms ~ 1007076.591 ms
    # Non parallel time: 354645.04 ms
    # num_parallel_calls=2 time: 277183.09 ms 1.3x
    # num_parallel_calls=4 time: 210844.06 ms 1.7x
    # num_parallel_calls=8 time: 196079.17 ms 1.8x
    # num_parallel_calls=16 time: 196569.59 ms 1.8x
    # num_parallel_calls=32 time: 194382.31 ms 1.8x

    # Parallel mapping effect (preprocess)
    # Generator time: 967673.608 ms ~ 1007076.591 ms
    # Non parallel time: 354645.04 ms
    # num_parallel_calls=2 time: 277183.09 ms 1.3x
    # num_parallel_calls=4 time: 210844.06 ms 1.7x
    # num_parallel_calls=8 time: 196079.17 ms 1.8x
    # num_parallel_calls=16 time: 196569.59 ms 1.8x
    # num_parallel_calls=32 time: 194382.31 ms 1.8x

    parallel_start = time.time()
    for element in dataset:
        rgb_img, label_img = element
        # print(np.shape(rgb_img), np.shape(label_img))
    parallel_end = time.time()
    print('time: %.2f ms' % ((parallel_end - parallel_start) * 1000))