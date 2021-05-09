# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import cv2
import numpy as np
from collections import namedtuple
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

def create_txt(mode):
    '''
    Create new txt files for image path and label path
    :param mode:
    :return:
    '''
    os.makedirs('../train_test_inputs', exist_ok=True)
    new_file = '../train_test_inputs/Cityscape_eigen_'+mode +'_files.txt'
    if os.path.exists(new_file):
        os.remove(new_file)
    new_f = open(new_file, 'a')
    print('Create file: {}'.format(new_file))
    return new_f

def parse_label(plot=False):

    # parse train, val, test data    
    for label_dir in [train_dir, val_dir, test_dir]:
        # f = open(csv_file, "w")
        # f.write("img,label\n")
        count = 0
        print(label_dir.split('/')[-1])

        if label_dir.split('/')[-1] == 'train':
            new_f = new__f_train
        if label_dir.split('/')[-1] == 'test':
            new_f = new__f_test
        if label_dir.split('/')[-1] == 'val':
            new_f = new__f_val


        for city in os.listdir(label_dir):

            city_dir = os.path.join(label_dir, city)
            data_dir = city_dir.replace("gtFine", "leftImg8bit")
            data_dir = data_dir.replace("gtFine_trainvaltest", "leftImg8bit_trainvaltest")

            # if label_dir.split('/')[-1] == 'test':
            #     data_dir = data_dir.replace("test", "train")
            # if label_dir.split('/')[-1] == 'val':
            #     data_dir = data_dir.replace("val", "train")

            # print(city_dir)
            # print(data_dir)
            # print('')
            for filename in os.listdir(city_dir):
                # print(filename)
                if 'color' not in filename:
                    continue
                img_name = filename.split("gtFine")[0] + "leftImg8bit.png"
                img_name = os.path.join(data_dir, img_name)
                semantic_img_name = os.path.join(city_dir, filename)
                # print('npy 写入：', img_name, semantic_img_name)
                new_f.write(img_name+' '+semantic_img_name + '\n')
                # print("Finish %s" % (filename))

                if plot:
                    # 1024, 2048
                    rgb_img = cv2.imread(img_name)
                    rgb_img = cv2.resize(rgb_img, (512, 256))
                    print('RGB image path: {}, shape: {}'.format(img_name, np.shape(rgb_img)))

                    semantic_img = cv2.imread(semantic_img_name)
                    semantic_img = cv2.resize(semantic_img, (512, 256))
                    print('Semantic image path: {},  shape: {}'.format(semantic_img_name, np.shape(semantic_img)))

                    CONCATENATE = np.concatenate([rgb_img, semantic_img], axis=1)
                    cv2.imshow('sample', CONCATENATE)
                    cv2.waitKey(100)

                count+=1
        print(count)

    new_f.close()

'''debug function'''
def imshow(img, title=None):
    try:
        img = mpimg.imread(img)
        imgplot = plt.imshow(img)
    except:
        plt.imshow(img, interpolation='nearest')

    if title is not None:
        plt.title(title)
    
    plt.show()


if __name__ == '__main__':
    new__f_train = create_txt('train')
    new__f_test = create_txt('test')
    new__f_val = create_txt('val')
    print('')

    #############################
    # global variables #
    #############################
    root_dir = "G:\Datasets\cityscapes"

    label_dir = os.path.join(root_dir, "gtFine_trainvaltest/gtFine")
    train_dir = label_dir + "/train"
    val_dir = label_dir + "/val"
    test_dir = label_dir + "/test"

    print('train_dir: ', train_dir)
    print('val_dir: ', val_dir)
    print('test_dir: ', test_dir)
    print('')

    parse_label(plot=False)

    print(' ')
    mode = 'train'
    file = '../train_test_inputs/Cityscape_eigen_'+mode +'_files.txt'
    with open(file, 'r') as f:
        filenames = f.readlines()
    print(mode, len(filenames))

    mode = 'val'
    file = '../train_test_inputs/Cityscape_eigen_'+mode +'_files.txt'
    with open(file, 'r') as f:
        filenames = f.readlines()
    print(mode, len(filenames))

