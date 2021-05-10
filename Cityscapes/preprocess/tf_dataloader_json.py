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

import os
import cv2
import random
import json
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from preparation.semantic_color2index import color_to_index, index_to_color

color2index = color_to_index()
index2color = index_to_color()

# from preparation.counting_objects import semantic_info
# index2color, index2name = semantic_info()
print(index2color)
# print(index2name)

# def color2semantic_labeling(path):
#     color_img = cv2.imread(path) # read aachen_000000_000019_gtFine_color.png
#     color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) # rgb
#     height, weight, _ = color_img.shape
#
#     idx_mat = np.zeros((height, weight))
#     except_color = []
#     for h in range(height):
#         for w in range(weight):
#             color = tuple(color_img[h, w])
#             try:
#                 index = color2index[color]
#                 idx_mat[h, w] = index
#             except:
#                 if color not in except_color:
#                     except_color.append(color)
#                 # no index, assign to void
#                 # idx_mat[h, w] = 19
#                 idx_mat[h, w] = 0
#     # Gt pixel labels
#     idx_mat = idx_mat.astype(np.uint8)
#     print(except_color)
#     return idx_mat

def index2color_labeling(label):
    height, weight, _ = label.shape
    color_semantic = np.zeros((height, weight, 3))
    for h in range(height):
        for w in range(weight):
            index = label[h, w]

            color = index2color[index[0]] if index[0] in index2color else index2color[0]
            for i in range(3):
                color_semantic[h, w][i] = color[i]
    # Gt pixel labels
    color_semantic = color_semantic.astype(np.uint8)
    return color_semantic

def create_onehot_encoding(label, n_class):
    h, w   = np.shape(label)
    target = np.zeros((n_class, h, w))
    for c in range(n_class):

        target[c][label==c] = 1
    target = np.transpose(target, (1, 2, 0))
    return target

def rotate_image(self, image, angle, flag=Image.BILINEAR):
    result = image.rotate(angle, resample=flag)
    return result

def input_standardization(images, mean, std):
    return (images-mean)/std

    # def random_crop(self, img, depth, height, width):
    #     assert img.shape[0] >= height
    #     assert img.shape[1] >= width
    #     assert img.shape[0] == depth.shape[0]
    #     assert img.shape[1] == depth.shape[1]
    #     x = random.randint(0, img.shape[1] - width)
    #     y = random.randint(0, img.shape[0] - height)
    #     img = img[y:y + height, x:x + width, :]
    #     depth = depth[y:y + height, x:x + width, :]
    #     return img, depth

    # def augment_image(self, image):
    #     # gamma augmentation
    #     gamma = random.uniform(0.9, 1.1)
    #     image_aug = image ** gamma

    #     # brightness augmentation
    #     # if self.args.dataset == 'nyu':
    #     #     brightness = random.uniform(0.75, 1.25)
    #     # else:
    #     brightness = random.uniform(0.9, 1.1)
    #     image_aug = image_aug * brightness

    #     # color augmentation
    #     colors = np.random.uniform(0.9, 1.1, size=3)
    #     white = np.ones((image.shape[0], image.shape[1]))
    #     color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
    #     image_aug *= color_image
    #     image_aug = np.clip(image_aug, 0, 1)

    #     return image_aug


    # def train_preprocess(self, image, depth_gt):
    #     # Random flipping
    #     do_flip = random.random()
    #     if do_flip > 0.5:
    #         image = (image[:, ::-1, :]).copy()
    #         depth_gt = (depth_gt[:, ::-1, :]).copy()

    #     # Random gamma, brightness, color augmentation
    #     do_augment = random.random()
    #     if do_augment > 0.5:
    #         image = self.augment_image(image)

    #     return image, depth_gt


class CityScape_DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, mode,
                       txt_file,
                       batch_size   = 32,
                       n_class      = 19,
                       crop         = False,
                       crop_dim     = (512,1024),
                       filp_rate    = 0,
                       scale_factor = 8,
                       max_degree   = 2.5,
                       one_hot      = True,
                       visual       = False,
                       shuffle      = True):

        'Initialization'
        self.mode        = mode
        self.scale       = scale_factor
        self.batch_size  = batch_size
        self.num_classes = n_class
        self.mean        = [0.485, 0.456, 0.406]
        self.std         = [0.229, 0.224, 0.225]
        self.flip_rate   = filp_rate
        self.visual      = visual
        self.shuffle     = shuffle
        self.new_height  = crop_dim[0]
        self.new_width   = crop_dim[1]
        self.onehot      = one_hot

        # Read data from pre-saved txt files
        with open(txt_file, 'r') as f:
            self.filenames = f.readlines()

        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        filenames_temp = [self.filenames[k] for k in indexes]

        # Generate data
        image_batch, target_batch, label_batch= self.__data_generation(filenames_temp)
        # print('Image_batch: {}'.format(image_batch.shape))
        # print('Target_batch: {}'.format(target_batch.shape))
        # print('Label_batch: {}'.format(label_batch.shape))

        if self.onehot:
            return image_batch, target_batch
        else:
            return image_batch, label_batch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, filenames_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        input_size   = (int(self.new_height*2/self.scale), int(self.new_width*2/self.scale))
        image_batch  = np.empty((self.batch_size, *input_size, 3))
        target_batch = np.empty((self.batch_size, *input_size, self.num_classes))
        label_batch  = np.empty((self.batch_size, *input_size, 1))
        target_reshape_batch = np.empty((self.batch_size, np.prod(input_size), self.num_classes))


        # Generate data
        for i, sample_path in enumerate(filenames_temp):
            # Store sample
            image_name = sample_path.split()[0]
            label_name = sample_path.split()[1]
            image_path = image_name

            # label_path = label_name.split('.')[0] + '.json'
            # label_path = label_path.replace("color", "polygons")
            # with open(label_path) as label_json:
            #     label = json.load(label_json)
            # print(label, np.shape(label))
            #
            label_path = label_name.replace("color", "labelIds")


            # load image and gt
            img    = cv2.imread(image_path)
            # label  = np.load(label_path)
            label  = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            # print('img: {}, label: {}'.format(img.shape, label.shape))
            target = create_onehot_encoding(label, self.num_classes)
            # print('img: {}, label: {}, target: {}'.format(img.shape, label.shape, target.shape))

            #Flip
            if random.random() < self.flip_rate:
                img    = np.fliplr(img)
                label  = np.fliplr(label)
                target = np.fliplr(target)

            if self.scale != 1:
                resized_height = int(input_size[0])
                resized_width  = int(input_size[1])
                image  = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
                target = cv2.resize(target, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
                label  = cv2.resize(label, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)

            else:
                image = img
            orig_image = image.copy()
            # print('Image {}, min {}, max {}'.format(np.shape(image), np.min(image), np.max(image)))
            image = input_standardization(image/255., self.mean, self.std)
            label = np.expand_dims(label, axis=-1)
           
            image_batch[i]  = image
            label_batch[i]  = label
            target_batch[i] = target
            target_reshape_batch[i] = np.reshape(np.array(target), [np.prod(input_size), self.num_classes])
            # target_batch = np.reshape(target_batch, [-1])
            # print('Image batch: {}, label_batch: {}, target_batch: {}'.format(np.shape(image_batch), np.shape(label_batch), np.shape(target_reshape_batch)))

            if self.visual:
                print('Image {}, min {}, max {}'.format(np.shape(image), np.min(image), np.max(image)))
                image = np.asarray((image * self.std + self.mean)* 255, dtype=np.uint8)
                print('Image {}, min {}, max {}'.format(np.shape(image), np.min(image), np.max(image)))
                label = np.asarray(label, dtype=np.uint8)
                color_semantic = index2color_labeling(label)
                from collections import Counter
                print(Counter(list(label.flatten())))
                print('Label {}, min {}, max {}'.format(np.shape(label), np.min(label), np.max(label)))
                target = np.asarray(target, dtype=np.uint8)
                print('target {}, min {}, max {}'.format(np.shape(target), np.min(target), np.max(target)))

                image          = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                color_semantic = cv2.cvtColor(color_semantic, cv2.COLOR_BGR2RGB)

                image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)
                color_semantic = cv2.resize(color_semantic, (1024, 512), interpolation=cv2.INTER_NEAREST)

                cv2.imshow('rgb ' + str(np.shape(image)), image)
                cv2.imshow('label ' + str(np.shape(color_semantic)), color_semantic)
                cv2.waitKey(10000)


        return image_batch, target_reshape_batch, label_batch

if __name__ == '__main__':
    root_dir   = "G:\Datasets\cityscapes"
    txt_dir    = '../train_test_inputs'
    train_file = os.path.join(txt_dir, "Cityscape_eigen_train_files.txt")
    val_file   = os.path.join(txt_dir, "Cityscape_eigen_val_files.txt")

    train_params = {'mode':         'train',
                    'txt_file':     train_file,
                    'batch_size':   5,
                    'n_class':      35,
                    'crop':         False,
                    'crop_dim':     (512, 1024),
                    'filp_rate':    0.5,
                    'scale_factor': 1,
                    'visual':       False,
                    'shuffle':      True}
    val_params =  {'mode':         'val',
                    'txt_file':     val_file,
                    'batch_size':   50,
                    'n_class':      35,
                    'crop':         False,
                    'crop_dim':     (512, 1024),
                    'filp_rate':    0,
                    'scale_factor': 1,
                    'visual':       False,
                    'shuffle':      False}

    # val_generator = CityScape_DataGenerator(**val_params)
    # next(iter(val_generator))
    train_generator = CityScape_DataGenerator(**train_params)
    next(iter(train_generator))


'''
{   0: [255, 255, 255], 
    11: array([70, 70, 70], dtype=uint8), 
    26: array([142,   0,   0], dtype=uint8), 
    8: array([232,  35, 244], dtype=uint8), 
    21: array([ 35, 142, 107], dtype=uint8), 
    20: array([  0, 220, 220], dtype=uint8), 
    13: array([153, 153, 190], dtype=uint8), 
    24: array([ 60,  20, 220], dtype=uint8), 
    3: array([0, 0, 0], dtype=uint8), 
    33: array([ 32,  11, 119], dtype=uint8), 
    25: array([  0,   0, 255], dtype=uint8), 
    19: array([ 30, 170, 250], dtype=uint8), 
    5: array([  0,  74, 111], dtype=uint8), 
    22: array([152, 251, 152], dtype=uint8), 
    27: array([70,  0,  0], dtype=uint8), 
    32: array([230,   0,   0], dtype=uint8), 
    7: array([128,  64, 128], dtype=uint8),
     6: array([81,  0, 81], dtype=uint8), 
     18: array([153, 153, 153], dtype=uint8), 
     15: array([100, 100, 150], dtype=uint8), 
     29: array([90,  0,  0], dtype=uint8)}
{   0: 'unknown', 
    11: 'building', 
    26: 'train', 
    8: 'wall', 
    21: 'trailer', 
    20: 'traffic sign', 
    13: 'fence', 
    24: 'persongroup', 
    3: 'guard rail', 
    33: 'bicyclegroup', 
    25: 'ridergroup', 
    19: 'traffic light', 
    5: 'dynamic', 
    22: 'terrain', 
    27: 'truck', 
    32: 'motorcycle',
    7: 'rail track',
    6: 'ground', 
    18: 'polegroup', 
    15: 'bridge', 
    29: 'caravan'}


'''
# train
'''
{   0: [255, 255, 255], 
    3: array([0, 0, 0], dtype=uint8), 
    26: array([142,   0,   0], dtype=uint8), 
    8: array([232,  35, 244], dtype=uint8), 
    17: array([153, 153, 153], dtype=uint8), 
    4: array([0, 0, 0], dtype=uint8), 
    24: array([ 60,  20, 220], dtype=uint8), 
    25: array([  0,   0, 255], dtype=uint8), 
    22: array([152, 251, 152], dtype=uint8), 
    19: array([ 30, 170, 250], dtype=uint8), 
    5: array([  0,  74, 111], dtype=uint8), 
    13: array([153, 153, 190], dtype=uint8), 
    33: array([ 32,  11, 119], dtype=uint8), 
    32: array([230,   0,   0], dtype=uint8), 
    7: array([128,  64, 128], dtype=uint8), 
    15: array([100, 100, 150], dtype=uint8), 
    30: array([110,   0,   0], dtype=uint8), 
    18: array([153, 153, 153], dtype=uint8), 
    29: array([90,  0,  0], dtype=uint8), 
    27: array([70,  0,  0], dtype=uint8), 
    14: array([180, 165, 180], dtype=uint8), 
    31: array([100,  80,   0], dtype=uint8)}
    
{   0: 'unknown', 
    3: 'rectification border', 
    26: 'rail track', 
    8: 'vegetation', 
    17: 'traffic sign', 
    4: 'wall', 
    24: 'person', 
    25: 'ridergroup', 
    22: 'ground', 
    19: 'truckgroup', 
    5: 'dynamic', 
    13: 'motorcyclegroup', 
    33: 'bicyclegroup', 
    32: 'motorcycle', 
    7: 'tunnel', 
    15: 'bridge', 
    30: 'trailer', 
    18: 'polegroup', 
    29: 'caravan', 
    27: 'truck', 
    14: 'guard rail', 
    31: 'train'}


'''

index2name = {}