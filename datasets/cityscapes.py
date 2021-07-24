# -*- coding: utf-8 -*-
# @Time    : 2021/7/11 5:59
# @Author  : Zeqi@@
# @FileName: cityscapes.py
# @Software: PyCharm

import json
import os
import cv2
import numpy as np
from PIL import Image
from collections import namedtuple
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from datasets.aug_utils import RandomHorizontalFlip
from datasets.aug_utils import Noramlize
from datasets.aug_utils import RandomCrop
from datasets.aug_utils import ColorJitter

class Cityscapes(Sequence):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.

        **Parameters:**
            - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
            - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
            - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
            - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
            - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    print('Number of colors: {}'.format(len(train_id_to_color)))
    id_to_train_id = np.array([c.train_id for c in classes])
    print('Number of ids: {}, {}'.format(len(id_to_train_id), id_to_train_id))
    class_name = [c.name for c in classes if (c.train_id != -1 and c.train_id != 255)]
    print('class names num: {}, {}'.format(len(class_name), class_name))
    cls_dir = {}
    for c in classes:
        cls_dir[c.name] = c.train_id
    print(cls_dir)
    # train_id 255 is class 19, 0-18 total 19 classes + 19 as one additonal classes

    # train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    # train_id_to_color = np.array(train_id_to_color)
    # id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    'Generates data for Keras'
    def __init__(self, root, split='train', batch_size=16,
                 mode='fine', target_type='semantic', **kwargs):
        '''

        :param root:
        :param split:
        :param batch_size:
        :param shuffle:
        :param mode:
        :param target_type:
        :param kwargs:
        '''
        self.is_original = kwargs.get('orignal_size', False)
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.shuffle = True if split == 'train' else False
        self.batch_size = batch_size
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        print('Image path: ', self.images_dir)
        print('Label path: ', self.targets_dir)

        self.is_plot = kwargs.get('is_plot', False)

        # data augmentation
        # random crop
        self.input_size = kwargs.get('input_size', (512, 1024))
        self.crop_size = kwargs.get('crop_size', 768) # image crop as (768, 768) for training while (1024, 2048) for test
        # color jitter
        self.brightness = kwargs.get('brightness', 0.5)
        self.contrast = kwargs.get('contrast', 0.5)
        self.saturation = kwargs.get('saturation', 0.5)
        # normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if split == 'train':
            self.dim = (self.crop_size, self.crop_size)

        elif not self.is_original:
            self.dim = self.input_size

        else:
            self.dim = (1024, 2048)

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))
        print('Number of images: {}'.format(len(self.images)))
        print('Number of targets: {}'.format(len(self.targets)))

        self.on_epoch_end()


    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        images_list = [self.images[k] for k in indexes]
        targets_list = [self.targets[k] for k in indexes]

        # Generate data
        images, targets = self.__data_generation(images_list, targets_list)

        return images, targets

    def __data_generation(self, images_list, targets_list):
        # Initialization
        images = np.empty((self.batch_size, *self.dim, 3)) # rgb
        targets = np.empty((self.batch_size, *self.dim)) # one channel fm

        # Generate data
        CJ = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)
        for i, image in enumerate(images_list):
            image = Image.open(images_list[i]).convert('RGB')
            target = Image.open(targets_list[i])
            if self.split=='train':
                image, target = RandomCrop(image, target, size=self.crop_size)
                image, target = CJ(image, target)
                image, target = RandomHorizontalFlip(image, target, p=0.5)
                if self.is_plot:
                    image = np.array(image, dtype=np.uint8)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    target = np.array(target, dtype=np.uint8)
                    cv2.imshow('image', image)
                    cv2.imshow('target', target)
                image, target = Noramlize(image, target, self.mean, self.std)
            else:
                if self.is_plot:
                    image = np.array(image, dtype=np.uint8)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    target = np.array(target, dtype=np.uint8)
                    cv2.imshow('image', image)
                    cv2.imshow('target', target)

                if not self.is_original:
                    image = image.resize((self.dim[1], self.dim[0]), resample=Image.BILINEAR)
                    target = target.resize((self.dim[1], self.dim[0]), resample=Image.NEAREST)
                    # target = tf.image.resize(target, (self.crop_size, self.crop_size), method=tf.image.ResizeMethod.BILINEAR)
                # print('target', np.min(target), np.max(target), type(target))
                image, target = Noramlize(image, target, self.mean, self.std)
            target = self.encode_target(target)
            target[target == 255] = 19
            # target = tf.keras.utils.to_categorical(target, num_classes=19)

            images[i] = image
            targets[i] = target

            cv2.waitKey(100)
        # print('images: ', np.shape(images))
        # print('targets: ', np.shape(targets))
        return images, targets

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.images) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)


if __name__ == '__main__':
    train_params = {'root': 'G:\Datasets\cityscapes',
                    'split': 'train',
                    'mode': 'fine',
                    'batch_size': 16,
                    'crop_size': 768,
                    'brightness': 0.5,
                    'contrast': 0.5,
                    'saturation': 0.5,
                    'is_plot': False,
                    'target_type': 'semantic'}

    val_params = {'root': 'G:\Datasets\cityscapes',
                    'split': 'val',
                    'mode': 'fine',
                    'batch_size': 16,
                    'is_plot': True,
                    'target_type': 'semantic'}


    dataset = Cityscapes(**val_params)

    for data in dataset:
        image, target = data
        print(target.shape)
        print(np.shape(image), target.shape)

