# -*- coding: utf-8 -*-
# @Time    : 2021/7/11 6:25
# @Author  : Zeqi@@
# @FileName: aug_utils.py
# @Software: PyCharm

import cv2
import random
import numbers
import numpy as np
from PIL import Image
from PIL import ImageEnhance

def RandomHorizontalFlip(img, target, p=0.5):
    """Horizontally flip the given PIL Image randomly with a given probability.
       Args:
           p (float): probability of the image being flipped. Default value is 0.5
    """
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        target = target.transpose(Image.FLIP_LEFT_RIGHT)
    return  img, target

def Noramlize(img,
              target,
              mean,
              std):
    '''Normalize a tensor image with mean and standard deviation.
        Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
        will normalize each channel of the input ``torch.*Tensor`` i.e.
        ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
        Args:
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.

    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]),
    :param img: RGB, 0~1
    :param target:
    :param mean:
    :param std:
    :return:
    '''
    img = np.array(img, np.float32)
    img = img/255
    for i in range(3):
        img[..., i] = (img[..., i] - mean[i]) / std[i]
    return img, target


def RandomCrop(img,
               target,
               size=768):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        size = size

    w, h = img.size
    th, tw = size
    if w == tw and h == th:
        left_top =[0, 0, th, tw]
    else:
        i = random.randint(0, h - th) # left-top height
        j = random.randint(0, w - tw) # left-top width
        left_top = [i, j, th, tw]

    assert img.size == target.size, 'size of img and lbl should be the same. %s, %s' % (img.size, target.size)

    left, top, right, bottom = left_top[1], left_top[0], left_top[1]+tw, left_top[0]+th
    img = img.crop((left, top, right, bottom))
    target = target.crop((left, top, right, bottom))
    return img, target


class ColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img, target, *args, **kwargs):
        # print('img', np.min(img), np.max(img))
        if self.brightness is not None:
            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            bright_enhancer = ImageEnhance.Brightness(img)
            img = bright_enhancer.enhance(brightness_factor)

        if self.contrast is not None:
            # print('contrast', self.contrast)
            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            contrast_enhancer = ImageEnhance.Contrast(img)
            img = contrast_enhancer.enhance(contrast_factor)

        if self.saturation is not None:
            # print('saturation', self.saturation)
            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            sat_enhancer = ImageEnhance.Color(img)
            img = sat_enhancer.enhance(saturation_factor)

        if self.hue is not None: # not used
            print('hue', self.hue)
            hue_factor = random.uniform(self.hue[0], self.hue[1])
            x = cv2.cvtColor(np.array(img, np.float32) / 255, cv2.COLOR_RGB2HSV)
            x[..., 0] += hue_factor * 360
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[x[:, :, 0] > 360, 0] = 360
            image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
            img = Image.fromarray(np.uint8(image_data)).convert('RGB')

        return img, target

