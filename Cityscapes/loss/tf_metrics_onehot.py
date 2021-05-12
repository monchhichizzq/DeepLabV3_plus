# -*- coding: utf-8 -*-
# @Time    : 2020/11/16 5:53
# @Author  : Zeqi@@
# @FileName: tf_metrics.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss, mean_squared_error
from preparation.semantic_color2index import index_to_name
from tensorflow.keras.metrics import MeanIoU


# class_name = index_to_name()
# n_class = len(class_name)
# print(class_name, n_class)

'''
https://blog.csdn.net/wangdongwei0/article/details/84576044
'''

def semantic_loss(n_class):
    '''
        https://www.jeremyjordan.me/semantic-segmentation/#loss/ https://zhuanlan.zhihu.com/p/101773544
        https://zhuanlan.zhihu.com/p/103426335

        The most commonly used loss function for the task of image segmentation is a pixel-wise corss entropy loss
        This loss examines each pixel individually, comparing the class predictions (depth-wise pixel vector) to our
        one-hot encoded target vector.
        Because the cross entropy loss evaluates the class predictions for each pixel vector individually and then
        averages over all pixels,we're essentially asserting equal learning to each pixel in the image.
        This can be a problem if your various classes have unbalanced representation in the image, as training can be
        dominated by the most prevalent class.

    '''

    def focalloss(y_true, y_pred):
        '''
           https://zhuanlan.zhihu.com/p/103426335
           To solve the imbalance between the dfficult labels and easy labels in the datasets

           alphas: To balance the amount of positive and negative samples (0.75)
           gamma:  To focus on the diffcult samples in the dataset (2)
           Focal Loss: only used for binary classification

           logits:     outputs of network (batch_size, class)
           prob:       softmax outputs of network (batch_size, class)
           prediction: argmax outputs of network (batch_size, 1)

           one hot:    (batch_size, class)
           label:      (batch_size, 1)

           # 注意，alpha是一个和你的分类类别数量相等的向量；
            alpha = [[1], [1], [1], [1]]

            Args:
                :param logits:  [batch_size, n_class]
                :param labels: [batch_size]  not one-hot !!!
            return:
                -alpha*(1-y)^r * log(y)
        '''

        epsilon = 1.e-7
        alpha = 0.75
        gamma = 2.0

        # (N, h, w, class) -> (N, class)
        y_true_pixels = tf.reshape(y_true, [-1, n_class])
        y_pred_pixels = tf.reshape(y_pred, [-1, n_class])
        alpha = tf.constant(alpha, dtype=tf.float32)

        # (N, class) -> (N, class)
        softmax_output = tf.nn.softmax(y_pred_pixels)

        # cross-entropy
        # loss = tf.reduce_sum(labels * log_p, axis=-1)
        # ce = tf.reduce_mean(loss)
        log_p = -tf.math.log(tf.clip_by_value(softmax_output, epsilon, 1 - epsilon))
        one_tensor = tf.ones(tf.shape(y_true_pixels))
        pt = tf.reduce_mean((one_tensor - softmax_output)*y_true_pixels, axis=-1)
        fc_loss = tf.reduce_mean(tf.reduce_sum(pt ** gamma * alpha * log_p, axis=-1))

        return fc_loss

    return focalloss

def confusion_matrix(y_true, y_pred, n_class):
    # (N, h, w, class) -> (N, class)
    # tf.print('1', y_pred.shape, tf.reduce_max(y_pred))
    y_true_pixels = tf.reshape(y_true, [-1, n_class])
    y_pred_pixels = tf.reshape(y_pred, [-1, n_class])
    # tf.print('2', y_pred_pixels.shape, tf.reduce_max(y_pred_pixels))

    # (N, class) -> (N, )
    y_true_arg = tf.argmax(y_true_pixels, 1)
    y_pred_arg = tf.argmax(y_pred_pixels, 1)
    # tf.print('3', y_pred_arg.shape, tf.reduce_max(y_pred_arg))

    # confusion matrix
    cm = tf.math.confusion_matrix(y_true_arg, y_pred_arg, num_classes=n_class)
    # confusion matrix: (20, 20)
    return cm


def Pixel_Acc():

    def pixel_acc(y_true, y_pred):
        cm   = confusion_matrix(y_true, y_pred, n_class)
        diag = [cm[i][i] for i in range(n_class)]
        acc  = tf.divide(tf.reduce_sum(diag), tf.reduce_sum(cm))
        return acc

    return pixel_acc


def MeanIOU():
    '''
    IOU class:  Intersection over uion for each class  IoU=TP / (TP+FP+FN)
    iIOU class: Instance Intersection over Union  iIoU = iTP / (iTP + FP + iFN)
    '''


    def mean_iou(y_true, y_pred):
        """  compute the value of mean iou
        :param pred:  2d array, int, prediction, (N, h, w, c)
        :param gt: 2d array, int, ground truth  (N, h, w, c)
        :return:
            miou: float, the value of miou
        """

        cm = confusion_matrix(y_true, y_pred, n_class)

        unions        = []
        intersections = []
        for i in range(n_class):
            # intersection = TP
            # union = (TP + FP + FN)
            inter = cm[i][i]
            union = tf.subtract(tf.add(tf.reduce_sum(cm[i]), tf.reduce_sum(cm[:][i])),cm[i][i])
            intersections.append(inter)
            unions.append(union)
             # ious.append(tf.divide(intersection, union))
        nombre_val      = tf.cast(tf.math.count_nonzero(unions), dtype=tf.float64)
        non_zero_unions = tf.where(tf.not_equal(unions, 0), unions, 1)
        ious            = tf.divide(intersections, non_zero_unions)
        sum_ious        = tf.reduce_sum(ious)
        mean            = tf.divide(sum_ious, nombre_val)
        return mean

    return mean_iou

def CELoss():

    def cross_entropy(y_true, y_pred):
        softmax_ce_with_logits = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        loss                   = tf.reduce_mean(softmax_ce_with_logits)
        return loss

    return cross_entropy


class Total_Loss():
    def __init__(self, n_class, class_names, alpha=0.25, gamma=2.0):
        self.epsilon = 1.e-7
        self.alpha = alpha
        self.gamma = gamma
        self.class_names = class_names
        self.n_class = n_class

    def softmax_loss(self, y_true, y_pred):
        y_pred = tf.maximum(y_pred, self.epsilon)
        softmax_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        return softmax_loss

    def focal_loss(self, y_true, y_pred):
        y_pred = tf.maximum(y_pred, self.epsilon)
        ones = tf.ones_like(y_true)
        # alpha should be a step function as paper mentioned, but ut doesn't matter
        alpha_t = tf.where(tf.equal(y_true,1), self.alpha*ones, 1-self.alpha*ones)
        focal_loss = -tf.reduce_sum(y_true * alpha_t * (1 - y_pred) ** self.gamma * tf.math.log(y_pred), axis=-1)
        return focal_loss

    def miou(self, y_true, y_pred):
        y_true_arg = tf.argmax(y_true, 1)
        y_pred_arg = tf.argmax(y_pred, 1)
        m = MeanIoU(num_classes=self.n_class)
        m.update_state(y_true_arg, y_pred_arg)
        mious = m.result()
        m.reset_states()
        return mious

    def mean_iou(self, y_true, y_pred):
        """  compute the value of mean iou
        :param pred:  2d array, int, prediction, (N, h, w, c)
        :param gt: 2d array, int, ground truth  (N, h, w, c)
        :return:
            miou: float, the value of miou
        """

        cm = confusion_matrix(y_true, y_pred, self.n_class)

        unions        = []
        intersections = []
        ious          = []
        self.miou_dir = {}
        for i in range(self.n_class):
            # intersection = TP
            # union = (TP + FP + FN)
            inter = cm[i][i]
            union = tf.subtract(tf.add(tf.reduce_sum(cm[i]), tf.reduce_sum(cm[:][i])),cm[i][i])
            intersections.append(inter)
            unions.append(union)
            ious.append(tf.divide(inter, union))

        # In case uions = 0
        nombre_val      = tf.cast(tf.math.count_nonzero(unions), dtype=tf.float64)
        non_zero_unions = tf.where(tf.not_equal(unions, 0), unions, 1)
        ious            = tf.divide(intersections, non_zero_unions)
        sum_ious        = tf.reduce_sum(ious)
        mean            = tf.divide(sum_ious, nombre_val)

        ious_real       = [tf.divide(inter, union) for inter, union in zip(intersections, unions)]


        for i in range(self.n_class):
            # print(i)
            name = self.class_names[i]
            # print(i, name)
            self.miou_dir[name] = ious_real[i]
            # tf.print(name, self.miou_dir[name])
        # tf.print('\n', miou_dir)
        return mean

    def pixel_acc(self, y_true, y_pred):
        cm   = confusion_matrix(y_true, y_pred, self.n_class)
        diag = [cm[i][i] for i in range(self.n_class)]
        acc  = tf.divide(tf.reduce_sum(diag), tf.reduce_sum(cm))
        return acc
