# -*- coding: utf-8 -*-
# @Time    : 2020/11/16 5:53
# @Author  : Zeqi@@
# @FileName: tf_metrics.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU


# class_name = index_to_name()
# n_class = len(class_name)
# print(class_name, n_class)

'''
https://blog.csdn.net/wangdongwei0/article/details/84576044
'''

n_class = 19

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
    # batch_size, h, w = y_true.shape
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, n_class])

    y_true_arg = y_true
    # y_true_arg[y_true_arg==255] = 19  # problem replace 255 by 19
    y_pred_arg = tf.argmax(y_pred, 1)

    # mask = y_true_arg != 255  # ignore label 255
    gt_masked = y_true_arg
    predict_masked = y_pred_arg
    # gt_masked = tf.boolean_mask(y_true_arg, mask)
    # predict_masked = tf.boolean_mask(y_pred_arg, mask)

    # print('gt', gt_masked.shape)
    # print('pre', predict_masked.shape)

    # confusion matrix
    cm = tf.math.confusion_matrix(gt_masked, predict_masked, num_classes=(n_class))
    # cm = tf.math.confusion_matrix(gt_masked, predict_masked)
    # confusion matrix: (20, 20)
    return cm


class Total_Loss():
    def __init__(self, n_class, class_names, alpha=0.25, gamma=2.0):
        self.epsilon = 1.e-7
        self.alpha = alpha
        self.gamma = gamma
        self.class_names = class_names
        self.n_class = n_class

    def create_onehot_encoding(self, label, n_class=20):
        h, w = np.shape(label)
        target = np.zeros((n_class, h, w))
        for c in range(n_class):
            target[c][label == c] = 1
        target[19][label == 255] = 1
        target = np.transpose(target, (1, 2, 0))
        return target

    def scc_loss(self, y_true, y_pred):
        '''
        Mask the pixels in the loss where y_true == 255 (ignore)
        :param y_true:
        :param y_pred:
        :return:
        '''
        mask = y_true != 19  # ignore label 255 - also 19
        gt_masked = tf.boolean_mask(y_true, mask)
        predict_masked = tf.boolean_mask(y_pred, mask)
        predict_masked = tf.maximum(predict_masked, self.epsilon)
        # tf.print('gt', gt_masked.shape)
        # tf.print('pre', predict_masked.shape)
        scc = tf.keras.losses.SparseCategoricalCrossentropy()
        return scc(gt_masked, predict_masked)


    def softmax_loss(self, y_true, y_pred):
        '''
        Mask the pixels in the loss where y_true == 255 (ignore)
        :param y_true:
        :param y_pred:
        :return:
        '''
        mask = y_true != 19  # ignore label 255
        gt_masked = tf.boolean_mask(y_true, mask)
        gt_masked = self.create_onehot_encoding(gt_masked, self.n_class)
        predict_masked = tf.boolean_mask(y_pred, mask)
        predict_masked = tf.maximum(predict_masked, self.epsilon)
        softmax_loss = -tf.reduce_sum(gt_masked * tf.math.log(predict_masked), axis=-1)
        return softmax_loss

    def miou(self, y_true, y_pred):
        '''
        miou should be global

        :param y_true:  (batch_size, 768, 768)
        :param y_pred:  (batch_size, 768, 768, n_class)
        :return:
        '''

        #batch_size, h, w = y_true.shape
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, self.n_class])

        y_true_arg = y_true
        y_pred_arg = tf.argmax(y_pred, 1)

        mask = y_true_arg != 19  # ignore label 255
        gt_masked = tf.boolean_mask(y_true_arg, mask)
        predict_masked = tf.boolean_mask(y_pred_arg, mask)

        m = MeanIoU(num_classes=(self.n_class-1))
        m.update_state(gt_masked, predict_masked)
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

        non_ignored_cls = self.n_class - 1
        cm = confusion_matrix(y_true, y_pred, self.n_class)
        cm = cm[:non_ignored_cls, :non_ignored_cls]
        # tf.print('confusion metrics: ', cm.shape)

        unions        = []
        intersections = []
        ious          = []
        self.miou_dir = {}
        for i in range(non_ignored_cls):
            # intersection = TP
            # union = (TP + FP + FN)
            inter = cm[i][i]
            union = tf.subtract(tf.add(tf.reduce_sum(cm[i]), tf.reduce_sum(cm[:][i])),cm[i][i])
            intersections.append(inter)
            unions.append(union)
            ious.append(tf.divide(inter, union))

        # In case uions = 0
        # nombre_val      = tf.cast(tf.math.count_nonzero(unions), dtype=tf.float64)
        # non_zero_unions = tf.where(tf.not_equal(unions, 0), unions, 1)
        # ious            = tf.divide(intersections, non_zero_unions)
        # sum_ious        = tf.reduce_sum(ious)
        # mean            = tf.divide(sum_ious, nombre_val)

        ious_real       = [tf.divide(inter, union) for inter, union in zip(intersections, unions)]
        non_nan_ious    = [ 0 if tf.math.isnan(iou) else iou for iou in ious_real]
        mean            = tf.reduce_mean(ious_real)
        non_nan_mious   = tf.reduce_mean(non_nan_ious)

        for i in range(self.n_class-1):
            name = self.class_names[i]
            # print(i, name)
            self.miou_dir[name] = ious_real[i]
            # tf.print(i, name, self.miou_dir[name])
        # tf.print('\n', miou_dir)
        return non_nan_mious

    def pixel_acc(self, y_true, y_pred):
        non_ignored_cls = self.n_class-1
        cm   = confusion_matrix(y_true, y_pred, self.n_class)
        cm   = cm[:non_ignored_cls, :non_ignored_cls]
        diag = [cm[i][i] for i in range(non_ignored_cls)]  # ignore class 19, which originally to be 255
        acc  = tf.divide(tf.reduce_sum(diag), tf.reduce_sum(cm))
        return acc
