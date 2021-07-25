
import os
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from prettytable import PrettyTable
from datasets.cityscapes import Cityscapes
from models.nets import deeplabv3plus_resnet
from metrics.loss import Total_Loss

config = {'batch_size': 1,
          'input_shape': (1024, 2048, 3), # (1024, 2048, 3),
          'num_classes': 20,
          'lr': 5e-3,
          'epochs': 500,
          'backbone': 'resnet18',
          'output_stride': 16,

          # 'model_path': 'logs/resnet18backbone/ep166-loss0.099-val_loss0.280-mean_iounan-pixel_acc0.965-val_mean_iounan-val_pixel_acc0.918.h5',
          'model_path': 'logs/resnet18backbone_os16_adam/ep137-loss0.090-val_loss0.307-pixel_acc0.968-val_pixel_acc0.914.h5',
         }

if __name__ == '__main__':
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)

    # load data 
    train_params = {'root': 'G:\Datasets\cityscapes',
                    'split': 'train',
                    'mode': 'fine',
                    'batch_size': config['batch_size'],
                    'crop_size': config['input_shape'][0],
                    'brightness': 0.5,
                    'contrast': 0.5,
                    'saturation': 0.5,
                    'is_plot': False,
                    'target_type': 'semantic'}

    val_params = {'root': 'G:\Datasets\cityscapes',
                'split': 'val',
                'mode': 'fine',
                'input_size': (config['input_shape'][0], config['input_shape'][1]),
                'batch_size': config['batch_size'],
                'is_plot': False,
                'orignal_size': True,
                'target_type': 'semantic'}

    train_dataset = Cityscapes(**train_params)
    val_dataset = Cityscapes(**val_params)
    print(val_dataset.class_name)

    # model
    inputs = Input(shape=config['input_shape'])
    model = deeplabv3plus_resnet(inputs, use_bn=True, use_bias=False,
                                 num_classes=config['num_classes'],
                                 output_stride=config['output_stride'],
                                 backbone=config['backbone'])
    # model.summary()
    model.load_weights(config['model_path'], by_name=True, skip_mismatch=True)

    # compile
    total_loss = Total_Loss(config['num_classes'], val_dataset.class_name, verbose=False)
    model.compile(loss=total_loss.scc_loss,
                  optimizer=Adam(lr=config['lr']),
                  metrics=[total_loss.pixel_acc])


    # train 
    metrics = model.evaluate(val_dataset, verbose=1)
    print(metrics) # pixel acc: 94.48

    non_ignore_cls = 19

    time_list = []
    zero_m = np.zeros((20, 20))
    Confusion_Metrics = tf.Variable(zero_m, dtype=tf.int32)
    iou_table = PrettyTable()
    iou_table.field_names = ["class index", "class name", "IoU"]
    for val_data in tqdm(val_dataset):
        val_image, val_gt = val_data
        # print('val_image: ', val_image.shape)
        # print('val_ground_truth:', val_gt.shape)

        start_time = time.time()
        y_pred = model.predict(val_image)
        end_time = time.time()
        time_list.append(np.round((end_time-start_time)*1000, 2))
        # print('val_prediction: ', y_pred.shape)

        mask = (val_gt != non_ignore_cls)  # ignore label 255 - also 19
        gt_masked = tf.boolean_mask(val_gt, mask)
        predict_masked = tf.boolean_mask(y_pred, mask)
        predict_masked = tf.maximum(predict_masked, 1e-7)
        # print(predict_masked.shape)

        y_true = tf.reshape(gt_masked, [-1])
        y_pred = tf.reshape(predict_masked, [-1, 20])

        y_true_arg = y_true
        y_pred_arg = tf.argmax(y_pred, 1)

        # confusion matrix
        cm = tf.math.confusion_matrix(y_true_arg, y_pred_arg, num_classes=20) # 20-1
        Confusion_Metrics.assign_add(cm)
        # print('cm: ', Confusion_Metrics.shape)

    unions = []
    intersections = []
    ious = []
    class_names = val_dataset.class_name
    Confusion_Metrics = Confusion_Metrics[:non_ignore_cls, :non_ignore_cls]
    print('cm: ', Confusion_Metrics.shape)
    for i in range(non_ignore_cls):
        # intersection = TP
        # union = (TP + FP + FN)
        inter = Confusion_Metrics[i][i]
        union = tf.subtract(tf.add(tf.reduce_sum(Confusion_Metrics[i]), tf.reduce_sum(Confusion_Metrics[:][i])), Confusion_Metrics[i][i])
        ious.append(tf.divide(inter, union))
        name = class_names[i] if i!=19 else 'others'
        iou_ =  np.round(tf.divide(inter, union) *100, 2)
        iou_table.add_row([i, name, iou_])
    mious =  np.round(tf.reduce_mean(ious).numpy()*100, 2)
    iou_table.add_row([ '' , 'Mean IoU', mious])
    print(iou_table)
    print('Inference time: {} ms'.format(np.mean(time_list)))
