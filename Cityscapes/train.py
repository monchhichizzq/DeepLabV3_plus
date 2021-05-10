# -*- coding: utf-8 -*-
# @Time    : 2021/5/10 0:04
# @Author  : Zeqi@@
# @FileName: train.py.py
# @Software: PyCharm

import os
from models.deeplab import Deeplabv3
from preprocess.tf_dataloader_json import CityScape_DataGenerator
from preparation.semantic_color2index import index_to_name
from loss.tf_metrics_onehot import Total_Loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard)


class_name = index_to_name()

input_shape = (512, 1024, 3)
root_dir = "G:\Datasets\cityscapes"
txt_dir = 'train_test_inputs'
train_file = os.path.join(txt_dir, "Cityscape_eigen_train_files.txt")
val_file = os.path.join(txt_dir, "Cityscape_eigen_val_files.txt")

# Hyperparameters
lr = 3e-4
batch_size = 32
num_classes = 34
scale_factor = 8
epochs = 500
visual = False

train_params = {'mode': 'train',
                'txt_file': train_file,
                'batch_size': batch_size,
                'n_class': num_classes,
                'crop': False,
                'crop_dim': (input_shape[0],
                             input_shape[1]),
                'filp_rate': 0.5,
                'scale_factor': scale_factor,
                'visual': visual,
                'shuffle': True}

val_params = {'mode': 'val',
              'txt_file': val_file,
              'batch_size': batch_size,
              'n_class': num_classes,
              'crop': False,
              'crop_dim': (input_shape[0],
                            input_shape[1]),
              'filp_rate': 0,
              'scale_factor': scale_factor,
              'visual': visual,
              'shuffle': False}


if __name__ == "__main__":
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    model = Deeplabv3(input_shape=(int(input_shape[0]/scale_factor*2), int(input_shape[1]/scale_factor*2), 3), classes=num_classes, alpha=1.)
    model.summary()

    train_generator = CityScape_DataGenerator(**train_params)
    val_generator = CityScape_DataGenerator(**val_params)

    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-mean_iou{mean_iou:.3f}-pixel_acc{pixel_acc:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    total_loss = Total_Loss(num_classes, class_name, alpha=0.75, gamma=2.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy',
                           'categorical_crossentropy',
                           total_loss.focal_loss,
                           total_loss.softmax_loss,
                           total_loss.mean_iou,
                           total_loss.pixel_acc])

    model.fit(train_generator,
            validation_data=val_generator,
            epochs=epochs,
            verbose=2,
            callbacks=[checkpoint, reduce_lr])
