
import os
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from datasets.cityscapes import Cityscapes
from models.nets import deeplabv3plus_resnet50
from metrics.loss import Total_Loss


config = {'batch_size': 16, 
          'input_shape': (768, 768, 3),
          'num_classes': 19,
          'lr': 5e-3,
          'epochs': 500,

          'backbone': 'renset50',
         }



if __name__ == '__main__':
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
                'batch_size': config['batch_size'],
                'is_plot': False,
                'target_type': 'semantic'}

    train_dataset = Cityscapes(**train_params)
    val_dataset = Cityscapes(**val_params)
    print(val_dataset.class_name)

    # model
    inputs = Input(shape=config['input_shape'])
    model = deeplabv3plus_resnet50(inputs, use_bn=True, use_bias=False, 
                                   num_classes=config['num_classes'], 
                                   backbone=config['backbone'])

    # compile
    total_loss = Total_Loss(config['num_classes'], val_dataset.class_name, alpha=0.75, gamma=2.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=config['lr']),
                  metrics=['accuracy',
                           'categorical_crossentropy',
                           total_loss.focal_loss,
                           total_loss.softmax_loss,
                           total_loss.mean_iou,
                           total_loss.pixel_acc])
    
    # callbacks 
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)       
    checkpoint = ModelCheckpoint(os.path.join(log_dir,
                                 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-mean_iou{mean_iou:.3f}-pixel_acc{pixel_acc:.3f}-val_mean_iou{val_mean_iou:.3f}-val_pixel_acc{val_pixel_acc:.3f}.h5'),
                                 monitor='val_loss', save_weights_only=False, save_best_only=True)


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1)

    # train 
    model.fit(train_dataset,
        validation_data=val_dataset,
        epochs=config['epochs'],
        verbose=1,
        callbacks=[checkpoint, reduce_lr])