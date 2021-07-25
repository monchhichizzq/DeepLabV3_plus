
import os
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from datasets.cityscapes import Cityscapes
from models.nets import deeplabv3plus_resnet
from metrics.loss import Total_Loss

config = {'batch_size': 8,
          'input_shape': (768, 768, 3),
          'num_classes': 20,
          'lr': 5e-3,
          'epochs': 500,
          'backbone': 'resnet18',
          'output_stride': 16,
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
                 'orignal_size': False,
                 'input_size': (config['input_shape'][0], config['input_shape'][1]),
                 'batch_size': config['batch_size'],
                 'is_plot': False,
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

    # compile
    total_loss = Total_Loss(config['num_classes'], val_dataset.class_name, verbose=False)
    model.compile(loss=total_loss.scc_loss,
                  optimizer=Adam(lr=config['lr']),
                  # optimizer=SGD(learning_rate=config['lr'], momentum=0.9),
                  metrics=[total_loss.pixel_acc])
    
    # callbacks 
    log_dir = "logs/resnet18backbone_os16_adam"
    os.makedirs(log_dir, exist_ok=True)       
    checkpoint = ModelCheckpoint(os.path.join(log_dir,
                                              # 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-pixel_acc{pixel_acc:.3f}-val_pixel_acc{val_pixel_acc:.3f}.h5'),
                                 monitor='val_loss', save_weights_only=False, save_best_only=True, verbose=1)


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1)

    # train 
    model.fit(train_dataset,
        validation_data=val_dataset,
        epochs=config['epochs'],
        verbose=2,
        callbacks=[checkpoint, reduce_lr])