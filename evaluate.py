# -*- coding: utf-8 -*-
"""
@author:sunwill

本地评价模型
"""

from train import *
from data_process import *

valid_batch_size = 500

model_list = ['deeplabv2_model.h5', 'resnet_model.h5']


def main(args):
    deeplabv1_f1 = []
    resnet_f1 = []
    for model_name in model_list:
        for image_size in [160, 224, 256]:
            if model_name == 'resnet_model.h5' and image_size == 160:continue
            validation_dir = './data_{}/quarterfinals/'.format(image_size)
            validation_data = Dataset_reader(dataset_dir=validation_dir,
                                             file_name='train.txt',
                                             image_size=image_size,
                                             image_channel=image_channel,
                                             label_channel=label_channel
                                             )
            valid_images, valid_annotations = validation_data.get_random_batch(valid_batch_size)
            print valid_images.shape
            print valid_annotations.shape
            if 'deeplab' in model_name:
                model = DeeplabV2(input_shape=(image_size, image_size, image_channel),
                                  classes=label_channel,
                                  weights=None,
                                  )
            else:
                model = make_fcn_resnet(input_shape=(image_size, image_size, image_channel),
                                        nb_labels=label_channel,
                                        use_pretraining=True,
                                        freeze_base=False
                                        )
            adam = keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=adam,
                          metrics=[f1, 'accuracy'])
            if os.path.exists(save_path + model_name):
                model.load_weights(save_path + model_name)
                print 'model restored from ', save_path, model_name
            loss, f1_score, accuracy = model.evaluate(valid_images, valid_annotations, batch_size=4, verbose=1)

            print('validation loss:', loss)
            print('validation f1_score:', f1_score)
            print('validation accuracy:', accuracy)
            if 'deeplab' in model_name:
                deeplabv1_f1.append(f1_score)
            else:
                resnet_f1.append(f1_score)

    print('deeplabv2 model average f1 score: ', np.mean(deeplabv1_f1))
    print('resnet model average f1 score: ', np.mean(resnet_f1))
    print('average f1 score: ', np.mean(deeplabv1_f1+resnet_f1))


if __name__ == '__main__':
    tf.app.run()