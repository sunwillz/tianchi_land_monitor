# -*- coding: utf-8 -*-

from train import *
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Permute, Reshape


def data_augmentation():
    training_data = Dataset_reader(dataset_dir=training_dir,
                                   file_name=train_file,
                                   image_size=image_size,
                                   image_channel=image_channel,
                                   label_channel=label_channel
                                   )
    train_images, train_annotations = training_data.get_all_data()

    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 10

    i = 0
    for _ in image_datagen.flow(
            train_images, seed=seed,
            batch_size=batch_size, save_to_dir='./data_224/images/data_aug/',
            save_prefix='data_aug', save_format='jpg'):
        i += 1
        if i == 100:
            break
    i = 0
    for _ in mask_datagen.flow(
            train_annotations, seed=seed,
            batch_size=batch_size, save_to_dir='./data_224/labels/data_aug/',
            save_prefix='data_aug', save_format='jpg'):
        i += 1
        if i == 100:
            break


if __name__ == "__main__":
    # data_augmentation()
    x = np.array(os.listdir('./data_224/images/data_aug/'))
    y = np.array(os.listdir('./data_224/labels/data_aug/'))
    print x.shape
    print y.shape
    assert x.shape == y.shape
    with open('./data_224/train.txt', 'a') as f:
        for item in x:
            f.write('data_aug/'+item+'\n')
