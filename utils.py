# -*- coding: utf-8 -*-
"""
@author：sunwill
滑动窗口、数据增广
"""
import os
import cv2
import re
import tifffile as tiff
import numpy as np
import matplotlib.image as pltimage

image_size = 256


def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix


## 数据采样：选择标记占20%-70%的样本作为训练集
def data_sample(images, labels, low_rate=0.2, high_rate=0.7):
    new_images = []
    new_labels = []
    assert np.max(labels) == 1.0
    for img, label in zip(images, labels):
        if low_rate <= np.mean(label[:, :, 1]) <= high_rate:
            new_images.append(img)
            new_labels.append(label)
    new_images = np.array(new_images)
    new_labels = np.array(new_labels)

    print 'after sample:', new_images.shape
    print 'after sample:', new_labels.shape

    return new_images, new_labels


# def data_augmentation():
#     training_data = Dataset_reader(dataset_dir=training_dir,
#                                    file_name=train_file,
#                                    image_size=image_size,
#                                    image_channel=image_channel,
#                                    label_channel=label_channel
#                                    )
#     train_images, train_annotations = training_data.get_all_data()
#
#     data_gen_args = dict(rotation_range=0.2,
#                          width_shift_range=0.2,
#                          height_shift_range=0.2,
#                          shear_range=0.2,
#                          zoom_range=0.2,
#                          horizontal_flip=True,
#                          fill_mode='nearest')
#
#     image_datagen = ImageDataGenerator(**data_gen_args)
#     mask_datagen = ImageDataGenerator(**data_gen_args)
#
#     seed = 10
#
#     i = 0
#     for _ in image_datagen.flow(
#             train_images, seed=seed,
#             batch_size=batch_size, save_to_dir='./data_224/images/data_aug/',
#             save_prefix='data_aug', save_format='jpg'):
#         i += 1
#         if i == 100:
#             break
#     i = 0
#     for _ in mask_datagen.flow(
#             train_annotations, seed=seed,
#             batch_size=batch_size, save_to_dir='./data_224/labels/data_aug/',
#             save_prefix='data_aug', save_format='jpg'):
#         i += 1
#         if i == 100:
#             break

## 滑动窗口分割图像
def gen_data_by_slide_window():
    base_dir = './data_{}'.format(image_size)
    img_2015 = tiff.imread('./original_data/quickbird2015.tif').transpose([1, 2, 0]) ## shape=(5106,15106,4)
    img_2017 = tiff.imread('./original_data/quickbird2017.tif').transpose([1, 2, 0])

    img_2015 = img_2015[:, :14400, :]
    img_2017 = img_2017[:, :14400, :]

    label_2015 = pltimage.imread(base_dir+'/2015.jpg') ## shape=(5106,14400,3)
    label_2017 = pltimage.imread(base_dir+'/2017.jpg')

    if image_size == 160:
        x_step = 120
        y_step = 120
        # 14400-160 = 14240 = 118*120+80
        # 5106-160 = 4946 = 41*120+26
        for i in range(42):
            for j in range(119):
                im_name = str(i) + '_' + str(j) + '_' + str(image_size) + '_.jpg'
                cv2.imwrite(base_dir+'/slide_window/images/2015/' + im_name, scale_percentile(
                    img_2015[i * y_step:i * y_step + image_size, j * x_step:j * x_step + image_size, :3]) * 255)
                cv2.imwrite(base_dir+'/slide_window/images/2017/' + im_name, scale_percentile(
                    img_2017[i * y_step:i * y_step + image_size, j * x_step:j * x_step + image_size, :3]) * 255)
                cv2.imwrite(base_dir+'/slide_window/labels/2015/' + im_name, scale_percentile(
                    label_2015[i * y_step:i * y_step + image_size, j * x_step:j * x_step + image_size, :3]) * 255)
                cv2.imwrite(base_dir+'/slide_window/labels/2017/' + im_name, scale_percentile(
                    label_2017[i * y_step:i * y_step + image_size, j * x_step:j * x_step + image_size, :3]) * 255)

    elif image_size == 224:
        x_step = 194
        y_step = 194
        # 14400-224 = 14176 = 73*194+14
        # 5106-224 = 4882 = 25*194+32
        for i in range(26):
            for j in range(74):
                im_name = str(i) + '_' + str(j) + '_' + str(image_size) + '_.jpg'
                cv2.imwrite(base_dir+'/slide_window/images/2015/' + im_name, scale_percentile(
                    img_2015[i * y_step:i * y_step + image_size, j * x_step:j * x_step + image_size, :3]) * 255)
                cv2.imwrite(base_dir+'/slide_window/images/2017/' + im_name, scale_percentile(
                    img_2017[i * y_step:i * y_step + image_size, j * x_step:j * x_step + image_size, :3]) * 255)
                cv2.imwrite(base_dir+'/slide_window/labels/2015/' + im_name, scale_percentile(
                    label_2015[i * y_step:i * y_step + image_size, j * x_step:j * x_step + image_size, :3]) * 255)
                cv2.imwrite(base_dir+'/slide_window/labels/2017/' + im_name, scale_percentile(
                    label_2017[i * y_step:i * y_step + image_size, j * x_step:j * x_step + image_size, :3]) * 255)
    elif image_size == 256:
        x_step = 224
        y_step = 224
        # 14400-256 = 14144 = 63*224+32
        # 5106-256 = 4856 = 21*224*152
        for i in range(22):
            for j in range(64):
                im_name = str(i) + '_' + str(j) + '_' + str(image_size) + '_.jpg'
                cv2.imwrite(base_dir+'/slide_window/images/2015/' + im_name, scale_percentile(
                    img_2015[i * y_step:i * y_step + image_size, j * x_step:j * x_step + image_size, :3]) * 255)
                cv2.imwrite(base_dir+'/slide_window/images/2017/' + im_name, scale_percentile(
                    img_2017[i * y_step:i * y_step + image_size, j * x_step:j * x_step + image_size, :3]) * 255)
                cv2.imwrite(base_dir+'/slide_window/labels/2015/' + im_name, scale_percentile(
                    label_2015[i * y_step:i * y_step + image_size, j * x_step:j * x_step + image_size, :3]) * 255)
                cv2.imwrite(base_dir+'/slide_window/labels/2017/' + im_name, scale_percentile(
                    label_2017[i * y_step:i * y_step + image_size, j * x_step:j * x_step + image_size, :3]) * 255)


if __name__ == "__main__":
    # data_augmentation()
    # x = np.array(os.listdir('./data_224/images/data_aug/'))
    # y = np.array(os.listdir('./data_224/labels/data_aug/'))
    # print x.shape
    # print y.shape
    # assert x.shape == y.shape
    # with open('./data_224/train.txt', 'a') as f:
    #     for item in x:
    #         f.write('data_aug/'+item+'\n')
    gen_data_by_slide_window()
    to_dir = './data_{}/slide_window/'.format(image_size)
    images_list_2015 = np.array(os.listdir(to_dir+'images/2015/'))
    label_list_2015 = np.array(os.listdir(to_dir+'labels/2015/'))
    images_list_2017 = np.array(os.listdir(to_dir+'images/2017/'))
    label_list_2017 = np.array(os.listdir(to_dir+'labels/2017/'))

    common_2015 = np.intersect1d(images_list_2015, label_list_2015)
    common_2017 = np.intersect1d(images_list_2017, label_list_2017)

    _2015 = ['2015/' + i for i in common_2015]
    _2017 = ['2017/' + i for i in common_2017]
    common = np.hstack([_2015, _2017])
    print common.shape[0]
    perm = np.arange(common.shape[0])

    np.random.shuffle(perm)

    common = common[perm]
    train = common
    print train.shape
    valid = common[int(common.shape[0]*0.8):]
    print valid.shape
    reg = r'([0-9]{4})\/[0-9]{0,3}_[0-9]{0,3}_[0-9]{3}_.jpg'
    with open(to_dir+'train.txt', 'w') as f:
        for line in train:
            if re.match(reg, line):
                f.write(line+'\n')

    with open(to_dir+'valid.txt', 'w') as f:
        for line in valid:
            if re.match(reg, line):
                f.write(line+'\n')

