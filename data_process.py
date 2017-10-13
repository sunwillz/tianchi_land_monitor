# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tifffile as tiff
import matplotlib.image as pltimage
from sklearn import preprocessing
from PIL import Image
from keras.utils.np_utils import to_categorical
from scipy import misc
import re

class Dataset_reader:

    images = []
    labels = []
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, dataset_dir=None, file_name=None, image_size=224, image_channel=3, label_channel=2, test=False):
        self.dataset_dir = dataset_dir
        self.filename = file_name
        self.images = []
        self.labels = []
        self.image_size = image_size
        self.image_channel = image_channel
        self.label_channel = label_channel
        self.read_images()
        if not test: #预测阶段不读取标签
            self.read_labels()

    ## 224*224
    def read_images(self):
        with open(os.path.join(self.dataset_dir, self.filename)) as f:
            images = f.readlines()
            images_list = [i.strip() for i in images]
        for image in images_list:
            img = pltimage.imread(os.path.join(self.dataset_dir, 'images/' + image))
            # 图像中心化
            img = scale_percentile(img)
            img = img.reshape([self.image_size, self.image_size, self.image_channel])
            self.images.append(img)
        self.images = np.array(self.images)
        print self.images.shape

    def read_labels(self):
        with open(os.path.join(self.dataset_dir, self.filename)) as f:
            images = f.readlines()
            images_list = [i.strip() for i in images]
        for image in images_list:
            label = pltimage.imread(os.path.join(self.dataset_dir, 'labels/' +image))
            label = (label[:, :, 0] > 0).astype(np.uint8) # 取其中一个通道
            label = to_categorical(label, num_classes=self.label_channel)
            label = label.reshape([self.image_size, self.image_size, self.label_channel])
            self.labels.append(label)
        self.labels = np.array(self.labels)
        print self.labels.shape

    ## 960*960
    # def read_images(self):
    #     file_list = os.listdir(os.path.join(self.dataset_dir, 'images/'))
    #     for filename in file_list: # 2015,2017
    #         if not os.path.isdir(os.path.join(self.dataset_dir, 'images/'+filename)):continue
    #         images_list = os.listdir(os.path.join(self.dataset_dir, 'images/'+filename))
    #         for image in images_list:
    #             img = pltimage.imread(os.path.join(self.dataset_dir, 'images/' + filename+'/' + image))
    #             img = scale_percentile(img)
    #             img = img.reshape([self.image_size, self.image_size, self.image_channel])
    #             self.images.append(img)
    #     self.images = np.array(self.images)
    #     print self.images.shape
    #
    # def read_labels(self):
    #     file_list = os.listdir(os.path.join(self.dataset_dir, 'labels/'))
    #     for filename in file_list:  # 2015,2017
    #         if not os.path.isdir(os.path.join(self.dataset_dir, 'labels/' + filename)):continue
    #         images_list = os.listdir(os.path.join(self.dataset_dir, 'labels/' + filename))
    #         for image in images_list:
    #             label = pltimage.imread(os.path.join(self.dataset_dir, 'labels/' + filename+'/'+image))
    #             label = (label[:, :, 0] > 0).astype(np.uint8) # 取其中一个通道
    #             label = to_categorical(label, num_classes=2)
    #             label = label.reshape([self.image_size, self.image_size, self.label_channel])
    #             self.labels.append(label)
    #     self.labels = np.array(self.labels)
    #     print self.labels.shape


    def normlization(self, x): # 对其中一个通道做图像归一化
        x_scaled = np.empty(shape=x.shape)
        # x_scaled[:, :, 0] = preprocessing.normalize(x[:, :, 0], norm='l2')
        # x_scaled[:, :, 1] = preprocessing.normalize(x[:, :, 1], norm='l2')
        # x_scaled[:, :, 2] = preprocessing.normalize(x[:, :, 2], norm='l2')

        scaler = preprocessing.MinMaxScaler()
        x_scaled[:, :, 0] = scaler.fit_transform(x[:, :, 0])
        x_scaled[:, :, 1] = scaler.fit_transform(x[:, :, 1])
        x_scaled[:, :, 2] = scaler.fit_transform(x[:, :, 2])

        return x_scaled

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            self.epochs_completed += 1
            print '-----------------Epochs completed: ' + str(self.epochs_completed) + '  ---------------------'
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset

        return self.images[start:end], self.labels[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.labels[indexes]

    def get_all_data(self, label=True):
        if label:
            return self.images, self.labels
        else:
            return self.images


def load_testing_data(file_name):

    return tiff.imread(file_name).transpose([1, 2, 0])


# 这个函数将小的三通道JPG图片拼接成大的图片
def concat_jpg_to_largefile(image_dir, to_dir, to_name, flag=False):
    if os.path.exists(os.path.join(to_dir, to_name)):
        return pltimage.imread(os.path.join(to_dir, to_name))


    images_list = os.listdir(image_dir)

    rows = 0
    cols = 0
    for img in images_list:
        rows = max(rows, int(img.split("_")[0])+1)
        cols = max(cols, int(img.split("_")[1])+1)

    little_image_width = 960
    little_image_height = 960
    little_image_channel = 3
    width = little_image_width*cols #大图片的宽度
    height = little_image_height*rows #大图片的高度
    channel = little_image_channel #大图片的通道数
    toarray = np.empty(shape=(height, width, channel))

    for i in range(rows):
        for j in range(cols):
            fname = '{}_{}_{}_.jpg'.format(i, j, little_image_width)
            fromImage = Image.open(os.path.join(image_dir, fname))
            fromImage = np.array(fromImage)
            toarray[i*little_image_width:i*little_image_width+fromImage.shape[0], j*little_image_height:j*little_image_height+fromImage.shape[1]:] = fromImage
    ## 将一部分的标注数据和2017年的标注图像结合
    toarray = toarray[:5106, :, :]
    if flag:
        biaozhu = tiff.imread('../land/data/result/submit.tiff') #shape=(5106,15106)
        toarray = toarray[:, :, 0]
        biaozhu = biaozhu[:5106, :14400]
        ret = ((toarray > 0) | (biaozhu > 1)).astype(np.uint8)
        ret = ret.reshape([5106, 14400, 1])
        toarray = np.repeat(ret, 3, axis=2)
    misc.imsave(os.path.join(to_dir, to_name), toarray)
    return toarray


def scale_percentile(matrix):
    """
    图像中心化
    :param matrix:
    :return:
    """

    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix


def split_tiff_file(img, to_dir):
    """
    将大的高分辨率卫星图像分割成224*224的小图片,同样的区域命名相同，分别放在不同文件夹下
    :return:
    """
    img_size = 224  # 15106/ 224 =67...98  5106/224=22..178
    for i in range(len(img) / img_size):
        for j in range(len(img[0]) / img_size):
            im_name = str(i) + '_' + str(j) + '_' + str(img_size) + '_.jpg'
            cv2.imwrite(to_dir + im_name, scale_percentile(
                img[i * img_size:i * img_size + img_size, j * img_size:j * img_size + img_size, :3]) * 255)


## 数据增广：采用重叠滑动窗口分割大图片，重叠区域大小为30*40
def split_tiff_file_overlap_window(img, to_dir):
    """
    224-30 = 194，224-40 = 184
    14400-224 = 14176= 77*184+8
    5106-224 = 4882 = 25*194+32
    经过计算，去掉余数，每个大图片可分为26列，78行，26*78=2028个小图片
    :param img:
    :param to_dir:
    :return:
    """
    img_size = 224
    x_step = 184
    y_step = 194
    for i in range(26):
        for j in range(78):
            im_name = str(i) + '_' + str(j) + '_' + str(img_size) + '_.jpg'
            cv2.imwrite(to_dir + im_name, scale_percentile(
                img[i * y_step:i * y_step + img_size, j * x_step:j * x_step + img_size, :3]) * 255)


if __name__ == '__main__':
    ## 将(960,960,3)的小图片拼接成(5106,14400,3)的大图片
    image_2015 = concat_jpg_to_largefile('./label/2015/', './data_224/', '2015.jpg')
    print image_2015.shape ##(5106, 14400, 3),image_2015.max()

    image_2017 = concat_jpg_to_largefile('./label/2017/', './data_224/', '2017_with_biaozhu.jpg', flag=True)
    print image_2017.shape, image_2017.max()
    assert image_2015.shape == image_2017.shape

    file_name = '../land/data/preliminary/quickbird2015.tif'
    im_2015 = load_testing_data(file_name)
    file_name = '../land/data/preliminary/quickbird2017.tif'
    im_2017 = load_testing_data(file_name)

    split_tiff_file_overlap_window(im_2015[:, :14400, :], './data_224/images/2015/')
    split_tiff_file_overlap_window(im_2017[:, :14400, :], './data_224/images/2017/')
    split_tiff_file_overlap_window(image_2015, './data_224/labels/2015/')
    split_tiff_file_overlap_window(image_2017, './data_224/labels/2017/')

    ## 得到公共的图片
    images_list_2015 = np.array(os.listdir('./data_224/images/2015/'))
    label_list_2015 = np.array(os.listdir('./data_224/labels/2015/'))
    images_list_2017 = np.array(os.listdir('./data_224/images/2017/'))
    label_list_2017 = np.array(os.listdir('./data_224/labels/2017/'))

    common_2015 = np.intersect1d(images_list_2015, label_list_2015)
    common_2017 = np.intersect1d(images_list_2017, label_list_2017)

    _2015 = ['2015/' + i for i in common_2015]
    _2017 = ['2017/' + i for i in common_2017]
    common = np.hstack([_2015, _2017])
    print common.shape[0]
    perm = np.arange(common.shape[0])

    np.random.shuffle(perm)

    common = common[perm]
    train = common[:int(common.shape[0]*0.85)]
    print train.shape
    valid = common[int(common.shape[0]*0.85):]
    print valid.shape
    reg = r'([0-9]{4})\/[0-9]{0,2}_[0-9]{0,2}_[0-9]{3}_.jpg'
    with open('./data_224/train.txt', 'w') as f:
        for line in train:
            if re.match(reg, line):
                f.write(line+'\n')

    with open('./data_224/valid.txt', 'w') as f:
        for line in valid:
            if re.match(reg, line):
                f.write(line+'\n')
