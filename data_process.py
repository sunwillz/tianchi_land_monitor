# -*- coding: utf-8 -*-
"""
@author:sunwill

数据预处理：图片生成、分割、拼接、读取
"""

from utils import *
import matplotlib.image as pltimage
from sklearn import preprocessing
from PIL import Image
from keras.utils.np_utils import to_categorical
from scipy import misc
import re
from PIL import ImageEnhance

image_size = 256  # 输入图像尺寸大小
image_channel = 3  # 输入图像通道数
label_size = image_size  # 输出图像尺寸大小
label_channel = 2  # 输出图像通道数
n_classes = 2


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

    def read_images(self):
        with open(os.path.join(self.dataset_dir, self.filename)) as f:
            images = f.readlines()
            images_list = [i.strip() for i in images]
        for image in images_list:
            img = pltimage.imread(os.path.join(self.dataset_dir, 'images/' + image))
            img = scale_percentile(img)
            img_arr = np.zeros(shape=[self.image_size, self.image_size, self.image_channel])
            img_arr[:img.shape[0], :img.shape[1], :img.shape[2]] = img
            self.images.append(img_arr)
        self.images = np.array(self.images)
        # np.save(str(image_size)+'.npy', self.images)
        # self.images = np.load(str(image_size)+'.npy')
        print self.images.shape

    def read_labels(self):
        with open(os.path.join(self.dataset_dir, self.filename)) as f:
            images = f.readlines()
            images_list = [i.strip() for i in images]
        for image in images_list:
            label = pltimage.imread(os.path.join(self.dataset_dir, 'labels/' +image))
            if len(label.shape) == 3:
                label = (label[:, :, 0] > 0).astype(np.uint8) # 取其中一个通道
            else:
                label = (label > 0).astype(np.uint8)
            label = to_categorical(label, num_classes=self.label_channel)
            label = label.reshape([self.image_size, self.image_size, self.label_channel])
            self.labels.append(label)
        self.labels = np.array(self.labels)
        print self.labels.shape

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
        self.images, self.labels = data_sample(self.images, self.labels)
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.labels[indexes]

    def get_all_data(self, label=True):
        if label:
            return self.images, self.labels
        else:
            return self.images


def load_testing_data(file_name):

    return tiff.imread(file_name).transpose([1, 2, 0])


# 这个函数将小的三通道标签图片拼接成大的图片
def concat_jpg_to_largefile(image_dir, to_dir, to_name):
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
    toarray = np.zeros(shape=(height, width, channel))

    for i in range(rows):
        for j in range(cols):
            fname = '{}_{}_{}_.jpg'.format(i, j, little_image_width)
            fromImage = Image.open(os.path.join(image_dir, fname))
            fromImage = np.array(fromImage)
            toarray[i*little_image_width:i*little_image_width+fromImage.shape[0], j*little_image_height:j*little_image_height+fromImage.shape[1]:] = fromImage
    toarray = toarray[:3840, :14400, :]
    misc.imsave(os.path.join(to_dir, to_name), toarray)
    return toarray


## 训练图像预处理：去噪、对比度增强、归一化
def process(image):
    img1 = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    enh_con = ImageEnhance.Contrast(Image.fromarray(img1))
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    img2 = np.array(image_contrasted)
    return scale_percentile(img2)


def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix


def split_image(img, to_dir, image_size):
    """
    将大的高分辨率卫星图像分割成image_size*image_size的小图片,同样的区域命名相同，分别放在不同文件夹下
    :return:
    """

    for i in range(len(img) / image_size):
        for j in range(len(img[0]) / image_size):
            im_name = str(i) + '_' + str(j) + '_' + str(image_size) + '_.jpg'
            if len(img.shape) == 3: ## 分割图像
                cv2.imwrite(to_dir + im_name, scale_percentile(
                img[i * image_size:i * image_size + image_size,
                j * image_size:j * image_size + image_size, :3]) * 255)
            else: ## 分割标签
                cv2.imwrite(to_dir + im_name,
                img[i * image_size:i * image_size + image_size,
                j * image_size:j * image_size + image_size])


## 数据增广：采用重叠滑动窗口分割大图片，重叠区域大小为30*40
def split_image_overlap_window(img, to_dir):
    """
    举例：
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
    ## 将(960,960,3)的小图片拼接成(3840,14400,3)的大图片
    to_dir = './data_{}/quarterfinals/'.format(image_size)
    label_2015 = concat_jpg_to_largefile('./label/quarterfinals/2015/', to_dir, '2015.jpg')
    print label_2015.shape, label_2015.max()##(3840, 14400, 3),

    label_2017 = concat_jpg_to_largefile('./label/quarterfinals/2017/', to_dir, '2017.jpg')
    print label_2017.shape, label_2017.max()
    assert label_2015.shape == label_2017.shape

    file_name = './original_data/quarterfinals_2015.tif'
    im_2015 = load_testing_data(file_name)
    file_name = './original_data/quarterfinals_2017.tif'
    im_2017 = load_testing_data(file_name)
    print im_2015.shape
    print im_2017.shape

    split_image(im_2015[:3840, :14400, :], to_dir+'images/2015/', image_size)
    split_image(im_2017[:3840, :14400, :], to_dir+'images/2017/', image_size)
    split_image(label_2015, to_dir+'labels/2015/', image_size)
    split_image(label_2017, to_dir+'labels/2017/', image_size)

    # 读取图片，存入txt,多次划分训练集和测试集训练模型
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
    reg = r'([0-9]{4})\/[0-9]{0,2}_[0-9]{0,2}_[0-9]{3}_.jpg'
    with open(to_dir+'train.txt', 'w') as f:
        for line in train:
            if re.match(reg, line):
                f.write(line+'\n')

    with open(to_dir+'valid.txt', 'w') as f:
        for line in valid:
            if re.match(reg, line):
                f.write(line+'\n')
