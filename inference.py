# -*- coding: utf-8 -*-
"""
predict the test data
"""
from train import *
from data_process import *

height = 3000
width = 15106
testing_file_2015 = 'test_2015.txt'
testing_file_2017 = 'test_2017.txt'
model_list = ['deeplabv2_model_160.h5', 'deeplabv2_model_256.h5', 'resnet_model_224.h5', 'resnet_model_256.h5']


def main(argv=None):

    pred_2015_summary = np.empty(shape=(len(model_list), height, width))
    pred_2017_summary = np.empty(shape=(len(model_list), height, width))

    for i, model_name in enumerate(model_list):
        image_size = int(model_name.split('.')[0][-3:])
        print('images size:', image_size)
        testing_dir = './data_{}/test/'.format(image_size)
        img_list_2015 = []
        with open('./data_{}/test/test_2015.txt'.format(image_size)) as f:
            lines = f.readlines()
            for line in lines:
                img_list_2015.append((line.split('/')[1].strip()).split('.')[0])
        img_list_2017 = []
        with open('./data_{}/test/test_2017.txt'.format(image_size)) as f:
            lines = f.readlines()
            for line in lines:
                img_list_2017.append((line.split('/')[1].strip()).split('.')[0])
        assert len(img_list_2015) == len(img_list_2017)

        ## 准备数据
        testSet_2015 = Dataset_reader(dataset_dir=testing_dir,
                                      file_name=testing_file_2015,
                                      image_size=image_size,
                                      image_channel=image_channel,
                                      label_channel=label_channel,
                                      test=True
                                      )
        testSet_2017 = Dataset_reader(dataset_dir=testing_dir,
                                      file_name=testing_file_2017,
                                      image_size=image_size,
                                      image_channel=image_channel,
                                      label_channel=label_channel,
                                      test=True
                                      )

        test_images_2015 = np.array(testSet_2015.get_all_data(label=False))
        test_images_2017 = np.array(testSet_2017.get_all_data(label=False))
        assert test_images_2015.shape[0] == test_images_2017.shape[0]
        print('Test_images:', test_images_2015.shape, test_images_2015.max())

        if os.path.exists(save_path + model_name):
            if 'deeplab' in model_name:
                ## 加载模型
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

            model.load_weights(save_path + model_name)
            print 'model restored from ', save_path, ' model name:', model_name

        ## 预测阶段
        pred_2015 = model.predict(test_images_2015, batch_size=4, verbose=1)
        pred_2017 = model.predict(test_images_2017, batch_size=4, verbose=1)
        pred_2015_summary[i] = submit_formation(pred_2015[:, :, :, 1], img_list_2015,
                                             image_size=image_size)
        pred_2017_summary[i] = submit_formation(pred_2017[:, :, :, 1], img_list_2017,
                                             image_size=image_size)
    assert pred_2015_summary.shape == pred_2017_summary.shape
    print pred_2015_summary.shape
    print('summary the result...')

    pred_2015 = pred_2015_summary.mean(axis=0)
    pred_2017 = pred_2017_summary.mean(axis=0)

    print('prediction 2015:', pred_2015.shape, pred_2015.max())
    print('prediction 2017:', pred_2017.shape, pred_2017.max())
    ## 将预测结果保存
    # if not os.path.exists(result_dir + '2015/'):
    #     os.makedirs(result_dir + '2015/')
    # if not os.path.exists(result_dir + '2017/'):
    #     os.makedirs(result_dir + '2017/')
    # for i in range(test_images_2015.shape[0]):
    #     misc.imsave(os.path.join(result_dir + '2015/', img_list_2015[i] + ".png"), pred_2015[i][:, :, 1])
    #     misc.imsave(os.path.join(result_dir + '2017/', img_list_2017[i] + ".png"), pred_2017[i][:, :, 1])
    # print('prediction has saved!')

    ## 将预测结果根据区域名字拼接成大数组
    # submit_array_2015 = submit_formation((pred_2015 > 0.8).astype(np.uint8), img_list_2015, image_size=image_size)
    # submit_array_2017 = submit_formation((pred_2017 > 0.8).astype(np.uint8), img_list_2017, image_size=image_size)
    pred_2015 = (pred_2015 > 0.5).astype(np.uint8)
    pred_2017 = (pred_2017 > 0.5).astype(np.uint8)
    assert ((pred_2015 > pred_2015.min()) & (pred_2015 < pred_2015.max())).sum() == 0
    assert ((pred_2017 > pred_2017.min()) & (pred_2017 < pred_2017.max())).sum() == 0

    diff = ((pred_2017 == 1) & (pred_2015 == 0)).astype(np.uint8)
    print diff.shape, diff.mean(), diff.max()
    tiff.imsave('submit.tiff', diff)
    print('Predicting process have done!')


def submit_formation(pred, name_list, image_size):
    rows = 0
    cols = 0
    for img in name_list:
        rows = max(rows, int(img.split("_")[0]) + 1)
        cols = max(cols, int(img.split("_")[1]) + 1)

    _width = max(cols*image_size, width)  # 大图片的宽度
    _height = max(rows*image_size, height)  # 大图片的高度
    toarray = np.zeros(shape=(_height, _width), dtype=pred.dtype)

    for i in range(pred.shape[0]):
        name = name_list[i]
        x = int(name.split('_')[0])
        y = int(name.split('_')[1])
        toarray[x*image_size:(x+1)*image_size,
                y*image_size:(y+1)*image_size] = pred[i]

    return toarray[:height, :width]


if __name__ == "__main__":

    ## 分割数据
    file_name = './quickbird2015_preliminary_2.tif'
    im_2015 = tiff.imread(file_name).transpose([1, 2, 0])
    file_name = './quickbird2017_preliminary_2.tif'
    im_2017 = tiff.imread(file_name).transpose([1, 2, 0])

    split_image(im_2015, './data_{}/test/images/2015/'.format(image_size), image_size)
    split_image(im_2017, './data_{}/test/images/2017/'.format(image_size), image_size)

    ## 创建测试数据
    images_list_2015 = np.array(os.listdir('./data_{}/test/images/2015/'.format(image_size)))
    images_list_2017 = np.array(os.listdir('./data_{}/test/images/2017/'.format(image_size)))
    reg = r'[0-9]{0,2}_[0-9]{0,2}_[0-9]{3}_.jpg'
    with open('./data_{}/test/test_2015.txt'.format(image_size), 'w') as f:
        for line in images_list_2015:
            if re.match(reg, line):
                f.write('2015/'+line+'\n')
    with open('./data_{}/test/test_2017.txt'.format(image_size), 'w') as f:
        for line in images_list_2015:
            if re.match(reg, line):
                f.write('2017/'+line+'\n')
    # tf.app.run()


