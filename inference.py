# -*- coding: utf-8 -*-

from train import *
from data_process import *

testing_dir = './data_224/test/'
testing_file_2015 = 'test_2015.txt'
testing_file_2017 = 'test_2017.txt'
model_list = ['model.h5', 'model_2.h5', 'model_3.h5', 'model_4.h5', 'model_5.h5']
# model_list = ['model_5.h5']

def main(argv=None):
    images_list_2015 = []
    with open('./data_224/test/test_2015.txt') as f:
        lines = f.readlines()
        for line in lines:
            images_list_2015.append((line.split('/')[1].strip()).split('.')[0])
    images_list_2017 = []
    with open('./data_224/test/test_2017.txt') as f:
        lines = f.readlines()
        for line in lines:
            images_list_2017.append((line.split('/')[1].strip()).split('.')[0])
    assert len(images_list_2015) == len(images_list_2017)

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

    ## 加载模型
    model = make_fcn_resnet(input_shape=(image_size, image_size, image_channel),
                            nb_labels=label_channel,
                            use_pretraining=True,
                            freeze_base=False
                            )

    test_images_2015 = np.array(testSet_2015.get_all_data(label=False))
    test_images_2017 = np.array(testSet_2017.get_all_data(label=False))
    assert test_images_2015.shape[0] == test_images_2017.shape[0]
    print('Test_images:', test_images_2015.shape, test_images_2015.max())

    pred_2015_summary = np.empty(shape=(len(model_list), test_images_2015.shape[0], 224, 224, 2))
    pred_2017_summary = np.empty(shape=(len(model_list), test_images_2015.shape[0], 224, 224, 2))

    for i,model_name in enumerate(model_list):
        if os.path.exists(save_path + model_name):
            model.load_weights(save_path + model_name)
            print 'model restored from ', save_path, ' model name:', model_name

        ## 预测阶段
        pred_2015 = model.predict(test_images_2015)
        pred_2017 = model.predict(test_images_2017)
        pred_2015_summary[i] = pred_2015
        pred_2017_summary[i] = pred_2017

    print pred_2015_summary.shape
    print('summary the result...')

    pred_2015 = pred_2015_summary.mean(axis=0)
    pred_2017 = pred_2017_summary.mean(axis=0)

    print('prediction 2015:', pred_2015.shape, pred_2015.max())
    print('prediction 2017:', pred_2017.shape, pred_2017.max())
    ## 将预测结果保存
    if not os.path.exists(result_dir + '2015/'):
        os.makedirs(result_dir + '2015/')
    if not os.path.exists(result_dir + '2017/'):
        os.makedirs(result_dir + '2017/')
    for i in range(test_images_2015.shape[0]):
        misc.imsave(os.path.join(result_dir + '2015/', images_list_2015[i] + ".png"), pred_2015[i][:, :, 1])
        misc.imsave(os.path.join(result_dir + '2017/', images_list_2017[i] + ".png"), pred_2017[i][:, :, 1])
    print('prediction has saved!')

    ## 将预测结果根据区域名字拼接成大数组
    submit_array_2015 = submit_formation((pred_2015[:, :, :, 1] > 0.5).astype(np.uint8), images_list_2015, image_size=image_size)
    submit_array_2017 = submit_formation((pred_2017[:, :, :, 1] > 0.5).astype(np.uint8), images_list_2017, image_size=image_size)

    assert ((submit_array_2015 > submit_array_2015.min()) & (submit_array_2015 < submit_array_2015.max())).sum() == 0
    assert ((submit_array_2017 > submit_array_2017.min()) & (submit_array_2017 < submit_array_2017.max())).sum() == 0

    diff = ((submit_array_2017 == 1) & (submit_array_2015 == 0)).astype(np.uint8)
    print diff.mean(), diff.max()
    tiff.imsave('submit.tiff', diff)
    print('Predicting process have done!')


def submit_formation(pred, name_list, image_size):
    rows = 0
    cols = 0
    for img in name_list:
        rows = max(rows, int(img.split("_")[0]) + 1)
        cols = max(cols, int(img.split("_")[1]) + 1)

    width = 15106  # 大图片的宽度
    height = 5106  # 大图片的高度
    toarray = np.zeros(shape=(height, width), dtype=pred.dtype)

    for i in range(pred.shape[0]):
        name = name_list[i]
        x = int(name.split('_')[0])
        y = int(name.split('_')[1])
        toarray[x*image_size:(x+1)*image_size, y*image_size:(y+1)*image_size] = pred[i]

    return toarray


if __name__ == "__main__":

    ## 分割数据
    # file_name = '../land/data/preliminary/quickbird2015.tif'
    # im_2015 = load_testing_data(file_name)
    # file_name = '../land/data/preliminary/quickbird2017.tif'
    # im_2017 = load_testing_data(file_name)
    #
    # split_tiff_file(im_2015, './data_224/test/2015/')
    # split_tiff_file(im_2017, './data_224/test/2017/')

    ## 创建测试数据
    # images_list_2015 = np.array(os.listdir('./data_224/test/2015/'))
    #
    # images_list_2017 = np.array(os.listdir('./data_224/test/2017/'))
    #
    # with open('./data_224/test/test_2015.txt', 'w') as f:
    #     for line in images_list_2015:
    #         f.write('2015/'+line+'\n')
    # with open('./data_224/test/test_2017.txt', 'w') as f:
    #     for line in images_list_2015:
    #         f.write('2017/'+line+'\n')

    tf.app.run()
    #
    # result_2015 = prediction_to_submit(result_dir + '2015/', result_dir, 'result_2015.png')
    # result_2017 = prediction_to_submit(result_dir + '2017/', result_dir, 'result_2017.png')
    # print result_2015.shape
    # print result_2017.shape
    #
    # to_submit_format(result_2015, result_2017)