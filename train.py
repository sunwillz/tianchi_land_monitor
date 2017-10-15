# -*- coding: utf-8 -*-
"""
train.py:训练模型
"""
import argparse
import keras
from keras.callbacks import ReduceLROnPlateau
from data_process import *
from model import *

learning_rate = 1e-5  # 学习率
decay = 0
batch_size = 8
valid_batch_size = 8  # 验证集样本数
epochs = 10  # 训练轮数

model_name = 'deeplabv2_model_{}.h5'.format(image_size)
training_dir = './data_{}/'.format(image_size)
train_file = 'train.txt'
validation_dir = './data_{}/'.format(image_size)
valid_file = 'valid.txt'
save_path = './logs/'  # 训练日志和模型存放目录
result_dir = './result/'  # 预测结果存放目录


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="resnet based fcn Network")
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--image-size", type=int, default=image_size)
    parser.add_argument("--learning_rate", type=float, default=learning_rate)
    return parser.parse_args()


def f1(y_true, y_pred):
    y_true = y_true[:, :, 1]
    y_pred = y_pred[:, :, 1]

    # 将标签值展平
    y_true = K.reshape(y_true, shape=[1, -1])
    y_pred = K.reshape(y_pred, shape=[1, -1])

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


def main(args):
    args = get_arguments()

    training_data = Dataset_reader(dataset_dir=training_dir,
                                   file_name=train_file,
                                   image_size=image_size,
                                   image_channel=image_channel,
                                   label_channel=label_channel
                                   )
    validation_data = Dataset_reader(dataset_dir=validation_dir,
                                     file_name=valid_file,
                                     image_size=image_size,
                                     image_channel=image_channel,
                                     label_channel=label_channel
                                     )

    train_images, train_annotations = training_data.get_all_data()
    valid_images, valid_annotations = validation_data.get_random_batch(valid_batch_size)
    test_images, test_annotations = validation_data.get_random_batch(valid_batch_size)
    print 'training max:', train_images.max()
    print 'training label max: ', train_annotations.max()

    print 'validation max:', valid_images.max()
    print 'validation lable max:', valid_annotations.max()

    # model = make_fcn_resnet(input_shape=(image_size, image_size, image_channel),
    #                         nb_labels=label_channel,
    #                         use_pretraining=True,
    #                         freeze_base=False
    #                         )
    model = DeeplabV2(input_shape=(image_size, image_size, image_channel),
                      classes=label_channel,
                      weights=None,
                      )
    if os.path.exists(save_path + model_name):
        model.load_weights(save_path + model_name)
        print 'model restored from ', save_path, model_name
    adam = keras.optimizers.Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=adam,
                  metrics=[f1, 'accuracy'])
    # 学习率衰减
    reduce_lr = ReduceLROnPlateau(monitor='val_f1',
                                  factor=0.1,
                                  patience=5,
                                  min_lr=1e-7)

    history = model.fit(train_images, train_annotations,
                        batch_size=batch_size,
                        epochs=args.epochs,
                        verbose=1,
                        validation_data=(valid_images, valid_annotations),
                        callbacks=[reduce_lr]
                        )
    # plt.plot()
    # plt.plot(history.history['f1'])
    # plt.plot(history.history['val_f1'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model evaluate score')
    # plt.ylabel('score')
    # plt.xlabel('epoch')
    # plt.legend(['train_f1', 'valid_f1', 'valid_acc'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    model.save_weights(save_path + model_name)
    print 'model saved at ', save_path + model_name

    loss, f1_score, accuracy = model.evaluate(test_images, test_annotations, verbose=0)

    print('Test loss:', loss)
    print('Test f1_score:', f1_score)
    print('Test accuracy:', accuracy)

    print('Test image shape:', test_images.shape, test_images.max())
    print('Test label shape:', test_annotations.shape, test_annotations.max())

    # 测试集预测结果可视化
    pred = model.predict(test_images)

    # pred.shape(10, 224, 224, 2)
    # Test_images.shape(10, 224, 224, 3)
    # Test_annotations.shape(10, 224, 224, 2)
    print('pred:', pred.shape, pred.max())
    print('Test_images:', test_images.shape, test_images.max())
    print('Test_annotations:', test_annotations.shape, test_annotations.max())

    for i in range(valid_batch_size):
        misc.imsave(os.path.join(result_dir, "pred" + str(i + 1) + ".png"), pred[i][:, :, 1])
        misc.imsave(os.path.join(result_dir, "gt" + str(i + 1) + ".png"), test_annotations[i][:, :, 1])
        misc.imsave(os.path.join(result_dir, "inp" + str(i + 1) + ".jpg"), test_images[i])


if __name__ == "__main__":
    tf.app.run()
