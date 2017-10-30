# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import (Input, Activation, Reshape, Conv2D, Lambda, Add)
import tensorflow as tf
import keras.backend as K
from resnet50 import ResNet50
from crf_layers import CrfLayer
import numpy as np

FCN_RESNET = 'fcn_resnet'
n_classes = 2


def make_fcn_resnet(input_shape, nb_labels, use_pretraining, freeze_base):
    nb_rows, nb_cols, _ = input_shape
    input_tensor = Input(shape=input_shape)
    weights = 'imagenet' if use_pretraining else None

    model = ResNet50(include_top=False, weights=weights, input_tensor=input_tensor)

    if freeze_base:
        for layer in model.layers:
            layer.trainable = False

    x32 = model.get_layer('act3d').output
    x16 = model.get_layer('act4f').output
    x8 = model.get_layer('act5c').output

    c32 = Conv2D(nb_labels, (1, 1), name='conv_labels_32')(x32)
    c16 = Conv2D(nb_labels, (1, 1), name='conv_labels_16')(x16)
    c8 = Conv2D(nb_labels, (1, 1), name='conv_labels_8')(x8)

    ## 采用双线性插值法调整图像大小，http://www.cnblogs.com/zzw-in/p/Bilinear_interpolation.html
    def resize_bilinear(images):
        return tf.image.resize_bilinear(images, [nb_rows, nb_cols])

    r32 = Lambda(resize_bilinear, name='resize_labels_32')(c32)
    r16 = Lambda(resize_bilinear, name='resize_labels_16')(c16)
    r8 = Lambda(resize_bilinear, name='resize_labels_8')(c8)

    m = Add(name='merge_labels')([r32, r16, r8])

    x = Reshape((nb_rows * nb_cols, nb_labels))(m)
    x = Activation('softmax')(x)
    x = Reshape((nb_rows, nb_cols, nb_labels))(x)

    # output = CrfLayer(image_dims=(nb_rows, nb_cols),
    #                   num_classes=n_classes,
    #                   theta_alpha=160.,
    #                   theta_beta=3.,
    #                   theta_gamma=3.,
    #                   num_iterations=10,
    #                   name='crf')([x, input_tensor])

    model = Model(inputs=input_tensor, outputs=x)

    return model
