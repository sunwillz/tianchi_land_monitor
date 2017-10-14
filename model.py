# -*- coding: utf-8 -*-

"""
ResNet based FCN.
"""
from keras.models import Model
from keras.layers import (Input, Activation, Reshape, Conv2D, Lambda, Add)
import tensorflow as tf
import keras.backend as K
# import pydensecrf.densecrf as dcrf
from resnet50 import ResNet50
# from crf_layers import CrfLayer
import numpy as np

FCN_RESNET = 'fcn_resnet'
n_classes = 2


# def dense_crf(probs, img=None, n_iters=10,
#               sxy_gaussian=(1, 1), compat_gaussian=4,
#               kernel_gaussian=dcrf.DIAG_KERNEL,
#               normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
#               sxy_bilateral=(49, 49), compat_bilateral=5,
#               srgb_bilateral=(13, 13, 13),
#               kernel_bilateral=dcrf.DIAG_KERNEL,
#               normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
#     """DenseCRF over unnormalised predictions.
#        More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.
#
#     Args:
#       probs: class probabilities per pixel.
#       img: if given, the pairwise bilateral potential on raw RGB values will be computed.
#       n_iters: number of iterations of MAP inference.
#       sxy_gaussian: standard deviations for the location component of the colour-independent term.
#       compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
#       kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
#       normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
#       sxy_bilateral: standard deviations for the location component of the colour-dependent term.
#       compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
#       srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
#       kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
#       normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
#
#     Returns:
#       Refined predictions after MAP inference.
#     """
#     _, h, w, _ = probs.shape
#     probs = K.eval(probs)
#     probs = probs[0].transpose(2, 0, 1).copy(order='C')  # Need a contiguous array.
#
#     d = dcrf.DenseCRF2D(w, h, n_classes)  # Define DenseCRF model.
#     U = -np.log(probs)  # Unary potential.
#     U = U.reshape((n_classes, -1))  # Needs to be flat.
#     d.setUnaryEnergy(U)
#     d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
#                           kernel=kernel_gaussian, normalization=normalisation_gaussian)
#     if img is not None:
#         assert (img.shape[1:3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
#         d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
#                                kernel=kernel_bilateral, normalization=normalisation_bilateral,
#                                srgb=srgb_bilateral, rgbim=img[0])
#     Q = d.inference(n_iters)
#     preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
#     return np.expand_dims(preds, 0)


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
