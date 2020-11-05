from __future__ import division
from __future__ import print_function

import time

import keras.backend as K
import numpy as np
import scipy.io
from keras.layers import Lambda, Dense, TimeDistributed, Input
from keras.models import Model
from keras.preprocessing import image

import utils
from RoiPooling import RoiPooling
from get_regions import rmac_regions, get_size_vgg_feat_map
from vgg16 import VGG16

def addition(x):
    sum = K.sum(x, axis=1)
    return sum


def weighting(input):
    x = input[0]
    w = input[1]
    w = K.repeat_elements(w, 512, axis=-1)
    out = x * w
    return out


def rmac(input_shape, num_rois, rank):
    # Load VGG16
    vgg16_model = VGG16(utils.DATA_DIR + "vgg16_weights_th_dim_ordering_th_kernels_" + str(rank) + ".h5", input_shape)

    # Regions as input
    in_roi = Input(shape=(num_rois, 4), name='input_roi')

    # ROI pooling
    x = RoiPooling([1], num_rois)([vgg16_model.layers[-5].output, in_roi])

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='norm1')(x)

    # PCA
    x = TimeDistributed(Dense(512, name='pca',
                              kernel_initializer='identity',
                              bias_initializer='zeros'))(x)

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='pca_norm')(x)

    # Addition
    rmac = Lambda(addition, output_shape=(512,), name='rmac')(x)

    # # Normalization
    rmac_norm = Lambda(lambda x: K.l2_normalize(x, axis=1), name='rmac_norm')(rmac)

    # Define model
    model = Model([vgg16_model.input, in_roi], rmac_norm)

    # Load PCA weights
    mat = scipy.io.loadmat(utils.DATA_DIR + utils.PCA_FILE)
    b = np.squeeze(mat['bias'], axis=1)
    w = np.transpose(mat['weights'])
    model.layers[-4].set_weights([w, b])
    del vgg16_model
    return model


def execute(file, curr, rank):
    # Load sample image
    # file = utils.DATA_DIR + 'sample.jpg'
    img = image.load_img(file)

    # Resize
    scale = utils.IMG_SIZE / max(img.size)
    new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), str(curr), "is processing...")
    img = img.resize(new_size)

    # Mean substraction
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_image(x)

    # Load RMAC model
    Wmap, Hmap = get_size_vgg_feat_map(x.shape[3], x.shape[2])
    regions = rmac_regions(Wmap, Hmap, 3)
    # print('Loading RMAC model...')
    model = rmac((x.shape[1], x.shape[2], x.shape[3]), len(regions), rank)

    # Compute RMAC vector
    # print('Extracting RMAC from image...')
    RMAC = model.predict([x, np.expand_dims(regions, axis=0)])
    return RMAC
