from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Deconvolution2D, K, Permute, Lambda, Concatenate, Conv2D, multiply
from keras.layers import merge, Input
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras import layers

from Datasets.data_config import *
from Models.attention_layer import cbam_block
from Models.bilinear_pooling import compact_bilinear


def get_model(args):
    # Dataset config
    config = data_constants[args.dataset.lower()]
    inputCNNshape = config['inputCNNshape']
    inputMLPshape = config['inputMLPshape']
    nb_classes = config['nb_classes']

    # Build the CNN
    inputCNN = Input(shape=inputCNNshape)
    inputNorm = Flatten()(inputCNN)
    inputNorm = BatchNormalization(input_shape=inputCNNshape)(inputNorm)
    inputNorm = layers.Reshape(inputCNNshape)(inputNorm)

    # conv = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputNorm)
    # conv = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(conv)
    # conv = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(conv)
    # conv = BatchNormalization()(conv)
    # pool = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    # Build the CNN
    inputCNN = Input(shape=inputCNNshape)
    inputNorm = Flatten()(inputCNN)
    inputNorm = BatchNormalization(input_shape=inputCNNshape)(inputNorm)
    inputNorm = layers.Reshape(inputCNNshape)(inputNorm)

    # conv1
    conv = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputNorm)
    conv = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    pool = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    pool = Dropout(0.25)(pool)

    # conv2
    conv = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(pool)
    conv = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    pool = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    pool = Dropout(0.25)(pool)

    # conv3
    conv = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(pool)
    conv = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    pool = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    pool = Dropout(0.25)(pool)

    # conv4
    conv = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(pool)
    conv = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(conv)
    conv = cbam_block(conv)
    conv = BatchNormalization()(conv)
    pool = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    pool = Dropout(0.25)(pool)
    reshape = Flatten()(pool)
    fcCNN = Dense(256, activation='relu')(reshape)

    # Build the MLP to CNN for cross-connections
    inputMLP = Input(shape=inputMLPshape)
    fcMLP = Dense(128, activation='relu')(inputMLP)
    fcMLP = BatchNormalization()(fcMLP)
    fcMLP = Dropout(0.5)(fcMLP)

    x12 = Dense(24 * 24)(fcMLP)
    x12 = PReLU()(x12)
    x12 = BatchNormalization()(x12)
    x12 = Dropout(0.5)(x12)
    x12 = Reshape((24, 24, 1))(x12)
    x12 = Deconvolution2D(16, 9, 9, output_shape=(None, 40, 30, 16), border_mode='valid')(x12)
    fcMLPTOCNN = PReLU()(x12)

    try:
    # Multimodal bilinear pooling
        merged = merge([fcCNN, fcMLPTOCNN], compact_bilinear)
    except:
        merged = merge.concatenate([fcCNN, fcMLP])
    merged = BatchNormalization()(merged)
    merged= Dropout(0.5)(merged)

    fc = Dense(512, activation='relu')(merged)
    fc = Dropout(0.5)(fc)
    out = Dense(nb_classes, activation='softmax')(fc)

    # Return the model object
    model = Model(input=[inputCNN, inputMLP], output=out)
    return model




