from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, merge, Input, K, Permute, Lambda, Concatenate, Conv2D, multiply
from keras.layers.normalization import BatchNormalization
from keras import layers

from BNN.binary_ops import binary_tanh as binary_tanh_op
from BNN.binary_layers import BinaryDense, BinaryConv2D, Clip
from Datasets.data_config import *
from Models.attention_layer import cbam_block

def binary_tanh(x):
    return binary_tanh_op(x)

def spatial_attention(input_feature):
    kernel_size = 7
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

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

    #conv1
    #conv = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputNorm)
    #conv = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(conv)

    conv = BinaryConv2D(16, kernel_size=(3, 3), border_mode='same', activation=binary_tanh, use_bias=False)(inputNorm)
    conv = BinaryConv2D(16, kernel_size=(3, 3), border_mode='same', activation=binary_tanh, use_bias=False)(conv)
    conv = BatchNormalization()(conv)
    pool = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    pool = Dropout(0.25)(pool)

    #conv2
    #conv = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(pool)
    #conv = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(conv)
    conv = BinaryConv2D(16, kernel_size=(3, 3), border_mode='same', activation=binary_tanh, use_bias=False)(pool)
    conv = BinaryConv2D(16, kernel_size=(3, 3), border_mode='same', activation=binary_tanh, use_bias=False)(conv)
    conv = BatchNormalization()(conv)
    pool = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    pool = Dropout(0.25)(pool)

    # cbam = cbam_block(pool)
    # reshape = Flatten()(cbam)
    # fcCNN = Dense(256, activation='relu')(reshape)

    reshape = Flatten()(pool)
    fcCNN = BinaryDense(512, activation=binary_tanh, use_bias=False)(reshape)
    # fcCNN = Dense(256, activation=binary_tanh)(reshape)

    # Build the MLP
    inputMLP = Input(shape=inputMLPshape)
    fcMLP = BinaryDense(512, activation=binary_tanh, use_bias=False)(inputMLP)
   # fcMLP = Dense(128, activation='relu')(inputMLP)
    fcMLP = BatchNormalization()(fcMLP)
    fcMLP = Dropout(0.5)(fcMLP)
    fcMLP = Dense(128, activation='relu')(fcMLP)

    # Merge the models
    try:
        merged = merge([fcCNN, fcMLP], mode='concat')
    except:
        merged = merge.concatenate([fcCNN, fcMLP])

    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)

    # fc = Dense(512, activation='relu')(merged)
    fc = BinaryDense(512, activation=binary_tanh, use_bias=False)(merged)

    fc = Dropout(0.5)(fc)
    #out = Dense(nb_classes, activation='softmax')(fc)
    out = BinaryDense(nb_classes, activation='softmax', use_bias=False)(fc)

    # Return the model object
    model = Model(input=[inputCNN, inputMLP], output=out)

    return model
