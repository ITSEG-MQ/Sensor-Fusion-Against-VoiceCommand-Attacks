from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, merge, Input, K, Permute, Lambda, Concatenate, Conv2D, multiply
from keras.layers.normalization import BatchNormalization
from keras import layers

from Datasets.data_config import *
from Models.attention_layer import cbam_block


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

    conv = Convolution2D(16, 3, 3, border_mode='same', activation="relu")(inputNorm)
    conv = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(conv)
    conv = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    pool = MaxPooling2D((2,2), strides=(2, 2))(conv)
    pool = Dropout(0.25)(pool)

    cbam= cbam_block(pool)
    reshape = Flatten()(cbam)
    fcCNN = Dense(256, activation='relu')(reshape)

    # Build the MLP
    inputMLP = Input(shape=inputMLPshape)
    fcMLP = Dense(128, activation='relu')(inputMLP)
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
    fc = Dense(512, activation='relu')(merged)
    fc = Dropout(0.5)(fc)
    out = Dense(nb_classes, activation='softmax')(fc)
    # Return the model object
    model = Model(input=[inputCNN, inputMLP], output=out)
    return model
