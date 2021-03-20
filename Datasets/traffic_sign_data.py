import os
import cv2
import numpy as np
from keras.utils import np_utils

from sklearn import preprocessing
from Datasets.data_config import *

def proc_audio(mfcc):
    audio_len, = traffic_sign_config['inputMLPshape']
    return mfcc[:audio_len]

def get_traffic_sign():
    input_h, input_w, input_c = traffic_sign_config['inputCNNshape']
    data_dir = traffic_sign_config['data_dir']
    # imgs_dir = os.path.join(data_dir, 'image')
    # audios_dir = os.path.join(data_dir, 'audio')

    imgs_dir = os.path.join(data_dir, 'audi_image')
    audios_dir = os.path.join(data_dir, 'audi_audio')

    imgs = list(sorted(os.listdir(imgs_dir)))
    #imgs = [p for p in imgs if p.endswith('jpg')]
    imgs = [p for p in imgs if p.endswith('txt')]

    audios = list(sorted(os.listdir(audios_dir)))
    audios = [p for p in audios if p.endswith('mfcc')]
    assert len(imgs) == len(audios)

    # Containers for the data
    X_image = []
    X_mfcc = []
    Y = []
    P = []

    cls2id = {s: i for i, s in enumerate(traffic_sign_config['classes'])}
    for i, a in zip(imgs,audios):
        class_name = i.split('_')[0]
        class_id = cls2id[class_name]

        # image = cv2.imread(os.path.join(imgs_dir, i), 1)
        # image = cv2.resize(image, (input_w, input_h))

        image = np.loadtxt(os.path.join(imgs_dir, i), delimiter=',')
        image = np.resize(image,(input_w, input_h, input_c))
       # min_max_scaler = preprocessing.MinMaxScaler()
       # image = min_max_scaler.fit_transform(image)

        mfccs = np.fromfile(os.path.join(audios_dir, a))
        mfccs = proc_audio(mfccs)
        X_image.append(image)
        X_mfcc.append(mfccs)
        Y.append(class_id)

    X_image = np.array(X_image, dtype=np.float32)
    X_mfcc = np.array(X_mfcc, dtype=np.float32)
    Y = np.array(Y, dtype=np.int)
    Y = np_utils.to_categorical(Y, 5)
    X_mfcc[np.isinf(X_mfcc)] = 0
    X_mfcc[np.isnan(X_mfcc)] = 0

    return (X_image, X_mfcc, P, Y)


def get_single_input(image_path, audio_path):
    input_h, input_w, _ = traffic_sign_config['inputCNNshape']
    image = cv2.imread(image_path, 1)
    image = cv2.resize(image, (input_w, input_h))

    mfccs = np.fromfile(audio_path)
    mfccs = proc_audio(mfccs)
    mfccs[np.isinf(mfccs)] = 0
    mfccs[np.isnan(mfccs)] = 0

    # predict
    image = image[np.newaxis, :, :, :]
    mfccs = mfccs[np.newaxis, :]

    return image, mfccs

def get_all_inputs(test_data_dir):
    input_h, input_w, _ = traffic_sign_config['inputCNNshape']
    data_dir = test_data_dir
    # data_dir = traffic_sign_config['data_dir']
    imgs_dir = os.path.join(data_dir, 'image')
    audios_dir = os.path.join(data_dir, 'audio')
    imgs = list(sorted(os.listdir(imgs_dir)))
    imgs = [p for p in imgs if p.endswith('jpg')]
    audios = list(sorted(os.listdir(audios_dir)))
    audios = [p for p in audios if p.endswith('mfcc')]
    assert len(imgs) == len(audios)
    # Containers for the data
    X_image = []
    X_mfcc = []
    Y = []

    for i, a in zip(imgs,audios):
        class_name = i.split('_')[0]
        image = cv2.imread(os.path.join(imgs_dir, i), 1)
        image = cv2.resize(image, (input_w, input_h))
        mfccs = np.fromfile(os.path.join(audios_dir, a))
        mfccs = proc_audio(mfccs)
        X_image.append(image)
        X_mfcc.append(mfccs)
        Y.append(class_name)
    X_image = np.array(X_image, dtype=np.float32)
    X_mfcc = np.array(X_mfcc, dtype=np.float32)
    X_mfcc[np.isinf(X_mfcc)] = 0
    X_mfcc[np.isnan(X_mfcc)] = 0

    return (X_image, X_mfcc, imgs, audios, Y)
