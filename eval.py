import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
try:
    from keras.layers import K
except:
    import keras.backend as K

from Datasets import traffic_sign_data
from Models.precision_recall import Metrics
from Models import ts_cnn_x_mlp, ts_cnn_mlp_baseline, ts_cnn1att_mlp, ts_cnn_x_mlp_cbp, ts_cnn_mlp_bnn
from triple_loss import triplet_loss

def get_logger(path, mode='a'):
    """
    Create a basic logger
    :param path: log file path
    :param mode: 'a' for append, 'w' for write
    :return: a logger with log level INFO
    """
    name, _ = os.path.splitext(os.path.basename(path))
    logging.basicConfig(format="%(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    handler = logging.FileHandler(path, mode=mode)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger.info

logger = get_logger('./train.log', mode='w')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

parser = argparse.ArgumentParser()
#  data and Model
parser.add_argument('--dataset', type=str, choices=['traffic_sign'])
parser.add_argument('--model', type=str, choices=["ts_cnn_mlp_baseline", "ts_cnn_x_mlp", "ts_cnn1att_mlp","ts_cnn_x_mlp_cbp","ts_cnn_mlp_bnn"])
# Optimization hyperparameters
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=300)
args = parser.parse_args()


def train_evaluate(model, X_image, X_audio, Y, indices, model_save_path):

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])


    val_accs = []
    n_split = 10
    kf = KFold(n_splits=n_split, shuffle=True, random_state=None)
    for cnt, (train, test) in enumerate(kf.split(indices)):
        checkpoint = ModelCheckpoint(os.path.join(model_save_path, 'model_best_{:0>2d}.h5'.format(cnt)),
                                     monitor='val_acc',
                                     verbose=0,
                                     save_weights_only=False,
                                     save_best_only=True,
                                     mode='max',
                                     period=0)
        logger('Training ----------- k fold: {:0>2d} ----------- '.format(cnt))
        history = model.fit([X_image[train], X_audio[train]], Y[train],
                            batch_size=args.batch_size,
                            epochs=args.num_epochs,
                            validation_data=([X_image[test], X_audio[test]], Y[test]),
                            callbacks=[checkpoint, Metrics(valid_data=([X_image[test], X_audio[test]], Y[test]))],
                            verbose=1)
        val_accs.append(np.array(history.history['val_acc']).max())
        labs = []
        scores = []
        for k, v in history.history.items():
            labs.append(k)
            scores.append(v)
            value = (k, np.array(v).max()) # 建议换成max
            logger(value)
        [plt.plot(scores[i], label=labs[i]) for i in range(len(scores)) if 'acc' in labs[i]]
        plt.title('train accuracy vs. val accuracy')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(model_save_path, '{:0>2d}_accuracy.png'.format(cnt)))
        plt.close()

        [plt.plot(scores[i], label=labs[i]) for i in range(len(scores)) if 'loss' in labs[i]]
        plt.title('train loss vs. val loss')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(model_save_path, '{:0>2d}_loss.png'.format(cnt)))
        plt.close()

        [plt.plot(scores[i], label=labs[i]) for i in range(len(scores)) if 'acc' not in labs[i] and 'loss' not in labs[i]]
        plt.title('Validation Precision & Recall & F1 score')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(model_save_path, '{:0>2d}_eval.png'.format(cnt)))
        plt.close()

    val_accs_mean = np.array(val_accs).mean()
    logger('-------------------------------------------------------------')
    logger('validation accuracies on {} fold'.format(n_split))
    logger(val_accs)
    logger('mean validation accuracy: {:.6f}'.format(val_accs_mean))
    logger('---------------------------DONE------------------------------')

    return val_accs_mean


if __name__ == '__main__':
    # Prepare model and dataset
    baseline = 'baseline' in args.model.lower()
    cross_connection = 'cnn_x_mlp' in args.model.lower()
    attention = "cnn1att" in args.model.lower()
    compact_bilinear_pooling = "cnn_mlp_cbp" in args.model.lower()
    binary_neural_network = "cnn_mlp_bnn" in args.model.lower()


    if args.dataset.lower() == 'traffic_sign':
        (X_image, X_audio, P, Y) = traffic_sign_data.get_traffic_sign()
        if baseline:
            model = ts_cnn_mlp_baseline.get_model(args)
        elif cross_connection:
            model = ts_cnn_x_mlp.get_model(args)
        elif attention:
            model = ts_cnn1att_mlp.get_model(args)
        elif compact_bilinear_pooling:
            model = ts_cnn_x_mlp_cbp.get_model(args)
        elif binary_neural_network:
            model = ts_cnn_mlp_bnn.get_model(args)

    model_save_path = 'results'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    indices = [i for i in range(len(X_image))]
    max_acc = train_evaluate(model, X_image, X_audio, Y, indices, model_save_path)
    # print(model.summary())
