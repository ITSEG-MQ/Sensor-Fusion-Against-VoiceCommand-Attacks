import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from Datasets.data_config import traffic_sign_config
from Datasets.traffic_sign_data import get_single_input, get_all_inputs


from keras import backend as K
def Precision(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))  # predicted positives
    precision = tp / (pp + K.epsilon())
    return precision

def Recall(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))  # possible positives
    recall = tp / (pp + K.epsilon())
    return recall

def Fmeasure(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

model = load_model('model_cnn1att.h5', custom_objects={'Precision': Precision,'Recall':Recall,'Fmeasure':Fmeasure})

TEST_SINGLE = False
if TEST_SINGLE:
    audio_path = './Inference/go_mfcc/go_251.mfcc'
    image_path = './Inference/left_jpg/left_320.jpg'
    image, mfccs = get_single_input(image_path, audio_path)
    probs = model.predict([image, mfccs])
    print(probs, probs.sum())
    pred_id = np.argmax(probs[0])
    pred_cls = traffic_sign_config['classes'][pred_id]
    print('Predicted class:', pred_cls, "Confidence", probs[0, pred_id])

else:
    # success rate
    test_data_dir = "./Test"
    (X_image, X_mfcc, imgs, audios, Y) = get_all_inputs(test_data_dir)
    pred_classes = []
    probs_file = open('probs.tsv', 'w', encoding='utf-8')
    meta_file = open('meta.tsv', 'w', encoding='utf-8')
    for image, mfccs in zip(X_image, X_mfcc):
        image = image[np.newaxis, :, :, :]
        mfccs = mfccs[np.newaxis, :]
        probs = model.predict([image, mfccs])
        pred_id = np.argmax(probs[0])
        pred_cls = traffic_sign_config['classes'][pred_id]
        pred_classes.append(pred_cls)

        probs2write = "\t".join(["{:.2f}".format(p) for p in probs[0]])
        # probs2write += "\r\n"
        probs2write += "\n"
        probs_file.write(probs2write)
        # meta_file.write(pred_cls + "\r\n")
        meta_file.write(pred_cls + "\n")
    probs_file.close()
    meta_file.close()
    results = [(im_name, audio_name, gt_lab, pred_cls) for im_name, audio_name, gt_lab, pred_cls in zip(imgs, audios, pred_classes, Y)]
    acc = 0
    cnt = 0
    for p, l in zip(pred_classes, Y):
        if p == l:
            cnt += 1
    acc = cnt / len(Y)
    # print(results)
    # use pretty table for better view
    print("image \taudio \tground_truth \tpredicted_class")
    for r in results:
        print(r)
    print("success rate:", acc)