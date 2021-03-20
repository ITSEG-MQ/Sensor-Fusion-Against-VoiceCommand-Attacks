
# 尝试将转化的npy文件展示其中的内容。
import numpy as np
from PIL import Image
import os
import tensorflow as tf
import cv2

img_path = 'D:/Data/image'
out_path = 'D:/Data/image_npy'


def get_file(path, rule='.jpg'):
    all = []
    for fpathe, dirs, files in os.walk(path):  # all 目录
        for f in files:
            filename = os.path.join(fpathe, f)
            if filename.endswith(rule):
                all.append((filename))
    return all


if __name__ == '__main__':
    paths = get_file(img_path, rule='.jpg')
    for ims in paths:
        print(ims)
        # cut path and '.jpg' ,保留 000000050755_bicLRx4图片名称
        file_name = ims.strip(".jpg")
        # 可以打印出来看看cut以后的效果
        # print((file_name,type(file_name)))

        im1 = cv2.imread(ims)
        im2 = np.array(im1)
        # print(type(im2))

        np.save(file_name + '.npy', im2)
        print(tf.shape(im2))

