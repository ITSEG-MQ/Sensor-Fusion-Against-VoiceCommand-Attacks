import numpy as np
from PIL import Image
import os

path = 'G:/JPEGImages/'
newpath = 'G:/JPEGImages_24/'


def turnto24(path):
    files = os.listdir(path)
    files = np.sort(files)
    i = 0
    for f in files:
        imgpath = path + f
        img = Image.open(imgpath).convert('RGB')
        dirpath = newpath
        file_name, file_extend = os.path.splitext(f)
        dst = os.path.join(os.path.abspath(dirpath), file_name + '.jpg')
        img.save(dst)


turnto24(path)
