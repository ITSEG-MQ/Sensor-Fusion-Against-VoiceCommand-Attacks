import os

BASE_DIR = "D:/"
path= BASE_DIR +'Data_preprocessing/test name/'
#path =BASE_DIR +'stop data/image/'


def rename():
    i = 1056
    filelist = os.listdir(path)
    for files in filelist:
        i = i + 1
        Olddir = os.path.join(path, files)
        if os.path.isdir(Olddir):       #如果为文件夹就继续往下搜索
            continue
        filename = os.path.splitext(files)[0]#读取文件名
        filetype = os.path.splitext(files)[1]#读取文件格式
        Newdir = os.path.join(path, str(i) + filetype)
        os.rename(Olddir, Newdir)#重命名

if __name__ == '__main__':
    rename()