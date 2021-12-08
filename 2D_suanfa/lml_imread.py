# encoding:utf-8

#   带中文路径的图片载入

##################################################################
#   调用请使用这种方式
#   img = lml_imread.imread('../DATA/img/轮廓查找/yt.jpg')
##################################################################

import cv2
import cv2 as cv
import numpy as np


def imread(img_path):
    # return cv.imdecode(np.fromfile(path_img, dtype=np.float64), -1)
    return cv.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)


def imread2(file_path=""):
    file_path_gbk = file_path.encode('gbk')  # unicode转gbk，字符串变为字节数组
    img_mat = cv2.imread(file_path_gbk.decode())  # 字节数组直接转字符串，不解码
    return img_mat
