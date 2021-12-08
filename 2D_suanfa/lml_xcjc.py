#   瑕疵检测 局部灰度二值化

import os

import cv2
import cv2 as cv
import numpy as np

import lml_imread


def zh_ch(string):
    return string.encode("gbk").decode(errors="ignore")


# 灰度区间
def huidu_qujian(img):
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except:
        #     print(ValueError)
        img_gray = img

    cb = 1  # 裁边1像素 抗干扰
    img_gray = img_gray[cb:img_gray.shape[0] - cb, cb:img_gray.shape[1] - cb]

    min_v, max_v, min_pt, max_pt = cv.minMaxLoc(img_gray)  # 获取灰度区间和极值点坐标
    # print(min_v, max_v, min_pt, max_pt)

    # plt.hist(img.ravel(), 256, [0, 256])
    # plt.show()
    # cv.imshow("test", img_gray)
    # cv.waitKey(0)

    return min_v, max_v, min_pt, max_pt, img_gray


# 灰度区间 测试
def huidu_cs():
    inputpath = 'C:\\Users\\LML-YLC-PC\\Desktop\\新建文件夹\\brown\\'
    outputpath = 'C:\\Users\\LML-YLC-PC\\Desktop\\新建文件夹\\b_t1\\'
    filenames = os.listdir(inputpath)

    g_c_max, g_c_min, g_c = -1, 300, 0
    for file_name in filenames:
        # img = lml_imread.imread(inputpath + file_name)  # ,cv2.IMREAD_GRAYSCALE)
        path = inputpath + file_name
        img = lml_imread.imread(path)
        # print()
        min_v, max_v, min_pt, max_pt, img_gray = huidu_qujian(img)
        g_c = max_v - min_v
        if g_c_max < g_c:
            g_c_max = g_c
        if g_c_min > g_c:
            g_c_min = g_c

        if 0:
            if g_c > 60:
                print("==========================================")
                print(min_v, max_v, min_pt, max_pt)
                img = lml_imread.imread(inputpath + file_name)
                cv.circle(img, min_pt, 100, (225, 0, 225), 7, 8)
                cv.circle(img, max_pt, 100, (225, 0, 225), 7, 8)
                cv2.imwrite(outputpath + file_name, img)
                cv2.imencode('.bmp', img)[1].tofile(outputpath + "_lab" + file_name)

        # img1 = img[0:1024, 0:1024]
        # img2 = img[0:1024, 1024:2048]
        # cv2.imwrite(outputpath + "L" + file_name, img1)
        # cv2.imwrite(outputpath + "R" + file_name, img2)
    print(g_c_max, g_c_min)


# huidu_cs()

# 局部方差
def jubu_fangcha(img):
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except:
        #     print(ValueError)
        img_gray = img

    img_gray2 = img_gray / 255.0  # 格式转换  灰度值转换为0~1区间

    # 计算均值图像和均值图像的平方图像
    img_blur = cv2.blur(img_gray2, (10, 10))  # 均值处理 即模糊
    reslut_1 = img_blur ** 2  # 灰度值平方 效果类似对比加深

    # 计算图像的平方和平方后的均值
    img_2 = img_gray2 ** 2
    reslut_2 = cv2.blur(img_2, (10, 10))

    reslut = np.sqrt(np.maximum(reslut_2 - reslut_1, 0))
    #   np.maximum(X, Y) 用于逐元素比较两个array的大小。返回同位元素中大的集合的一个同长度array
    #   np.sqrt(B):求B的开方
    # reslut_0 = np.sqrt(reslut_2 - reslut_1 )

    min_v, max_v, min_pt, max_pt = cv.minMaxLoc(reslut)
    # print(len(reslut))
    # print(min_v, max_v)

    # cv.namedWindow("stdfilt", cv.WINDOW_NORMAL)
    # cv.resizeWindow("stdfilt", 1720, 980)
    # cv2.imshow('huidu', img_gray)   # 灰度图
    # cv2.imshow('mh->pf', reslut_1)  # 先均值模糊+再平方加深
    # cv2.imshow('pf->mh', reslut_2)  # 先平方加深+再均值模糊
    # cv2.imshow('fangcha', reslut)   # 方差图
    # cv2.waitKey(0)
    # cv2.destroyWindow(all)
    return len(reslut), min_v, max_v, min_pt, max_pt  # 图片像素数 最小方差 最大方差 最小方差点位置 最大方差点位置


# 局部方差 测试
def jubu_fangcha_cs():
    inputpath = 'C:\\Users\\LML-YLC-PC\\Desktop\\新建文件夹\\brown\\'
    outputpath = 'C:\\Users\\LML-YLC-PC\\Desktop\\新建文件夹\\b_t1\\'
    filenames = os.listdir(inputpath)

    # g_c_max, g_c_min, g_c = -1, 300, 0
    fc_max, fc_min = 0, 0
    hit_nub = 0

    for file_name in filenames:
        # img = lml_imread.imread(inputpath + file_name)  # ,cv2.IMREAD_GRAYSCALE)
        path = inputpath + file_name
        # print(type(file_name))
        # print(file_name)

        img = lml_imread.imread(path)

        # print()
        long, min_v, max_v, min_pt, max_pt = jubu_fangcha(img)

        # g_c = max_v - min_v
        if fc_max < max_v:
            fc_max = max_v
        if fc_min > min_v or fc_min == 0:
            fc_min = min_v

        if 1:

            if max_v * 100 > 1.5:
                print("==========================================")
                hit_nub += 1
                # print(min_v, max_v, min_pt, max_pt)
                img = lml_imread.imread(inputpath + file_name)
                cv2.imencode('.bmp', img)[1].tofile(outputpath + file_name)
                # cv.circle(img, min_pt, 100, (225, 0, 225), 7, 8)
                cv.circle(img, max_pt, 100, (225, 0, 225), 7, 8)
                # cv2.imwrite(outputpath + file_name, img)

                list_file_name = list(file_name)
                # print(list_file_name)
                list_file_name.insert(-4, "_lab_")
                # print(list_file_name)
                # print(("".join(list_file_name)))

                cv2.imencode('.bmp', img)[1].tofile(outputpath + "".join(list_file_name))

        # img1 = img[0:1024, 0:1024]
        # img2 = img[0:1024, 1024:2048]
        # cv2.imwrite(outputpath + "L" + file_name, img1)
        # cv2.imwrite(outputpath + "R" + file_name, img2)
    print(fc_max)
    print(fc_min)
    print(hit_nub)


jubu_fangcha_cs()
