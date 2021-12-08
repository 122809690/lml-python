# 模板匹配

import cv2 as cv
import numpy as np

import lml_time


def template_demo():
    path_img_mb = "../DATA/img/模板匹配算法/测试图/mb1.jpg"
    # img_mb = cv.imread("DATA/img/mbpp/mb1.jpg")  # 模板
    img_mb = cv.imdecode(np.fromfile(path_img_mb, dtype=np.uint8), -1)
    # tpl = cv.imread("DATA/img/mbpp/yt1.jpg")  # 模板
    cv.namedWindow("MB", 0)
    cv.imshow("MB", img_mb)

    path_img_yt = "../DATA/img/模板匹配算法/测试图/yt1.jpg"
    img_yt = cv.imdecode(np.fromfile(path_img_yt, dtype=np.uint8), -1)
    # img_yt = cv.imread(path_img_yt)  # 图像
    cv.namedWindow('YT', cv.WINDOW_NORMAL)
    cv.imshow("YT", img_yt)

    # cv.namedWindow("target image", cv.WINDOW_NORMAL)
    # cv.resizeWindow("YT", 800, 800)

    # cv.waitKey()
    # time.sleep(5)

    # cv.imshow("target image", tpl)

    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]  # 3种模版匹配友法
    # https://zhuanlan.zhihu.com/p/110425960
    methods = [cv.TM_SQDIFF,  # 平方差匹配 method=CV_TM_SQDIFF 这类方法利用平方差来进行匹配,最好匹配为0.匹配越差,匹配值越大.
               cv.TM_SQDIFF_NORMED,  # 标准平方差匹配 method=CV_TM_SQDIFF_NORMED
               cv.TM_CCORR,  # 相关匹配 method=CV_TM_CCORR 这类方法采用模板和图像间的乘法操作,所以较大的数表示匹配程度较高,0标识最坏的匹配效果.
               cv.TM_CCORR_NORMED,  # 标准相关匹配 method=CV_TM_CCORR_NORMED
               cv.TM_CCOEFF,
               # 相关匹配 method=CV_TM_CCOEFF 这类方法将模版对其均值的相对值与图像对其均值的相关值进行匹配,1表示完美匹配,-1表示匹配很差,0表示没有任何相关性(随机序列).
               cv.TM_CCOEFF_NORMED  # 标准相关匹配 method=CV_TM_CCOEFF_NORMED
               ]  # 6种模版匹配法

    th, tw = img_mb.shape[:2]

    for md in methods:
        print(md)
        img_lab = img_yt.copy()
        # cv.imshow("yt-" + np.str_(md), img_yt)
        print(lml_time.get_time_ymd_hms_ms())
        result = cv.matchTemplate(img_lab, img_mb, md)
        print(lml_time.get_time_ymd_hms_ms())

        # if md == 0:
        # print(result)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED or md == cv.TM_SQDIFF:
            tl = min_loc
            ntl = max_loc
        else:
            tl = max_loc
            ntl = min_loc
        # if md == 0:
        # print(tl)
        br = (tl[0] + tw, tl[1] + th)  # br是矩形右下角的点的坐标
        br2 = (ntl[0] + tw, ntl[1] + th)
        # print(br)
        cv.rectangle(img_lab, tl, br, (0, 0, 255), 15)
        cv.rectangle(img_lab, ntl, br2, (255, 0, 0), 12)
        cv.namedWindow("match-" + np.str_(md), cv.WINDOW_NORMAL)
        cv.resizeWindow("match-" + np.str_(md), 400, 300)
        cv.imshow("match-" + np.str_(md), img_lab)


template_demo()
cv.waitKey(0)
cv.destroyAllWindows()

'''
matchTemplate：函数的完整表达：
matchTemplate(image, templ, method[, result[, mask]])
Image：参数表示待搜索源图像，必须是8位整数或32位浮点。
Templ：参数表示模板图像，必须不大于源图像并具有相同的数据类型。
Method：参数表示计算匹配程度的方法。
Result：参数表示匹配结果图像，必须是单通道32位浮点。如果image的尺寸为W x H，templ的尺寸为w x h，则result的尺寸为(W-w+1)x(H-h+1)。

minMaxLoc函数的完整表达：
minMaxLoc(src[, mask]，minVal, maxVal, minLoc, maxLoc)
src参数表示输入单通道图像。
mask参数表示用于选择子数组的可选掩码。
minVal参数表示返回的最小值，如果不需要，则使用NULL。
maxVal参数表示返回的最大值，如果不需要，则使用NULL。
minLoc参数表示返回的最小位置的指针（在2D情况下）； 如果不需要，则使用NULL。
maxLoc参数表示返回的最大位置的指针（在2D情况下）； 如果不需要，则使用NULL。
'''
