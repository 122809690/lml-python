#   sift特征提取  与模板匹配

# import xfeatures2d
import cv2
import cv2 as cv
# import cv2.cv2 as cv
import numpy as np

import lml_img_xuanzhuan
import lml_time

# import opencv

print('OpenCv version: ', cv2.__version__)
print('OpenCv path: ', cv2.__path__)


# OBR实时  xfeatures2d.surf更先进

# surf = cv.xfeatures2d.SURF_create(400)

# SIFT在OpenCV中的调用和具体实现(HELU版)
# https://www.cnblogs.com/jsxyhelu/p/7628664.html

def bgr_rgb(img):
    (r, g, b) = cv2.split(img)
    return cv2.merge([b, g, r])


def orb_detect(image_a, image_b):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image_a, None)
    kp2, des2 = orb.detectAndCompute(image_b, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(image_a, kp1, image_b, kp2, matches[:100], None, flags=2)
    return bgr_rgb(img3)


def sift_detect(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # for m, n in matches:
    #     print("m ",m.distance)
    #     print("n ",0.5 * n.distance)
    # good_qz = [[0.5 * n.distance - m.distance] for m, n in matches if m.distance < 0.5 * n.distance]
    # print(type(good_qz))
    # good_qz.sort(reverse=True)
    # print(good_qz)
    good = [[m] for m, n in matches if m.distance < 0.5 * n.distance]
    # print("goodT:\n",type(good))
    # print("good:\n",good)
    # print("good[0]:\n",good[0])
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    # cv.imshow("img3", img3)
    # cv.imshow("img2", img2)
    return bgr_rgb(img3)


#   sift算法学习测试 debug输出
if 0:
    def sift_test(img1):
        sift = cv2.xfeatures2d.SIFT_create()
        Keypoints, features = sift.detectAndCompute(img1, None)  # kp表示输入的关键点，ft表示输出的sift特征向量，通常是128维的
        # class KeyPoint{
        # Point2f pt; // 特征点坐标
        # float size; // 特征点邻域直径
        # float angle; // 特征点的方向，值为0~360，负值表示不使用
        # float response; // 特征点的响应强度，代表了该点是特征点的稳健度，可以用于后续处理中特征点排序
        # int octave; // 特征点所在的图像金字塔的层组
        # int class_id; // 用于聚类的id
        # }
        print("特征点个数： ", len(Keypoints))
        s_max, r_max = 0, 0
        s_min, r_min = -1, -1
        for k in Keypoints:
            # print(k.size)
            # print(k.response)
            # k.response = k.response * 20
            if k.size > s_max:
                s_max = k.size
            if k.size < s_min or s_min == -1:
                s_min = k.size
            if k.response > r_max:
                r_max = k.response
            if k.response < r_min or r_min == -1:
                r_min = k.response
        # print("特征点邻域直径size区间为", s_min, " ~ ", s_max)
        print("特征点邻域直径size区间为")
        print(s_min)
        print(s_max)
        # print("特征点相应强度response区间为", r_min, " ~ ", r_max)
        print("特征点相应强度response区间为")
        print(r_min)
        print(r_max)
        print("\n等待任意按键之后  ——> 关闭窗口并结束程序")

        cv.drawKeypoints(img1, Keypoints, img1)
        # cv.drawKeypoints(image,keypoints,outputimage,color,flags)
        # image: 也就是原始图片
        # keypoints：从原图中获得的关键点，这也是画图时所用到的数据
        # outputimage：输出 // 可以是原始图片
        # color：颜色设置，通过修改（b, g, r）的值, 更改画笔的颜色，b = 蓝色，g = 绿色，r = 红色。
        # flags：绘图功能的标识设置
        cv.namedWindow("match2", cv.WINDOW_NORMAL)
        cv.resizeWindow("match2", 800, 600)
        cv2.imshow("match2", img1)
        cv2.waitKey(0)


    image_yt = cv2.imdecode(np.fromfile('DATA/img/模板匹配算法/测试图/yt1.jpg', dtype=np.uint8), -1)
    sift_test(image_yt)
    cv2.waitKey(0)

#   sift算法案例
if 1:
    def sift_test():
        # if __name__ == "__main__":
        # image_yt = cv2.imdecode(np.fromfile('../DATA/img/模板匹配算法/测试图/yt1.jpg', dtype=np.uint8), -1)
        # image_mb = cv2.imdecode(np.fromfile('../DATA/img/模板匹配算法/测试图/mb1.jpg', dtype=np.uint8), -1)
        image_yt = cv2.imdecode(np.fromfile('../DATA/img/仪表识别/xianshiyibiao.jpg', dtype=np.uint8), -1)
        image_mb = cv2.imdecode(np.fromfile('../DATA/img/仪表识别/yuanbiao.jpg', dtype=np.uint8), -1)
        # cv2.imshow("yt", image_yt)
        # cv2.imshow("mb", image_mb)

        # image_ytxz = cv2.imdecode(np.fromfile('DATA/img/模板匹配算法/测试图/yt1-xz2.jpg', dtype=np.uint8), -1)

        time11 = lml_time.get_time_ymd_hms_ms()
        img = sift_detect(image_yt, image_mb)
        time12 = lml_time.get_time_ymd_hms_ms()

        cv.namedWindow("match1", cv.WINDOW_NORMAL)
        cv.resizeWindow("match1", 800, 600)
        cv2.imshow("match1", img)

        image_xz = image_yt.copy()
        image_xz = lml_img_xuanzhuan.img_xuanzhuang(image_xz, 105)
        time21 = lml_time.get_time_ymd_hms_ms()
        img2 = sift_detect(image_xz, image_mb)
        time22 = lml_time.get_time_ymd_hms_ms()

        cv.namedWindow("match2", cv.WINDOW_NORMAL)
        cv.resizeWindow("match2", 800, 600)
        cv2.imshow("match2", img2)

        print(time11)
        print(time12)
        print(" ")
        print(time21)
        print(time22)

        print("\n等待任意按键 关闭窗口结束程序")
        cv2.waitKey(0)
        # plt.imshow(img)
        # plt.show()


    sift_test()
