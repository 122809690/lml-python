# -*- coding:utf-8 -*-

#   sift特征提取  与模板匹配

# import xfeatures2d
import cv2
import cv2 as cv

import lml_img_xuanzhuan
import lml_qiujiajiao
import lml_qiujuli
import lml_time

# import cv2.cv2 as cv

# import opencv

print('OpenCv version: ', cv2.__version__)
print('OpenCv path: ', cv2.__path__)

# OBR实时  xfeatures2d.surf更先进

# surf = cv.xfeatures2d.SURF_create(400)

# SIFT在OpenCV中的调用和具体实现(HELU版)
# https://www.cnblogs.com/jsxyhelu/p/7628664.html


# coding=utf-8

# 仿射变换

import numpy as np
import cv2


def goodpoints_shaixuan(goodpoints, keypoints1, keypoints2, img_yt, img_mb, times=1):
    # 获取具体坐标
    ytp = [[keypoints1[i[0].queryIdx].pt] for i in goodpoints]
    mbp = [[keypoints2[i[0].queryIdx].pt] for i in goodpoints]
    # print(ytp)
    # print([i[0][0] for i in ytp])
    # ytpx =[i[0][0] for i in ytp]

    # 分别求横纵坐标的均值 [584.58138275 569.50188065]
    # sum_point_yt = np.mean(ytpx)
    sum_point_yt = np.mean([i[0][0] for i in ytp]), np.mean([i[0][1] for i in ytp])
    sum_point_mb = np.mean([i[0][0] for i in mbp]), np.mean([i[0][1] for i in mbp])

    # 筛选方案二
    # 求同位置在原图和模板两个点对  其和各自点（图像匹配点重点）的角度差 的比
    # times = 3
    # dis = np.array([])
    deg_p_mbyt = np.array([])

    dissum = 0
    for i in range(len(goodpoints)):
        deg_p_mbyt = lml_qiujiajiao.onelines(mbp[i][0], ytp[i][0])

    pdeg_max = np.argmax(deg_p_mbyt).astype(int)

    # 删除times个差距最大的映射点对
    # times = int(len(goodpoints)/10 +1)
    for i in range(times):
        goodpoints = np.delete(goodpoints, np.argmax(deg_p_mbyt).astype(int), axis=0)
        # goodpoints = goodpoints.pop(np.argmax(dis))
        # goodpoints = goodpoints.pop(np.argmin(dis))
        # print(goodpoints)
        # times -= 1
    # print(goodpoints.shape)

    goodpoints = goodpoints.tolist()
    # print(goodpoints)

    return goodpoints


def points_paixu(pts):
    (p1, p2, p3, p4) = pts
    # 将x按顺序排列
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # 左边两个点
    left_most = x_sorted[:2, :]
    # 右边两个点
    right_most = x_sorted[2:, :]

    # 按y排列，找到最左上角的值
    left_top_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_top_most

    right_bottom_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_bottom_most

    dst_px = np.array([tl, tr, br, bl])
    # print(np.argwhere(dst_px==p1))
    # print(np.argwhere(dst_px==p1)[0][0])
    # print(np.argwhere(dst_px==p2))
    # print(np.argwhere(dst_px==p2)[0][0])
    # print(np.argwhere(dst_px==p3))
    # print(np.argwhere(dst_px==p4))
    # print(pts)
    # print(dst_px)
    dst_ch = [(np.argwhere(dst_px == p1)[0][0]), (np.argwhere(dst_px == p2)[0][0]), (np.argwhere(dst_px == p3)[0][0]),
              (np.argwhere(dst_px == p4)[0][0])]

    return dst_px, dst_ch


def goodpoint_quchong(goodpq, kp1, kp2, xs_flag=1):
    # 筛选方案一
    # 分开求原图与模板上各点间的距离 排除距离太近的点
    ytp = [[kp1[i[0].queryIdx].pt] for i in goodpq]
    mbp = [[kp2[i[0].trainIdx].pt] for i in goodpq]

    list_del = []

    # print("ytplen",len(ytp))
    # print("mbplen",len(mbp))

    for i in range(len(mbp)):
        for j in range(len(mbp)):
            # print(lml_qiujuli.point_distance(mbp[i][0],mbp[j][0]))
            if i != j and lml_qiujuli.point_distance(mbp[i][0], mbp[j][0]) < xs_flag and i not in list_del:
                list_del.append(i)

    for i in range(len(ytp)):
        for j in range(len(ytp)):
            # print(lml_qiujuli.point_distance(ytp[i][0],ytp[j][0]))
            if i != j and lml_qiujuli.point_distance(ytp[i][0], ytp[j][0]) < xs_flag and i not in list_del:
                list_del.append(i)
            if i != j and lml_qiujuli.point_distance(ytp[i][0], ytp[j][0]) < xs_flag and i not in list_del:
                list_del.append(i)

    list_del.sort(reverse=True)
    # print(list_del)
    # print("len",len(goodpq))

    for index in list_del:
        goodpq.pop(index)

    return goodpq
    # print(goodpj)


def order_points(pts):
    # print("pts\n", pts)
    # 初始化坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 获取左上角和右下角坐标点

    s = pts.sum(axis=2)
    # print("s\n", s)
    # s = pts.sum(axis=0)
    # print("s\n", s)
    rect[0] = pts[np.argmin(s)]
    # print("r0\n", rect[0])
    # print("r00\n", rect[0][0])
    # print("s\n", s)
    # print("p\n", pts[np.argmax(s2)])
    rect[2] = pts[np.argmax(s)]
    # print("r2\n", rect[2])

    # print("r\n", rect)

    # 分别计算左上角和右下角的离散差值
    diff = np.diff(pts, axis=2)
    # print("d\n",diff)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(img_yt, img_mb, dst_yt, dst_mb):
    # 获取坐标点，并将它们分离开来
    # rect = order_points(dst_yt)
    # (tl, tr, br, bl) = rect
    # print(rect)

    # # 计算新图片的宽度值，选取水平差值的最大值
    # widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    # widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # maxWidth = max(int(widthA), int(widthB))
    #
    # # 计算新图片的高度值，选取垂直差值的最大值
    # heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    # heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # maxHeight = max(int(heightA), int(heightB))

    maxWidth = img_mb.shape[0]
    maxWidth = img_yt.shape[0]
    maxHeight = img_mb.shape[1]
    maxHeight = img_yt.shape[1]

    # 构建新图片的4个坐标点
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    # 获取仿射变换矩阵并应用它
    # M = cv2.getPerspectiveTransform(rect, dst)
    M = cv2.getPerspectiveTransform(dst_yt, dst_mb)
    # M = cv2.getPerspectiveTransform(dst_mb, dst_yt)
    # 进行仿射变换
    # warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warped = cv2.warpPerspective(img_yt, M, (img_mb.shape[0], img_mb.shape[1]))

    # 返回变换后的结果
    return warped


def bgr_rgb(img):
    (r, g, b) = cv2.split(img)
    return cv2.merge([b, g, r])


def sift_detect(img_yt, img_mb, distance_flag=0.6):  # (img1原图  img2模板)
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img_yt, None)
    kp2, des2 = sift.detectAndCompute(img_mb, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # 返回最佳的两个点

    # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
    # trainIdx：样本图像的特征点描述符下标, 同时也是描述符对应特征点的下标。
    # distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。

    # flag_it = 0.6
    # 最佳匹配点误差值 次佳匹配点误差值 之比 权重flag
    # 需要比这个值低 即次点误差大 最佳点误差小 才是good点
    # 数值增加 匹配点变多

    # good_qz = [[m.distance / n.distance] for m, n in matches if m.distance / n.distance < distance_flag]
    # # and kp1[m.queryIdx] !=
    # # print(type(good_qz))
    # good_qz.sort()
    # # print(good_qz)
    # # print(type(good_qz[0]))
    # # print(len(good_qz))
    # # good = [[m] for m, n in matches if   m.distance / 0.5 * n.distance <= good_qz[int(len(good_qz)/50-1)][0]]
    # good = [[m] for m, n in matches if m.distance / n.distance < good_qz[int(len(good_qz) - 1)][0]]

    good = [[m] for m, n in matches if matches if m.distance / n.distance < distance_flag]

    imgt = cv2.drawMatchesKnn(img_yt, kp1, img_mb, kp2, good, None, flags=2)
    cv.imshow("imgt1", imgt)
    good = goodpoint_quchong(good, kp1, kp2)

    imgt = cv2.drawMatchesKnn(img_yt, kp1, img_mb, kp2, good, None, flags=2)
    cv.imshow("imgt2", imgt)

    # good_t = [[m] for m, n in matches if m.distance / n.distance < distance_flag]
    #
    # imgt = cv2.drawMatchesKnn(img_yt, kp1, img_mb, kp2, good_t , None, flags=2)
    # cv.imshow("imgt",imgt)
    #
    # img_d = img_mb.copy()
    # for i in range(len(good_t)):
    #     # print("i",kp2[good_t[i][0].trainIdx].pt)
    #     cv.circle(img_d, (int(kp2[good_t[i][0].trainIdx].pt[0]),int(kp2[good_t[i][0].trainIdx].pt[1])), 10, (0, 0, 255), thickness=-1)
    # cv.imshow("imgt-d", img_d)

    mbp = [[kp2[i[0].queryIdx].pt] for i in good]

    img_mbd = img_mb.copy()
    for i in mbp:
        # print("mbp",i[0][0],i[0][1])
        cv.circle(img_mbd, (int(i[0][0]), int(i[0][1])), 10, (0, 0, 255), thickness=-1)
    cv.imshow("mbd", img_mbd)

    # 获取前四个特征点坐标
    box_yt = np.array([[kp1[i[0].queryIdx].pt] for i in good]).astype(np.float32)
    # print("boxt:\n",box_yt)
    # print("boxt-type:\n",box_yt.shape)
    dst_yt = np.array(box_yt[:, 0, :])
    # print(dst_yt)

    # print(type(dst_yt))

    # 将点集筛选为四边形的四个顶点
    pzs = pys = pzx = pyx = np.array([0.0, 0.0])
    fspoint = [0, 0, 0, 0]

    for i in range(len(dst_yt)):
        if bool(dst_yt[i][0] + dst_yt[i][1] < pzs[0] + pzs[1]) or (any(pzs == (0.0, 0.0))):
            pzs = dst_yt[i]
            fspoint[0] = i
        if bool(dst_yt[i][0] + dst_yt[i][1] > pyx[0] + pyx[1]) or (any(pyx == (0.0, 0.0))):
            pyx = dst_yt[i]
            fspoint[2] = i
        if bool(dst_yt[i][0] - dst_yt[i][1] > pys[0] - pys[1]) or (any(pys == (0.0, 0.0))):
            pys = dst_yt[i]
            fspoint[1] = i
        if bool(-dst_yt[i][0] + dst_yt[i][1] > -pzx[0] + pzx[1]) or (any(pzx == (0.0, 0.0))):
            pzx = dst_yt[i]
            fspoint[3] = i

    dst_yt_sx = np.array([pzs, pys, pyx, pzx])

    img_fs = img_yt.copy()
    for i in range(len(dst_yt_sx)):
        # print("dst",dst_yt_sx[i][0], dst_yt_sx[i][1])
        cv.circle(img_fs, (dst_yt_sx[i][0], dst_yt_sx[i][1]), 10, (0, 0, (255 / 4) * i), thickness=-1)
    cv.imshow("fs-dian", img_fs)

    box_mb = np.array([[kp2[i[0].trainIdx].pt] for i in good]).astype(np.float32)
    # print("boxtm:\n", box_mb)
    # print("boxm-type:\n", box_mb.shape)
    dst_mb = box_mb[:, 0, :]
    # dst_mbpx, dst_mbch = points_paixu(dst_mb)
    # print(dst_mbch)
    dst_mb_sx = np.array([[dst_mb[i]] for i in fspoint])
    # dst_mb_sx = np.delete(dst_mb, list(min_min_px2[4:]), axis=0)

    # 根据特征点方向手动做原图旋转预处理 不然仿射变换效果拉胯

    # warped2 = four_point_transform(img1, img2, dst_ytpx, dst_mbpx)
    warped2 = four_point_transform(img_yt, img_mb, dst_yt_sx, dst_mb_sx)
    cv2.imshow("fsbh", warped2)

    # print(good)
    # kp2_good = [[kp2[i]]for i in good ]

    # exit()
    img3 = cv2.drawMatchesKnn(img_yt, kp1, img_mb, kp2, good, None, flags=2)

    return img3


def sift_jioadu(img_yt, img_mb, distance_flag=0.6):  # (img1原图  img2模板)
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img_yt, None)
    kp2, des2 = sift.detectAndCompute(img_mb, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 返回最佳的两个点

    # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
    # trainIdx：样本图像的特征点描述符下标, 同时也是描述符对应特征点的下标。
    # distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。

    goodp = [[m] for m, n in matches if distance_flag * n.distance - m.distance > 0]

    goodp = goodpoint_quchong(goodp, kp1, kp2)

    good_sum = [[m[0].distance] for m in goodp]
    # # print(type(good_qz))

    good_sum.sort(reverse=False)
    # print(good_sum)
    # exit()
    # print(good_sum[int(len(good_sum)/2)] )
    # goodpj = [m for m, n in matches if distance_flag * n.distance - m.distance > good_sum[int(len(good_sum)/2)][0]]

    # print("gp",goodp)
    good = [m[0] for m in goodp if m[0].distance <= good_sum[3][0]]

    # print("g",good)

    # box_mb = np.array([[kp2[i[0].trainIdx].pt] for i in good]).astype(np.float32)
    # dst_mb = box_mb[:, 0, :]

    # box_yt = np.array([[kp1[i[0].queryIdx].pt] for i in good]).astype(np.float32)
    # dst_yt = box_yt[:, 0, :]

    # 获取特征点坐标
    box_mb = np.array([[kp2[i.trainIdx].pt] for i in good]).astype(np.float32)
    # print("boxtm:\n", box_mb)
    # print("boxm-type:\n", box_mb.shape)
    dst_mb = box_mb[:, 0, :]
    # dst_mbpx, dst_mbch = points_paixu(dst_mb)
    # print(dst_mb)

    box_yt = np.array([[kp1[i.queryIdx].pt] for i in good]).astype(np.float32)
    # print("boxt:\n",box_yt)
    # print("boxt-type:\n",box_yt.shape)
    dst_yt = box_yt[:, 0, :]

    # 根据误差值最小的四组特征点对  连线的方向差求旋转角  做原图旋转预处理 不然仿射变换效果拉胯
    xzj1 = lml_qiujiajiao.lines_orientation1(dst_yt[0], dst_yt[1], dst_mb[0], dst_mb[1])
    xzj2 = lml_qiujiajiao.lines_orientation1(dst_yt[2], dst_yt[3], dst_mb[2], dst_mb[3])
    if xzj1 * xzj2 < 0:
        xzj1 = -(360 - xzj1)
    return (xzj1 + xzj2) / 2


#   sift算法案例

def sift_test():
    # if __name__ == "__main__":
    # image_yt = cv2.imdecode(np.fromfile('../DATA/img/模板匹配算法/测试图/yt1.jpg', dtype=np.uint8), -1)
    # image_mb = cv2.imdecode(np.fromfile('../DATA/img/模板匹配算法/测试图/mb1.jpg', dtype=np.uint8), -1)
    image_yt = cv2.imdecode(np.fromfile('../DATA/img/仪表识别/xianshiyibiao.jpg', dtype=np.uint8), -1)
    # image_mb = cv2.imdecode(np.fromfile('../DATA/img/仪表识别/yuanbiao.jpg', dtype=np.uint8), -1)
    image_mb = cv2.imdecode(np.fromfile('../DATA/img/仪表识别/kongbiao.jpg', dtype=np.uint8), -1)
    cv2.imshow("yt", image_yt)
    cv2.imshow("mb", image_mb)

    # image_ytxz = cv2.imdecode(np.fromfile('DATA/img/模板匹配算法/测试图/yt1-xz2.jpg', dtype=np.uint8), -1)

    time11 = lml_time.get_time_ymd_hms_ms()

    xzj = sift_jioadu(image_yt, image_mb)
    img_xz = lml_img_xuanzhuan.img_xuanzhuang(image_yt, xzj)
    xzj2 = xzj + sift_jioadu(img_xz, image_mb)
    img_xz2 = lml_img_xuanzhuan.img_xuanzhuang(image_yt, xzj2)
    xzj3 = xzj2 + sift_jioadu(img_xz2, image_mb)
    img_xz3 = lml_img_xuanzhuan.img_xuanzhuang(image_yt, xzj3)
    # print("xzj",xzj,xzj2, xzj3)

    img = sift_detect(img_xz3, image_mb)

    # img = sift_detect(image_yt, image_mb)
    time12 = lml_time.get_time_ymd_hms_ms()

    cv.namedWindow("sift", cv.WINDOW_NORMAL)
    cv.resizeWindow("sift", 800, 600)
    cv2.imshow("sift", img)

    # print("\n等待任意按键 关闭窗口结束程序")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if 1:
    sift_test()
