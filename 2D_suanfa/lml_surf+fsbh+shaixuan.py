# -*- coding:utf-8 -*-

#   sift特征提取  与模板匹配

# import xfeatures2d
import cv2
import cv2 as cv

import lml_img_xuanzhuan
import lml_qiujiajiao
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
    # print(goodpoints)
    # print(goodpoints)
    # print("len",len(goodpoints))
    # print(goodpoints)
    # print(goodpoints[0][0].queryIdx)
    # print([[i[0].queryIdx]for i in goodpoints])
    # print([[keypoints1[i[0].queryIdx].pt]for i in goodpoints])

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
    # sum_point_mb = np.mean(goodpoints, axis=1)  # 分别求横纵坐标的均值 [584.58138275 569.50188065]
    # print(sum_point_yt)

    # 舍去  有可能方向不同但是距离差不多 换成角度试试
    # 求同位置在原图和模板两个点对  其和各自点（图像匹配点重点）的距离差 的比
    # dis = np.array([])
    # dissum = 0
    # for i in range(len(goodpoints)):
    #     # print(ytp[i][0])
    #     dis_p_yt = lml_qiujuli.point_distance(ytp[i][0],sum_point_yt)
    #     dis_p_mb = lml_qiujuli.point_distance(mbp[i][0],sum_point_mb)
    #     dissum += dis_p_yt/dis_p_mb
    #     dis = np.append(dis,dis_p_yt/dis_p_mb)
    # dissum = dissum/len(goodpoints)
    # print(dis)
    #
    # for i in range(len(goodpoints)):
    #     # print(ytp[i][0])
    #     dis[i] = abs(dis[i]/dissum - 1)
    # print(dis_p_yt, dis_p_mb, dis)

    # print(dis)

    # 筛选方案二
    # 求同位置在原图和模板两个点对  其和各自点（图像匹配点重点）的角度差 的比
    # times = 3
    # dis = np.array([])
    deg_p_mbyt = np.array([])

    dissum = 0
    for i in range(len(goodpoints)):
        # print(ytp[i][0])
        # dis_p_yt = lml_qiujiajiao.onelines(ytp[i][0],sum_point_yt)
        # dis_p_mb = lml_qiujiajiao.onelines(mbp[i][0],sum_point_mb)
        deg_p_mbyt = lml_qiujiajiao.onelines(mbp[i][0], ytp[i][0])
        # ori = abs(dis_p_yt-dis_p_mb)
        # print(ori)
        # dis = np.append(dis,abs(dis_p_yt-dis_p_mb))
        # dis = np.append(dis,abs(dis_p_mbyt))
        # print(dis_p_yt, dis_p_mb, dis)
    # print(deg_p_mbyt)
    # dis = dis_p_mbyt

    pdeg_max = np.argmax(deg_p_mbyt).astype(int)
    # pdis_min=np.argmin(dis).astype(int)
    # print(deg_p_mbyt)
    # print(pdis_min)
    # print(type(max))
    # print(np.argmax(dis))
    # min=np.argmin(dis)
    # print(np.argmin(dis))
    #
    # print(goodpoints)
    # print(goodpoints[0])
    # print(goodpoints[max])
    # print("=======================")
    # print(goodpoints.pop(max))
    # print(goodpoints.pop(min))
    # print(goodpoints)

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

    # # exit(0)
    # x1 = []
    # x2 = []
    # tuple_dim = (1.,)
    # # print(len(points))
    # # fe = 0
    # for i in goodpoints():
    #     # fe += 1
    #     # print(i)    #[<DMatch 0000021AC48EB730>]
    #     # print(i[0]) #<DMatch 0000021AC48EB730>
    #     tuple_x1 = keypoints1[i[0].queryIdx].pt + tuple_dim
    #     tuple_x2 = keypoints2[i[0].trainIdx].pt + tuple_dim
    #     # print(keypoints1[i[0].queryIdx].pt) # (639.863037109375, 603.8123779296875)
    #     # print(tuple_x1)                     # tuple_x1 =  (639.863037109375, 603.8123779296875, 1.0)
    #     x1.append(tuple_x1)
    #     x2.append(tuple_x2)


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
    # dst_mb = pts_mb[:,0,:]
    # dst_yt = pts_yt[:,0,:]

    # dst_mb = points_paixu(dst_mb).astype(np.float32)

    # dst_yt = points_paixu(dst_yt).astype(np.float32)

    # print("dst\n",dst)
    # print(dst.shape)
    # print(type(dst))
    # print(dst[0])
    # print(type(dst[0]))
    # print(type(dst[0][0]))
    # print("dstyt\n",dst_yt)
    # print(dst_yt.shape)
    # print(type(dst_yt))
    # print(dst_yt[0])
    # print(type(dst_yt[0]))
    # print(type(dst_yt[0][0]))
    # dst_yt = dst_yt.astype(np.float32)
    # print(type(dst_yt[0][0]))
    # print("dstmb\n",dst_mb)
    # print(dst_mb.shape)
    # print("rect\n",rect)
    # print(rect.shape)
    # # print(dst1)
    # print(dst1.shape)
    # print(rect)

    # 获取仿射变换矩阵并应用它
    # M = cv2.getPerspectiveTransform(rect, dst)
    M = cv2.getPerspectiveTransform(dst_yt, dst_mb)
    # M = cv2.getPerspectiveTransform(dst_mb, dst_yt)
    # 进行仿射变换
    # warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warped = cv2.warpPerspective(img_yt, M, (img_mb.shape[0], img_mb.shape[1]))

    # 返回变换后的结果
    return warped


if 0:
    if __name__ == '__main__':
        # image = cv2.imread("../DATA/img/仪表识别/fsbh.png")
        # image = lml_imread.imread('../DATA/img/仪表识别/f2.png')
        image = lml_imread.imread('../DATA/img/仪表识别/fsbh2.png')
        canny = cv2.Canny(image, 50, 150, 3)
        cv2.imshow("1", canny)
        # cv2.waitKey(0)

        # 寻找轮廓
        fcr = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(fcr.shape())
        # print(len(fcr))
        image2, contours, hier = fcr
        maxarea = 0
        maxint = 0
        i = 0
        for c in contours:
            if cv2.contourArea(c) > maxarea:
                maxarea = cv2.contourArea(c)
            maxint = i
            i += 1
        # 绘制矩形
        # (x,y,w,h) = cv2.boundingRect(contours[maxint]) #外接矩形
        # rect = cv2.minAreaRect(contours[maxint]) #最小外接矩形 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        # cv2.rectangle(canny,(x,y),(x+w,y+h),(255,255,0),4)
        # box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x 获取最小外接矩形的4个顶点
        # box = np.int0(box)

        box = cv2.approxPolyDP(contours[maxint], 15, True)  # 多边形拟合 True 代表封闭
        # print(box.shape)
        # print(box)

        poly = np.zeros(canny.shape)
        cv2.polylines(poly, [box], True, (255, 0, 0))  # 连线

        cv2.imshow("2", poly)
        # cv2.waitKey()
        # 对原始图片进行变换
        # pts = np.array([(x,y),(x+w,y),(x+w,y+h),(x,y+h)], dtype = "float32")

        # 仿射变换 box=需要变换的四个点坐标(只有左上右下也行？)
        warped = four_point_transform(image, image, box, box)

        # 结果显示
        cv2.imshow("Original", image)
        cv2.imshow("Warped", warped)
        cv2.waitKey(0)


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


def sift_detect(img1, img2):  # (img1原图  img2模板)
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # 返回最佳的两个点

    # print(type(matches)) # <class 'list'>

    # exit(0)

    # matches_t = numpy.array(matches)
    # List转Numpy：numpy.array(list)
    # Numpy转List：array.tolist()
    # print(matches_t[:,:1])
    # print(matches_t[:,:1])
    # print(matches_t[:,:1][0][0].queryIdx)

    # for m in range(len(matches)):
    #     print(matches_t[:,:1][m][0].queryIdx)

    # c = (matches_t[:,:1][m][0].queryIdx).count()
    # print(c)
    # print("n ",0.5 * n.distance)

    # for m, n in matches:
    #     print("m ",m.distance)
    #     print("n ",0.5 * n.distance)

    # for matche,ma in matches:0‘
    #     print(matche)
    #     print(ma)
    #     print(matche.queryIdx)
    #     print(matche.trainIdx)
    #     print(matche.distance)
    #     print(ma.queryIdx)
    #     print(ma.trainIdx)
    #     print(ma.distance)
    # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
    # trainIdx：样本图像的特征点描述符下标, 同时也是描述符对应特征点的下标。
    # distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
    # for m, n in matches:
    #     print("m ",m.distance)
    #     print("n ",0.5 * n.distance)

    flag_it = 0.6
    # 最佳匹配点误差值 次佳匹配点误差值 之比 权重flag
    # 需要比这个值低 即次点误差大 最佳点误差小 才是good点
    # 数值增加 匹配点变多
    good_qz = [[m.distance / n.distance] for m, n in matches if m.distance / n.distance < flag_it]
    # and kp1[m.queryIdx] !=
    # print(type(good_qz))
    good_qz.sort()
    # print(good_qz)
    # print(type(good_qz[0]))
    # print(len(good_qz))
    # good = [[m] for m, n in matches if   m.distance / 0.5 * n.distance <= good_qz[int(len(good_qz)/50-1)][0]]
    good = [[m] for m, n in matches if m.distance / n.distance < good_qz[int(len(good_qz) - 1)][0]]

    # 异常点筛选
    # Max_num, good_F, inlier_points = lml_ransac.ransac(good,kp1,kp2,iter_num=1)
    good = goodpoints_shaixuan(good, kp1, kp2, img1, img2)

    # print(good)
    # print(len(good))
    # for i in good:
    #     # print(i)
    #     print(i[0].distance)
    # print("======================")

    # good_qz = [[ n.distance / m.distance  ] for m, n in matches if ( n.distance / m.distance ) > 5 ]
    # # print(type(good_qz))
    # good_qz.sort(reverse=True)
    # # print(good_qz)
    # print(len(good_qz))
    # print(good_qz[0])
    # print(type(good_qz[0]))
    # good = [[m] for m, n in matches if (n.distance / m.distance) <= good_qz[int(len(good_qz)-1)][0]]
    # print(good)

    # good = [[np.array(i)]for i in good]
    # print(good)
    # good[np.lexsort(good.T[, None])]
    # for m, n in matches:
    #     if 0.5 * n.distance - m.distance >= good_qz[3][0] and \
    #     len(good) >= 5 and \
    #     kp1[m.trainIdx].pt[0] :
    #         print(1)

    # good = [[m] for m, n in matches
    #         if  0.5 * n.distance - m.distance >= good_qz[3][0]
    #         # and [kp2[m.trainIdx].pt]
    #
    #
    #         ]
    # print("goodT:\n",type(good))
    # print("good:\n",good)
    # print("good[0]:\n",good[0])
    #
    # for m in range(len(good)):
    #     print(good[m][0].queryIdx)
    #     print(good[m][0].trainIdx)

    # print("kp1:\n",kp1)
    # print("kp1[0]:\n",kp1[0])
    # print("kp1[0].pt:\n",kp1[0].pt)

    # for i in good:
    # print(i[0].queryIdx)

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
    # print(pzs == (0.0,0.0))
    # print(type(pzs == (0.0,0.0)))
    # print(type(pzs == (0.0,0.0)))

    # print(dst_yt[0][0]+dst_yt[0][1] < pzs[0]+pzs[1])
    # print(type(dst_yt[0][0]+dst_yt[0][1] < pzs[0]+pzs[1]))
    for i in range(len(dst_yt)):
        # print(type(i[0]+i[1] < pzs[0]+pzs[1]))
        # print(type(bool(pzs == (0.0,0.0))))
        # print(dst_yt[i][0]+dst_yt[i][1])
        # print(pzs[0]+pzs[1])
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
    # print(dst_yt_sx)
    # print(fspoint)
    # exit(1)

    #
    # #获取每个点到其他点的最小值   有bug  有可能是一条直线上的四个点 舍弃方案。。。
    # min_all = [[[lml_qiujuli.point_distance(m[0],n[0])]for n in box_yt if all(m[0] != n[0])]for m in box_yt]
    # min_all = [[[lml_qiujuli.point_distance(box_yt[m][0],box_yt[n][0])]for n in range(len(box_yt)) if m != n]for m in range(len(box_yt))]
    # min_min = np.array([min(i) for i in min_all])
    # # for m in box_yt:
    # #     for n in box_yt:
    # #         lml_qiujuli.point_distance(m,n)
    # # print(min_all)
    # # print(min_min)
    #
    # # min_min_px = min_min.copy()
    # min_min_px = np.array(list(enumerate(min_min.copy())),dtype=object)
    # # print(type(min_min_px))
    # # min_min_px.sort(reverse=True)
    # # print(min_min_px)
    # min_min_px2 = np.lexsort(-min_min_px.T[1, None])
    # # print(min_min_px2)
    # # np.sort()
    #
    # # print(min_min_px2[4:])
    # # print(dst_yt[min_min_px2[4:]])
    # dst_yt_sx = np.delete(dst_yt, list(min_min_px2[4:]), axis=0)
    # # print(dst_yt_sx)

    # cv.drawKeypoints(img1, dst_yt_sx, img1)
    img_fs = img1.copy()
    for i in range(len(dst_yt_sx)):
        cv.circle(img_fs, (dst_yt_sx[i][0], dst_yt_sx[i][1]), 10, (0, 0, (255 / 4) * i), thickness=-1)
    cv.imshow("fs-dian", img_fs)

    # dst_yt, min_min_px2[4:])

    # print(dst_yt)
    # print(list(min_min_px2[:4]))
    # print()

    # exit()

    box_mb = np.array([[kp2[i[0].trainIdx].pt] for i in good]).astype(np.float32)
    # print("boxtm:\n", box_mb)
    # print("boxm-type:\n", box_mb.shape)
    dst_mb = box_mb[:, 0, :]
    # dst_mbpx, dst_mbch = points_paixu(dst_mb)
    # print(dst_mbch)
    dst_mb_sx = np.array([[dst_mb[i]] for i in fspoint])
    # dst_mb_sx = np.delete(dst_mb, list(min_min_px2[4:]), axis=0)

    # 根据特征点方向手动做原图旋转预处理 不然仿射变换效果拉胯
    # print(dst_mb[0][0])
    # print(dst_yt[0][1])
    # xzj = lml_qiujiajiao.lines_orientation1(dst_yt[0],dst_yt[1],dst_mb[0],dst_mb[1],1)
    # print(xzj)

    # img_ytxz = lml_img_xuanzhuan.img_xuanzhuang(img1, xzj)
    # kp3, des3 = sift.detectAndCompute(img_ytxz, None)
    # box_ytxz = np.array([[kp3[i[0].queryIdx].pt] for i in good]).astype(np.float32)
    # print("boxt:\n",box_yt)d
    # print("boxt-type:\n",box_yt.shape)
    # dst_ytxz = box_ytxz[:, 0, :]

    # dst_ytpx = dst_yt.copy()
    # for i in range(len(dst_mbch)):
    #     # print(i)
    #     dst_ytpx[dst_mbch[i]] = dst_yt[i]
    #
    # print(dst_yt)
    # print(dst_mb)
    # print(dst_ytpx)
    # print(dst_mbpx)

    # warped2 = four_point_transform(img1, img2, dst_ytpx, dst_mbpx)
    warped2 = four_point_transform(img1, img2, dst_yt_sx, dst_mb_sx)
    cv2.imshow("fsbh", warped2)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    # img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, inlier_points, None, flags=2)
    # cv2.imshow("ransac", img4)

    # cv.imshow("img3", img3)
    # cv.imshow("img2", img2)
    return bgr_rgb(img3)


def sift_jioadu(img1, img2):  # (img1原图  img2模板)
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # 返回最佳的两个点
    # print(type(matches)) # <class 'list'>

    # exit(0)

    # matches_t = numpy.array(matches)
    # List转Numpy：numpy.array(list)
    # Numpy转List：array.tolist()
    # print(matches_t[:,:1])
    # print(matches_t[:,:1])
    # print(matches_t[:,:1][0][0].queryIdx)

    # for m in range(len(matches)):
    #     print(matches_t[:,:1][m][0].queryIdx)

    # c = (matches_t[:,:1][m][0].queryIdx).count()
    # print(c)
    # print("n ",0.5 * n.distance)

    # for m, n in matches:
    #     print("m ",m.distance)
    #     print("n ",0.5 * n.distance)

    # for matche,ma in matches:0‘
    #     print(matche)
    #     print(ma)
    #     print(matche.queryIdx)
    #     print(matche.trainIdx)
    #     print(matche.distance)
    #     print(ma.queryIdx)
    #     print(ma.trainIdx)
    #     print(ma.distance)
    # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
    # trainIdx：样本图像的特征点描述符下标, 同时也是描述符对应特征点的下标。
    # distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
    # for m, n in matches:
    #     print("m ",m.distance)
    #     print("n ",0.5 * n.distance)
    good_qz = [[0.5 * n.distance - m.distance] for m, n in matches if m.distance < 0.5 * n.distance]
    # print(type(good_qz))
    good_qz.sort(reverse=True)
    # print(good_qz)
    good = [[m] for m, n in matches if 0.5 * n.distance - m.distance >= good_qz[3][0]]
    # good = [[m] for m, n in matches if 0.5 * n.distance - m.distance > 0]
    # print("goodT:\n",type(good))
    # print("good:\n",good)
    # print("good[0]:\n",good[0])
    #
    # for m in range(len(good)):
    #     print(good[m][0].queryIdx)
    #     print(good[m][0].trainIdx)

    # print("kp1:\n",kp1)
    # print("kp1[0]:\n",kp1[0])
    # print("kp1[0].pt:\n",kp1[0].pt)

    # for i in good:
    #     print(i[0].queryIdx)

    box_mb = np.array([[kp2[i[0].trainIdx].pt] for i in good]).astype(np.float32)
    dst_mb = box_mb[:, 0, :]

    box_yt = np.array([[kp1[i[0].queryIdx].pt] for i in good]).astype(np.float32)
    dst_yt = box_yt[:, 0, :]

    # 获取特征点坐标
    box_mb = np.array([[kp2[i[0].trainIdx].pt] for i in good]).astype(np.float32)
    # print("boxtm:\n", box_mb)
    # print("boxm-type:\n", box_mb.shape)
    dst_mb = box_mb[:, 0, :]
    # dst_mbpx, dst_mbch = points_paixu(dst_mb)
    # print(dst_mb)

    box_yt = np.array([[kp1[i[0].queryIdx].pt] for i in good]).astype(np.float32)
    # print("boxt:\n",box_yt)
    # print("boxt-type:\n",box_yt.shape)
    dst_yt = box_yt[:, 0, :]

    # print(dst_yt)
    # print(int(len(good)/2))

    # for i in range(int(len(good)/2)):
    #     print()

    # 根据误差值最小的四组特征点对  连线的方向差求旋转角  做原图旋转预处理 不然仿射变换效果拉胯
    # print(dst_mb[0][0])
    # print(dst_yt[0][1])
    xzj1 = lml_qiujiajiao.lines_orientation1(dst_yt[0], dst_yt[1], dst_mb[0], dst_mb[1])
    xzj2 = lml_qiujiajiao.lines_orientation1(dst_yt[2], dst_yt[3], dst_mb[2], dst_mb[3])
    # print(xzj1)
    # print(xzj2)
    # print((xzj1+xzj2)/2)
    if xzj1 * xzj2 < 0:
        xzj1 = -(360 - xzj1)
    # print(xzj1)
    # print(xzj2)
    # print((xzj1+xzj2)/2)
    return (xzj1 + xzj2) / 2

    # return xzj2
    # return xzj1
    #
    # img_ytxz = lml_img_xuanzhuan.img_xuanzhuang(img1, xzj)
    # kp3, des3 = sift.detectAndCompute(img_ytxz, None)
    # box_ytxz = np.array([[kp3[i[0].queryIdx].pt] for i in good]).astype(np.float32)
    # # print("boxt:\n",box_yt)d
    # # print("boxt-type:\n",box_yt.shape)
    # dst_ytxz = box_ytxz[:, 0, :]
    #
    # # dst_ytpx = dst_yt.copy()
    # # for i in range(len(dst_mbch)):
    # #     # print(i)
    # #     dst_ytpx[dst_mbch[i]] = dst_yt[i]
    # #
    # # print(dst_yt)
    # # print(dst_mb)
    # # print(dst_ytpx)
    # # print(dst_mbpx)
    #
    # # warped2 = four_point_transform(img1, img2, dst_ytpx, dst_mbpx)
    # warped2 = four_point_transform(img_ytxz, img2, dst_ytxz, dst_mb)
    # cv2.imshow("fsbh", warped2)
    #
    # img3 = cv2.drawMatchesKnn(img_ytxz, kp3, img2, kp2, good, None, flags=2)
    # # cv.imshow("img3", img3)
    # # cv.imshow("img2", img2)
    # return bgr_rgb(img3)


#   sift算法案例
if 1:
    def sift_test():
        # if __name__ == "__main__":
        # image_yt = cv2.imdecode(np.fromfile('../DATA/img/模板匹配算法/测试图/yt1.jpg', dtype=np.uint8), -1)
        # image_mb = cv2.imdecode(np.fromfile('../DATA/img/模板匹配算法/测试图/mb1.jpg', dtype=np.uint8), -1)
        image_yt = cv2.imdecode(np.fromfile('../DATA/img/仪表识别/xianshiyibiao.jpg', dtype=np.uint8), -1)
        image_mb = cv2.imdecode(np.fromfile('../DATA/img/仪表识别/yuanbiao.jpg', dtype=np.uint8), -1)
        # image_mb = cv2.imdecode(np.fromfile('../DATA/img/仪表识别/kongbiao.jpg', dtype=np.uint8), -1)
        cv2.imshow("yt", image_yt)
        cv2.imshow("mb", image_mb)

        # image_ytxz = cv2.imdecode(np.fromfile('DATA/img/模板匹配算法/测试图/yt1-xz2.jpg', dtype=np.uint8), -1)

        time11 = lml_time.get_time_ymd_hms_ms()

        xzj = sift_jioadu(image_yt, image_mb)

        img_xz = lml_img_xuanzhuan.img_xuanzhuang(image_yt, xzj)
        img = sift_detect(img_xz, image_mb)
        time12 = lml_time.get_time_ymd_hms_ms()

        cv.namedWindow("sift", cv.WINDOW_NORMAL)
        cv.resizeWindow("sift", 800, 600)
        cv2.imshow("sift", img)

        # image_xz = image_yt.copy()
        # image_xz = lml_img_xuanzhuan.img_xuanzhuang(image_xz, 120)
        # time21 = lml_time.get_time_ymd_hms_ms()
        # img2 = sift_detect(image_xz, image_mb)
        # time22 = lml_time.get_time_ymd_hms_ms()
        #
        # cv.namedWindow("match2", cv.WINDOW_NORMAL)
        # cv.resizeWindow("match2", 800, 600)
        # cv2.imshow("match2", img2)
        #
        # print(time11)
        # print(time12)
        # print(" ")
        # print(time21)
        # print(time22)

        # print("\n等待任意按键 关闭窗口结束程序")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # plt.imshow(img)
        # plt.show()


    sift_test()
