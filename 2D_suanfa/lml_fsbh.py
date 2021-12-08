# coding=utf-8

# 仿射变换

import cv2
import numpy as np

import lml_imread


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


def four_point_transform(image, pts):
    # 获取坐标点，并将它们分离开来
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    print(rect)

    # 计算新图片的宽度值，选取水平差值的最大值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 计算新图片的高度值，选取垂直差值的最大值
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 构建新图片的4个坐标点
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    # 获取仿射变换矩阵并应用它
    M = cv2.getPerspectiveTransform(rect, dst)
    # 进行仿射变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后的结果
    return warped


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
    poly = np.zeros(canny.shape)
    cv2.polylines(poly, [box], True, (255, 0, 0))  # 连线

    cv2.imshow("2", poly)
    # cv2.waitKey()
    # 对原始图片进行变换
    # pts = np.array([(x,y),(x+w,y),(x+w,y+h),(x,y+h)], dtype = "float32")

    # 仿射变换 box=需要变换的四个点坐标(只有左上右下也行？)
    warped = four_point_transform(image, box)

    # 结果显示
    cv2.imshow("Original", image)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)
