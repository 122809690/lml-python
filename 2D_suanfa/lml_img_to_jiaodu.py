# encoding:utf-8

# 仪表指针角度识别

import cv2 as cv
import numpy as np

import lml_imread


def line_detect_possible_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  # apertureSize是sobel算子大小，只能为1,3,5，7
    # HoughLinesP函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    # cv.imshow("edges", edges)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    # cv2.HoughLinesP()
    # 函数原型：
    # HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None)
    # image： 必须是二值图像，推荐使用canny边缘检测的结果图像；
    # rho: 线段以像素为单位的距离精度，double类型的，推荐用1.0
    # theta： 线段以弧度为单位的角度精度，推荐用numpy.pi / 180
    # threshod: 累加平面的阈值参数，int类型，超过设定阈值才被检测出线段，值越大，基本上意味着检出的线段越长，检出的线段个数越少。根据情况推荐先用100试试
    # lines：结果(=返回值)存储的位置，默认为None，因为一般用返回值直接做存储
    # minLineLength：线段以像素为单位的最小长度，根据应用场景设置
    # maxLineGap：同一方向上两条线段判定为一条线段的最大允许间隔（断裂），超过了设定值，则把两条线段当成一条线段，值越大，允许线段上的断裂越大，越有可能检出潜在的直线段

    # print(lines)
    # print(lines.shape)
    # print("==========================================")

    # # 在返回的坐标数组中添加线段长度
    # lines = np.insert(lines, 4, 0, axis=2)
    # maxL = (-1, -1)
    # # print(maxL[1])
    # # print(lines)
    # # print(lines.shape)
    # for i in range(len(lines)):
    #     x1, y1, x2, y2, ll = lines[i][0]
    #     lineL = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    #     if lineL >= maxL[0]:
    #         maxL = lineL, i
    #     # t = np.insert(lines[i], 1, lineL, axis=0)
    #     # # print(type(line))
    #     # # # np.insert(line, 0, ((x2-x1)**2+(y2-y1)**2)**0.5, axis=1)
    #     # # t = np.insert(lines, 0, ((x2-x1)**2+(y2-y1)**2)**0.5, axis=1)
    #     # t = np.insert(lines, 4, ((x2-x1)**2+(y2-y1)**2)**0.5, axis=2)
    #     lines[i][0][4] = lineL
    #     # print(t)
    #     # print("==========================================")
    #     # line[1] = ((x2-x1)**2+(y2-y1)**2)**0.5
    # # print(lines)
    # # print(lines.shape)
    #
    # # # 画图
    # # for line in lines:
    # #     print(line)  # 多维数组
    # #     x1, y1, x2, y2, ll = line[0]
    # #     cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # # cv.imshow("line_detect_possible_demo", image)
    # # cv.waitKey(0)
    #
    # # print(maxL[1])
    # # print(lines[maxL[1]])
    #
    # # 返回最长的一组线段
    # return lines[maxL[1]]
    return lines


def lines_choose(line, img):
    # 在返回的坐标数组中添加线段长度
    # lines = np.insert(line, 4, 0, axis=2)
    lines = np.insert(line, [1][0], -1, axis=1)
    # print(lines)
    # exit(1)
    maxL = (-1, -1, -1)
    # print(maxL[1])
    # print(lines)
    # print(lines.shape)
    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i][0]

        # 将远离图片中点的直线端点设置为x2y2  便于之后求角度的方向
        px = img.shape[0] / 2
        py = img.shape[1] / 2
        # print(point)
        ppd1 = ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
        ppd2 = ((px - x2) ** 2 + (py - y2) ** 2) ** 0.5
        print(ppd1)
        print(ppd2)
        if ppd1 > ppd2:
            x2, y2, x1, y1 = lines[i][0]

        # exit(0)
        lineL = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        angleL = np.rad2deg(np.arctan2(x2 - x1, y2 - y1))
        if lineL >= maxL[0]:
            maxL = lineL, i, angleL
        # t = np.insert(lines[i], 1, lineL, axis=0)
        # # print(type(line))
        # # # np.insert(line, 0, ((x2-x1)**2+(y2-y1)**2)**0.5, axis=1)
        # # t = np.insert(lines, 0, ((x2-x1)**2+(y2-y1)**2)**0.5, axis=1)
        # t = np.insert(lines, 4, ((x2-x1)**2+(y2-y1)**2)**0.5, axis=2)
        # 在第二组数据第一位加入直线长度
        lines[i][1][0] = lineL
        # 在第二组数据第二位加入直线角度
        lines[i][1][1] = angleL

        # print(t)
        # print("==========================================")
        # line[1] = ((x2-x1)**2+(y2-y1)**2)**0.5
    # print(lines)
    # print(lines.shape)

    # # 画图
    # for line in lines:
    #     print(line)  # 多维数组
    #     x1, y1, x2, y2, ll = line[0]
    #     cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv.imshow("line_detect_possible_demo", image)
    # cv.waitKey(0)

    # print(maxL[1])
    # print(lines[maxL[1]])
    # print(lines)

    # 返回最长的一组线段
    return lines[maxL[1]]
    # return lines


if __name__ == '__main__':
    # path = '..DATA/img/仪表识别/yuanbiao.png'

    # img = lml_imread.imread('../DATA/img/轮廓查找/yt.jpg')
    img = lml_imread.imread('../DATA/img/仪表识别/yuanbiao.png')
    # img = lml_imread.imread(path)

    lines = line_detect_possible_demo(img)
    line = lines_choose(lines, img)
    print(line)
