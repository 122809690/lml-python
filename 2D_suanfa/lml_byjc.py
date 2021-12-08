#   边缘检测 基于Canny

import cv2

import lml_imread


def CannyThreshold(lowThreshold):
    # 高斯平滑处理原图像降噪
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    cv2.imshow('detected_edges', detected_edges)
    # Canny边缘检测
    detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
    cv2.imshow('Canny', detected_edges)
    dst = cv2.bitwise_and(img, img, mask=detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('canny demo', dst)
    return dst


lowThreshold = 50
max_lowThreshold = 100
ratio = 3
kernel_size = 3

img = lml_imread.imread('../DATA/img/轮廓查找/yt.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('canny demo')

cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyThreshold)

# cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold,
#                    CannyThreshold(lowThreshold, gray, img, ratio, kernel_size))

cv2.imshow('canny demo', img)

CannyThreshold(lowThreshold)  # initialization 预览窗口预存

cv2.waitKey(0)
cv2.destroyAllWindows()
