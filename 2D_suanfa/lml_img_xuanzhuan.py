#   图片旋转

import cv2
import numpy as np


# # 定义旋转矩阵，3个参数分别为：旋转中心，旋转角度，缩放比率
# M = cv2.getRotationMatrix2D(center, 45, 1.0)
# # 正式旋转，这样就得到了和原始图片img1不太一样的照片
# rotated = cv2.warpAffine(blurred, M, (width,height))


def img_xuanzhuang(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

# # image = cv2.imread('DATA/img/模板匹配算法/测试图/yt1.jpg')
# image = cv2.imdecode(np.fromfile('DATA/img/模板匹配算法/测试图/yt1.jpg', dtype=np.uint8), -1)
# angle = 45
# imag = rotate_bound(image, angle)
# cv2.namedWindow("xuanzhuang", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("xuanzhuang", 800, 800)
# cv2.imshow('xuanzhuang', imag)
# cv2.waitKey()


# import numpy as np  # 1
# import argparse  # 2
# import imutils  # 3
# import cv2  # 4
#
# ap = argparse.ArgumentParser()  # 5
# ap.add_argument("-i", "--image", required=True,
#                 help="Path to the image")  # 6
# args = vars(ap.parse_args())  # 7
#
# image = cv2.imread(args["image"])  # 8
# cv2.imshow("Original", image)  # 9
#
# (h, w) = image.shape[:2]  # 10
# center = (w // 2, h // 2)  # 11
#
# M = cv2.getRotationMatrix2D(center, 45, 1.0)  # 12
# rotated = cv2.warpAffine(image, M, (w, h))  # 13
# cv2.imshow("Rotated by 45 Degrees", rotated)  # 14
#
# M = cv2.getRotationMatrix2D(center, -90, 1.0)  # 15
# rotated = cv2.warpAffine(image, M, (w, h))  # 16
# cv2.imshow("Rotated by -90 Degrees", rotated)  # 17
#
# rotated = imutils.rotate(image, 180)  # 18
# cv2.imshow("Rotated by 180 Degrees", rotated)  # 19
# cv2.waitKey(0)  # 20
