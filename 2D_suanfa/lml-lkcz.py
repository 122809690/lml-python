# https://blog.csdn.net/sunny2038/article/details/12889059
# 轮廓查找
import cv2

import lml_imread


def lml_lkcz(img, yuzhi=100):
    # 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 转二值化图
    ret, binary = cv2.threshold(gray, yuzhi, 255, cv2.THRESH_BINARY)
    # 进行边缘查找
    img_binary, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制边缘
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    return img


def lml_lkcz_test():
    img = lml_imread.imread('../DATA/img/轮廓查找/yt.jpg')
    cv2.imshow("img1", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("img2", gray)
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("img3", binary)

    # t1 = lml_time.get_time_ymd_hms_ms()
    img1, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # t2 = lml_time.get_time_ymd_hms_ms()
    # print(lml_time.get_time_yunsuan(t1, t2))
    # cv2.imshow("img4", img1)

    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

    # i = 6
    # cv2.drawContours(img, contours, i, (0, 0, 255), 3)
    # print(len(contours[i]))

    cv2.imshow("img4", img)
    # cv2.imshow("img5", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit(0)


if __name__ == '__main__':
    lml_lkcz_test()

# cv2.findContours()函数
# 函数的原型为
#
# cv2.findContours(image, mode, method[, contours[, hierarchy[, offset ]]])
# 返回两个值：contours：hierarchy。
# 参数
# 第一个参数是寻找轮廓的图像；
#
# 第二个参数表示轮廓的检索模式，有四种（本文介绍的都是新的cv2接口）：
#     cv2.RETR_EXTERNAL表示只检测外轮廓
#     cv2.RETR_LIST检测的轮廓不建立等级关系
#     cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
#     cv2.RETR_TREE建立一个等级树结构的轮廓。
#
# 第三个参数method为轮廓的近似办法
#     cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
#     cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
#     cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
#
# 返回值
# cv2.findContours()函数返回两个值，一个是轮廓本身，还有一个是每条轮廓对应的属性。
# contour返回值
# cv2.findContours()函数首先返回一个list，list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示。这个概念非常重要。在下面drawContours中会看见。通过
# print (type(contours))
# print (type(contours[0]))
# print (len(contours))
# 每个轮廓是一个ndarray，每个ndarray是轮廓上的点的集合。
# cv2.drawContours(img,contours,0,(0,0,255),3)
# cv2.drawContours(img,contours,1,(0,255,0),3)
# 单独绘制一个轮廓
# print (len(contours[0]))
# print (len(contours[1]))
# 输出两个轮廓中存储的点的个数
# 轮廓中并不是存储轮廓上所有的点，而是只存储可以用直线描述轮廓的点的个数，比如一个“正立”的矩形，只需4个顶点就能描述轮廓了(少见)。
#
# hierarchy返回值
# 此外，该函数还可返回一个可选的hiararchy结果，这是一个ndarray，其中的元素个数和轮廓个数相同，每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数。
# 通过
# print (type(hierarchy))
# print (hierarchy.ndim)
# print (hierarchy[0].ndim)
# print (hierarchy.shape)
# 得到
# 3
# 2
# (1, 2, 4)
# 可以看出，hierarchy本身包含两个ndarray，每个ndarray对应一个轮廓，每个轮廓有四个属性。


# cv2.drawContours()函数
# cv2.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset ]]]]])
# 第一个参数是指明在哪幅图像上绘制轮廓；
# 第二个参数是轮廓本身，在Python中是一个list。
# 第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓。
# 后面的参数很简单。其中thickness表明轮廓线的宽度，如果是-1（cv2.FILLED），则为填充模式。
