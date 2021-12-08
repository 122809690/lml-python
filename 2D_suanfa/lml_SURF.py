#   surf特征提取  与模板匹配

# import xfeatures2d
import cv2
import cv2 as cv
import numpy as np

import lml_img_xuanzhuan
import lml_time

# noinspection PyUnresolvedReferences
print('OpenCv version: ', cv2.__version__)


# OBR实时  xfeatures2d.surf更先进

# surf由于专利影响
# 需要手动编译opencv  -D OPENCV_ENABLE_NONFREE =ON
# cmake vs运行库VC.exe  vs编译库vs_BuildTools.exe  win-SDK  win平台通用桌面开发工具
# 或者安装低版本的一套 conda create -n py37 python=3.7  pip install opencv-python==3.4.2.16  opencv-contrib-python==3.4.2.16
# https://blog.csdn.net/SiriusWilliam/article/details/104891856

# surf = cv.xfeatures2d.SURF_create(400)


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
    good = [[m] for m, n in matches if m.distance < 0.5 * n.distance]
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    return bgr_rgb(img3)


def surf_detect(img1, img2):
    surf = cv2.xfeatures2d.SURF_create()

    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [[m] for m, n in matches if m.distance < 0.5 * n.distance]
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    return bgr_rgb(img3)


def surf_detect_guolv(img1, img2, guolv_num):
    surf = cv2.xfeatures2d.SURF_create()

    kp1, des1 = surf.detectAndCompute(img1, None)
    # kp1, des1 = Keypoints_guolv(kp1, des1, guolv_num)
    kp2, des2 = surf.detectAndCompute(img2, None)
    # kp2, des2 = Keypoints_guolv(kp2, des2, guolv_num)

    # BF匹配器
    # BFMatcher.match()和BFMatcher.knnMatch()。
    # 第一个返回最匹配的，第二个方法返回k个最匹配的，k由用户指定。当我们需要多个的时候很有用。
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # bf.Match()函数的返回值是一个DMatch，DMatch是一个匹配之后的集合。
    # 后续需要使用 for m in matches
    # DMatch中的每个元素含有三个参数：
    # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
    # trainIdx：样本图像的特征点描述符下标, 同时也是描述符对应特征点的下标。
    # distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
    # for matche in matches:
    #     print(matche)
    #     print(matche.queryIdx)
    #     print(matche.trainIdx)
    #     print(matche.distance)
    # bf.knnMatch()返回一个多维度DMatch 如果k=2 则返回最匹配的两个点 后续需要使用 for m, n in matches
    # m固定比n小  即m点的匹配度更高
    # for m, n in matches:
    # if m.distance < n.distance:
    # print("=============================================")

    # good_list = np.array([[m.distance/n.distance] for m, n in matches])
    good_list = [[m.distance / n.distance] for m, n in matches]
    good_list.sort()
    good_flag = good_list[guolv_num - 1][0]  # 不知道为什么是二维数组  虽然第二维只有一个元素。。。
    # print("good_flag", good_flag)
    # print(type(good_flag))
    # exit(0)
    # for m, n in matches:
    #     print(type(m.distance/n.distance))
    #     exit(0)
    good = [[m] for m, n in matches if m.distance / n.distance <= good_flag]
    # print(len(matches))
    # print(len(good))
    # exit(0)
    # good = [[m] for m, n in matches if m.distance < 0.1 * n.distance]
    # cv2.drawMatches() 画匹配的结果   drawMatchesKnn() 画k个最匹配的。如果k=2，它会给每个关键点画两根匹配线
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    return bgr_rgb(img3)


# 失败  弃用方案
def Keypoints_guolv(Keypoints_in, features_in, glnum):  # 失败  弃用方案
    t1 = lml_time.get_time_ymd_hms_ms()
    Keypoints = Keypoints_in.copy()
    # features = features_in.copy()
    print("\nKeypoints_guolv    start")

    # print(type(Keypoints_in))
    # print(type(features_in))
    # print(features_in.shape)
    Keypoints_response = [i.response for i in Keypoints]
    Keypoints_response.sort()  # 排序 (cmp=None, key=None, reverse=False) cmp -- 可选参数, 如果指定了该参数会使用该参数的方法进行排序。
    # key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
    # reverse -- 排序规则，reverse = True 降序， reverse = False 升序（默认）。

    # print("Keypoints_response", Keypoints_response)
    flag = int(len(Keypoints_response) * glnum / 100)
    # print(len(Keypoints_response))
    # print(i)
    # Keypoints_response[flag]

    # for i in range(len(Keypoints)-1, -1, -1):   # 倒序删除因为列表总是‘向前移’，所以可以倒序遍历，即使后面的元素被修改了，
    #                                         # 还没有被遍历的元素和其坐标还是保持不变的, 避免遍历总数不变数组长度变小后的数组访问越界
    #                                         # 时间过长 舍去
    #     # print(Keypoints[i].response)
    #     # print(Keypoints_response[flag])
    #     # exit(0)
    #     if Keypoints[i].response < Keypoints_response[flag]:
    #         Keypoints.pop(i)
    #         # Keypoints = Keypoints - {i}，
    #         np.delete(features, i, axis=0)

    Keypoints2 = []  # l = list()
    Keypoints_flag = 0
    features2 = np.zeros(shape=((len(Keypoints_response) - flag), 64))  # 可以是shape(0,0) 不过只后每次插入会重新加载数组 浪费开销
    # print("features2 = ", features2)
    for i in range(len(Keypoints)):
        if Keypoints[i].response >= Keypoints_response[flag]:
            Keypoints2.append(Keypoints[i])
            # features2 = np.append(features_in[i])
            # features2 = np.insert(features2, i, features_in[0], axis=0)
            features2[Keypoints_flag] = features_in[i]
            Keypoints_flag += 1
            # features2 = np.insert(features2, i, features_in[i], axis=0)
            # print("features2", features2)
        # elif Keypoints[i].response == Keypoints_response[flag]:
        #     print("====================================================")
        # print("features_in[", i, "]=\n", features_in[i])
        # print("features2n=\n", features2)
        # print(i, Keypoints[i].response, len(features2))
        # print(features2.shape, features_in[i].shape)
    # print(features2.shape, features_in.shape)
    # exit(0)
    # print(len(Keypoints2))
    # print(len(features2))
    # print("Keypoints_flag=", Keypoints_flag)
    # print("flag=", flag)
    # print(features2[1])
    # print(features_in[1])
    print(features2.shape)
    print(features_in.shape)

    t2 = lml_time.get_time_ymd_hms_ms()

    print("yunsuan=", lml_time.get_time_yunsuan(t1, t2))
    return Keypoints2, features2

    # np.delete(ndarray, elements, axis)函数删除元素。此函数会沿着指定的轴从给定ndarray中删除给定的元素列表。
    # 对于秩为1的ndarray，不需要使用关键字axis。对于秩为2的ndarray，axis = 0表示选择行，axis = 1表示选择列
    # Y=[[1,2],[3,4]]  w = np.delete(Y, 0, axis=0)  v = np.delete(Y, [0,1], axis=1)


def sift_test(img):
    print("\nSIFT算法:")
    img1 = img.copy()
    sift = cv2.xfeatures2d.SIFT_create()
    Keypoints, features = sift.detectAndCompute(img1, None)  # kp表示输入的关键点，ft表示输出的sift特征向量，通常是128维的
    # class KeyPoint{
    # Point2f pt; // 特征点坐标
    # float size; // 特征点邻域直径    # 该点直径的大小
    # float angle; // 特征点的方向，值为0~360，负值表示不使用
    # float response; // 特征点的响应强度，代表了该点是特征点的稳健度，可以用于后续处理中特征点排序  # 即是角点的可能性大小
    # int octave; // 特征点所在的图像金字塔的层组 代表是从金字塔哪一层提取的得到的数据。
    # int class_id; // 用于聚类的id 当要对图片进行分类时，我们可以用class_id对每个特征点进行区分，未设定时为-1，需要靠自己设定
    # }

    print("特征点个数： ", len(Keypoints))
    # print("特 ： ", type(Keypoints))
    # print("特征点：\n ", Keypoints[1])
    # print("特征点个数： ", len(features))
    # print("特 ： ", type(features))
    # print("特 ： ", features)
    # print("特 ： ", features[1])

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
    # print("\n等待任意按键之后  ——> 关闭窗口并结束程序")

    cv.drawKeypoints(img1, Keypoints, img1)
    # cv.drawKeypoints(image,keypoints,outputimage,color,flags)
    # image: 也就是原始图片
    # keypoints：从原图中获得的关键点，这也是画图时所用到的数据
    # outputimage：输出 // 可以是原始图片
    # color：颜色设置，通过修改（b, g, r）的值, 更改画笔的颜色，b = 蓝色，g = 绿色，r = 红色。
    # flags：绘图功能的标识设置
    cv.namedWindow("SIFT", cv.WINDOW_NORMAL)
    cv.resizeWindow("SIFT", 800, 600)
    cv2.imshow("SIFT", img1)
    # cv2.waitKey(0)


def surf_test(img):
    print("\nSURF算法:")
    img1 = img.copy()
    img2 = img.copy()
    surf = cv2.xfeatures2d.SURF_create()
    Keypoints, features = surf.detectAndCompute(img1, None)  # kp表示输入的关键点，ft表示输出的sift特征向量，通常是128维的
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
    # print("\n等待任意按键之后  ——> 关闭窗口并结束程序")

    cv.drawKeypoints(img1, Keypoints, img1)

    # cv.drawKeypoints(image,keypoints,outputimage,color,flags)
    # image: 也就是原始图片
    # keypoints：从原图中获得的关键点，这也是画图时所用到的数据
    # outputimage：输出 // 可以是原始图片
    # color：颜色设置，通过修改（b, g, r）的值, 更改画笔的颜色，b = 蓝色，g = 绿色，r = 红色。
    # flags：绘图功能的标识设置
    cv.namedWindow("SURF", cv.WINDOW_NORMAL)
    cv.resizeWindow("SURF", 800, 600)
    cv2.imshow("SURF", img1)

    guolv_num = 90  # 过滤系数  舍弃掉低概率的%多少的特征点
    Keypoints2, features2 = Keypoints_guolv(Keypoints, features, guolv_num)  # 舍弃掉低概率的%多少
    print("过滤后:")
    print("特征点个数： ", len(Keypoints2))
    s_max, r_max = 0, 0
    s_min, r_min = -1, -1
    for k in Keypoints2:
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
    # print("\n等待任意按键之后  ——> 关闭窗口并结束程序")

    cv.drawKeypoints(img2, Keypoints2, img2)
    # cv.drawKeypoints(image,keypoints,outputimage,color,flags)
    # image: 也就是原始图片
    # keypoints：从原图中获得的关键点，这也是画图时所用到的数据
    # outputimage：输出 // 可以是原始图片
    # color：颜色设置，通过修改（b, g, r）的值, 更改画笔的颜色，b = 蓝色，g = 绿色，r = 红色。
    # flags：绘图功能的标识设置
    cv.namedWindow("SURF-guolv-" + str(guolv_num) + "%", cv.WINDOW_NORMAL)
    cv.resizeWindow("SURF-guolv-" + str(guolv_num) + "%", 800, 600)
    cv2.imshow("SURF-guolv-" + str(guolv_num) + "%", img2)
    # cv2.waitKey(0)


#   sift算法学习测试 debug输出
if 0:
    image_yt = cv2.imdecode(np.fromfile('DATA/img/模板匹配算法/测试图/yt1.jpg', dtype=np.uint8), -1)
    sift_test(image_yt)
    print("\n等待任意按键之后  ——> 关闭窗口并结束程序")
    cv2.waitKey(0)

#   surf算法学习测试 debug输出
if 0:
    image_yt = cv2.imdecode(np.fromfile('DATA/img/模板匹配算法/测试图/yt1.jpg', dtype=np.uint8), -1)
    surf_test(image_yt)
    sift_test(image_yt)
    print("\n等待任意按键之后  ——> 关闭窗口并结束程序")
    cv2.waitKey(0)

#   sift算法案例
if 0:
    def sift_test():
        # if __name__ == "__main__":
        image_yt = cv2.imdecode(np.fromfile('../DATA/img/模板匹配算法/测试图/yt1.jpg', dtype=np.uint8), -1)
        image_mb = cv2.imdecode(np.fromfile('../DATA/img/模板匹配算法/测试图/mb1.jpg', dtype=np.uint8), -1)
        # image_ytxz = cv2.imdecode(np.fromfile('DATA/img/模板匹配算法/测试图/yt1-xz2.jpg', dtype=np.uint8), -1)

        time11 = lml_time.get_time_ymd_hms_ms()
        img = sift_detect(image_yt, image_mb)
        time12 = lml_time.get_time_ymd_hms_ms()

        cv.namedWindow("match1", cv.WINDOW_NORMAL)
        cv.resizeWindow("match1", 800, 600)
        cv2.imshow("match1", img)

        image_xz = image_yt.copy()
        image_xz = lml_img_xuanzhuan.img_xuanzhuang(image_xz, 135)
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

#   surf算法案例
if 1:
    def surf_test():
        # if __name__ == "__main__":
        image_yt = cv2.imdecode(np.fromfile('../DATA/img/模板匹配算法/测试图/yt1.jpg', dtype=np.uint8), -1)
        image_mb = cv2.imdecode(np.fromfile('../DATA/img/模板匹配算法/测试图/mb1.jpg', dtype=np.uint8), -1)
        # image_ytxz = cv2.imdecode(np.fromfile('DATA/img/模板匹配算法/测试图/yt1-xz2.jpg', dtype=np.uint8), -1)

        # SIFT
        time01 = lml_time.get_time_ymd_hms_ms()
        img0 = sift_detect(image_yt, image_mb)
        time02 = lml_time.get_time_ymd_hms_ms()

        cv.namedWindow("SIFT", cv.WINDOW_NORMAL)
        cv.resizeWindow("SIFT", 800, 600)
        cv2.imshow("SIFT", img0)

        # SURF
        time11 = lml_time.get_time_ymd_hms_ms()
        img = surf_detect(image_yt, image_mb)
        time12 = lml_time.get_time_ymd_hms_ms()

        cv.namedWindow("SURF", cv.WINDOW_NORMAL)
        cv.resizeWindow("SURF", 800, 600)
        cv2.imshow("SURF", img)

        # SURF过滤
        guolv_num = 30
        time111 = lml_time.get_time_ymd_hms_ms()
        # guolv_num 保留guolv_num个的最高可能性特征点 其余舍弃
        imggl = surf_detect_guolv(image_yt, image_mb, guolv_num)
        time121 = lml_time.get_time_ymd_hms_ms()

        cv.namedWindow("SURF-guolv", cv.WINDOW_NORMAL)
        cv.resizeWindow("SURF-guolv", 800, 600)
        cv2.imshow("SURF-guolv", imggl)

        # 旋转图片
        image_xz = image_yt.copy()
        image_xz = lml_img_xuanzhuan.img_xuanzhuang(image_xz, 135)

        # SIFT
        time31 = lml_time.get_time_ymd_hms_ms()
        img3 = sift_detect(image_xz, image_mb)
        time32 = lml_time.get_time_ymd_hms_ms()

        cv.namedWindow("SIFT-xz", cv.WINDOW_NORMAL)
        cv.resizeWindow("SIFT-xz", 800, 600)
        cv2.imshow("SIFT-xz", img3)

        # SURF
        time21 = lml_time.get_time_ymd_hms_ms()
        img2 = surf_detect(image_xz, image_mb)
        time22 = lml_time.get_time_ymd_hms_ms()

        cv.namedWindow("SURF-xz", cv.WINDOW_NORMAL)
        cv.resizeWindow("SURF-xz", 800, 600)
        cv2.imshow("SURF-xz", img2)

        # SURF过滤
        guolv_num = 30
        time211 = lml_time.get_time_ymd_hms_ms()
        imggl2 = surf_detect_guolv(image_xz, image_mb, guolv_num)
        time221 = lml_time.get_time_ymd_hms_ms()

        cv.namedWindow("SURF-xz-guolv", cv.WINDOW_NORMAL)
        cv.resizeWindow("SURF-xz-guolv", 800, 600)
        cv2.imshow("SURF-xz-guolv", imggl2)

        print("\n原图匹配(sift)：")
        print(time01)
        print(time02)
        print('time = ', lml_time.get_time_yunsuan(time01, time02))

        print("\n原图匹配(surf)：")
        print(time11)
        print(time12)
        print('time = ', lml_time.get_time_yunsuan(time11, time12))

        print("\n原图匹配(surf)-过滤：")
        print(time111)
        print(time121)
        print('time = ', lml_time.get_time_yunsuan(time111, time121))

        print("\n旋转图匹配(sift)：")
        print(time31)
        print(time32)
        print('time = ', lml_time.get_time_yunsuan(time31, time32))

        print("\n旋转图匹配(surf)：")
        print(time21)
        print(time22)
        print('time = ', lml_time.get_time_yunsuan(time21, time22))

        print("\n旋转图匹配(surf)-过滤：")
        print(time211)
        print(time221)
        print('time = ', lml_time.get_time_yunsuan(time211, time221))

        print("\n等待任意按键 关闭窗口结束程序")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # exit(0)
        # cv2.destroyWindow(all)
        # plt.imshow(img)
        # plt.show()


    surf_test()
