# encoding:utf-8

# 求两点的距离 输入两个点的坐标

# 当flag = 0时返回弧度，当flag！-0时返回角度


def point_distance(A1, A2, flag=1):
    #   求直线的斜率  再反三角函数求角度
    # print(A1)
    d = ((A2[1] - A1[1]) ** 2 + (A2[0] - A1[0]) ** 2) ** 0.5
    # print(d)
    # 返回距离
    return d


if __name__ == "__main__":
    # print("1")
    LA1 = (0.0, 0.0)
    LA2 = (1.0, 1.0)
    LB1 = (0.0, 0.0)
    LB2 = (0.0, 1.0)
    # a = lines_orientation1(LA1, LA2, LB1, LB2, 2)
    # print(a)

    # cv2.waitKey(0)
    # sys.system("pause")
    # return 0
