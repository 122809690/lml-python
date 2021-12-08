# encoding:utf-8

# 求两直线的夹角  根据每直线上两点  共计四个点计算

# 当flag = 0时返回弧度，当flag！-0时返回角度
import numpy as np


def lines_orientation1(A1, A2, B1, B2, flag=1):
    #   求直线的斜率  再反三角函数求角度
    angle1 = np.rad2deg(np.arctan2(A2[1] - A1[1], A2[0] - A1[0]))
    angle2 = np.rad2deg(np.arctan2(B2[1] - B1[1], B2[0] - B1[0]))
    # print(angle1)
    # print(angle2)
    # print(angle2 - angle1)
    # print(np.pi)

    # exit(0)
    if flag == 1:
        # 返回角度
        # print("111111")
        return angle2 - angle1
    else:
        # 返回弧度
        return (angle2 - angle1) * np.pi / 180.0


def onelines(A1, A2, flag=1):
    #   求直线的斜率  再反三角函数求角度
    angle1 = np.rad2deg(np.arctan2(A2[1] - A1[1], A2[0] - A1[0]))
    # angle2 = np.rad2deg(np.arctan2(B2[1] - B1[1], B2[0] - B1[0]))
    # print(angle1)
    # print(angle2)
    # print(angle2 - angle1)
    # print(np.pi)

    # exit(0)
    if flag == 1:
        # 返回角度
        # print("111111")
        return angle1
    else:
        # 返回弧度
        return angle1 * np.pi / 180.0


if __name__ == "__main__":
    # print("1")
    LA1 = (0.0, 0.0)
    LA2 = (1.0, 1.0)
    LB1 = (0.0, 0.0)
    LB2 = (0.0, 1.0)
    a = lines_orientation1(LA1, LA2, LB1, LB2, 2)
    print(a)

    # cv2.waitKey(0)
    # sys.system("pause")
    # return 0
