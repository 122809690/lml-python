#! /usr/bin/env python
# coding=utf-8

import math

# import DATA.install.JXB.auboi5SDK.robotcontrol as robotcontrol
from JXB.pyd import lml_JXB_lib as robotcontrol


def robot_login():
    ret = robotcontrol.Auboi5Robot().initialize()
    print("Auboi5Robot().initialize() is {0}".format(ret))

    # 实例化一个控制类对象
    robot = robotcontrol.Auboi5Robot()

    # 创建一个句柄
    handle = robot.create_context()

    # 登录机器人
    ip = "192.168.135.129"
    port = 8899  # 这个端口号好像可以随意设置   因为不是tcp通信而是调用厂家sdk写死的通信
    result = robot.connect(ip, port)

    if result == 0:
        # 机械臂上电,碰撞等级6·工具动力学参数(e,e,e) , 0kg
        collision = 6
        tool_dynamics = {"position": (0, 0, 0), "payload": 0.0, "inertia": (0, 0, 0, 0, 0, 0)}
        ret = robot.robot_startup(collision, tool_dynamics)
        print("robot_startup ret is {0}".format(ret))

        # 关节运动
        # 初始化全局运动属性
        robot.init_profile()
        # 设置关节最大加速度
        robot.set_joint_maxacc((2.0, 2.0, 2.0, 2.0, 2.0, 2.0))
        # 设置关节最大速度
        robot.set_joint_maxvelc((0.5, 0.5, 0.5, 0.5, 0.5, 0.5))

        # 设置目标路点1
        joint1 = (math.radians(-17.682977), math.radians(27.785112), math.radians(-138.61),
                  math.radians(-76.400734), math.radians(-90), math.radians(-107.682980))
        # 关节运动至目标路点1
        ret = robot.move_joint(joint1)
        # 设置目标路点2
        joint2 = (math.radians(-0.000172), math.radians(-7.291862), math.radians(-75.694718),
                  math.radians(21.596727), math.radians(-89.999982), math.radians(-0.000458))
        ret = robot.move_joint(joint2)

        print("robot move_joint ret is {0}".format(ret))

    else:
        print("login failed!")


def robot_nys():
    ret = robotcontrol.Auboi5Robot().initialize()
    print("Auboi5Robot().initialize() is {0}".format(ret))

    # 实例化一个控制类对象
    robot = robotcontrol.Auboi5Robot()

    # 创建一个句柄
    handle = robot.create_context()

    # 登录机器人
    # ip = "192.168.0.18"
    ip = "192.168.135.129"
    # ip = "192.168.135.1"
    # port = 8899  # sdk控制默认端口号
    port = 8899
    print(ip)

    # result = robot.connect(ip, port)
    result = robot.connect(ip, port)

    print("result ==", result)

    if result == 0:
        # 机械臂上电,碰撞等级6·工具动力学参数(e,e,e) , 0kg
        collision = 6
        tool_dynamics = {"position": (0, 0, 0), "payload": 0.0, "inertia": (0, 0, 0, 0, 0, 0)}
        ret = robot.robot_startup(collision, tool_dynamics)
        print("robot_startup ret is {0}".format(ret))

        # 关节运动
        # 初始化全局运动属性
        robot.init_profile()
        # 设置关节最大加速度
        robot.set_joint_maxacc((2.0, 2.0, 2.0, 2.0, 2.0, 2.0))
        # 设置关节最大速度
        robot.set_joint_maxvelc((0.5, 0.5, 0.5, 0.5, 0.5, 0.5))

        pos = (-0.400319, -0.121499, 0.547598)  # xyz
        rpy_xyz = (179.999588, -0.000081, -89.999641)  # rx ry rz
        # rpy = {math.radians(89.999962), math.radians(0.0), math.radians(0.0)}

        ret = robot.move_to_target_in_cartesian(pos, rpy_xyz)

        # print(ret)

        print("robot move_joint ret is {0}".format(ret))
        return 't'
        exit(1)

    else:
        print("login failed!")
        return 'f'
        return 'f'


def robot_run(ip, flag, f1, f2, f3, f4, f5, f6):
    # 192.168.135.129,1,-0.400319,-0.121499,0.547598,179.999588,-0.000081,-89.999641
    # print(ip)
    # print(flag)
    # print(f1)
    # print(f2)
    # print(f3)
    # print(f4)
    # print(f5)
    # print(f6)
    ret = robotcontrol.Auboi5Robot().initialize()
    # print("Auboi5Robot().initialize() is {0}".format(ret))

    # 实例化一个控制类对象
    robot = robotcontrol.Auboi5Robot()

    # 创建一个句柄
    handle = robot.create_context()

    # 登录机器人
    # ip = "192.168.0.18"
    # ip = "192.168.135.129"
    # ip = "192.168.135.1"
    # port = 8899  # sdk控制默认端口号
    port = 8899
    # print(ip)

    # result = robot.connect(ip, port)
    result = robot.connect(ip, port)

    # print("result ==", result)

    if result == 0:
        # 机械臂上电,碰撞等级6·工具动力学参数(e,e,e) , 0kg
        collision = 6
        tool_dynamics = {"position": (0, 0, 0), "payload": 0.0, "inertia": (0, 0, 0, 0, 0, 0)}
        ret = robot.robot_startup(collision, tool_dynamics)
        # print("robot_startup ret is {0}".format(ret))

        # 关节运动
        # 初始化全局运动属性
        robot.init_profile()
        # 设置关节最大加速度
        robot.set_joint_maxacc((2.0, 2.0, 2.0, 2.0, 2.0, 2.0))
        # 设置关节最大速度
        robot.set_joint_maxvelc((0.5, 0.5, 0.5, 0.5, 0.5, 0.5))

        if flag == 1:
            # 192.168.135.129,1,-0.400319,-0.121499,0.547598,179.999588,-0.000081,-89.999641
            # pos = (-0.400319, -0.121499, 0.547598)  # xyz
            # rpy_xyz = (179.999588, -0.000081, -89.999641)   # rx ry rz
            # # rpy = {math.radians(89.999962), math.radians(0.0), math.radians(0.0)}
            # ret = robot.move_to_target_in_cartesian(pos, rpy_xyz)
            pos = (f1, f2, f3)  # xyz
            rpy_xyz = (f4, f5, f6)  # rx ry rz
            # rpy = {math.radians(89.999962), math.radians(0.0), math.radians(0.0)}
            try:
                ret = robot.move_to_target_in_cartesian(pos, rpy_xyz)
            except:
                return 'f'
                # exit(1)

        if flag == 2:
            # 192.168.135.129,2,0,0,0,0,0,0
            joint1 = (math.radians(f1), math.radians(f2), math.radians(f3),
                      math.radians(f4), math.radians(f5), math.radians(f6))
            # 关节运动至目标路点1
            try:
                ret = robot.move_joint(joint1)
            except:
                return 'f'
                # exit(1)
        # print(ret)
        # print("robot move_joint ret is {0}".format(ret))

        return 't'
        # exit(1)
    else:
        # print("login failed!")
        return 'f'
        # exit(1)

if __name__ == '__main__':
    # robot_login()
    robot_nys()
