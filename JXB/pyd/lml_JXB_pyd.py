#! /usr/bin/env python
# coding=utf-8

import math

# import DATA.install.JXB.auboi5SDK.robotcontrol as robotcontrol
import lml_JXB_lib as robotcontrol


def robot_run(ip, flag, f1=0, f2=0, f3=0, f4=0, f5=0, f6=0):
    # 192.168.135.129,1,-0.400319,-0.121499,0.547598,179.999588,-0.000081,-89.999641
    # 192.168.135.129,2,0,0,0,0,0,0
    # 192.168.135.129,0,0,0,0,0,0,0
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

        # # 设置关节最大加速度
        # robot.set_joint_maxacc((19.0, 19.0, 19.0, 19.0, 19.0, 19.0))
        # # 设置关节最大速度
        # robot.set_joint_maxvelc((140, 140, 140, 140, 140, 140))
        # # 设置机械臂末端最大线加速度
        # robot.set_end_max_line_acc(end_maxacc=0.1)
        # # 设置机械臂末端最大线速度
        # robot.set_end_max_line_velc(end_maxvelc=0.1)
        # # 设置机械臂末端最大角加速度
        # robot.set_end_max_angle_acc(end_maxacc=0.1)
        # # 设置机械臂末端最大角速度
        # robot.set_end_max_angle_velc(end_maxvelc=0.1)

        if flag == 1:
            # 169.254.92.39,1,-0.400319,-0.121499,0.547598,179.999588,-0.000081,-89.999641
            # 169.254.92.39,1,-0.4,-0.1,0.5,180,-0,-90
            # pos = (-0.400319, -0.121499, 0.547598)  # xyz
            # rpy_xyz = (179.999588, -0.000081, -89.999641)   # rx ry rz
            # # rpy = {math.radians(89.999962), math.radians(0.0), math.radians(0.0)}
            # ret = robot.move_to_target_in_cartesian(pos, rpy_xyz)
            pos = (f1, f2, f3)  # xyz
            rpy_xyz = (f4, f5, f6)  # rx ry rz
            # rpy = {math.radians(89.999962), math.radians(0.0), math.radians(0.0)}
            try:
                ret = robot.move_to_target_in_cartesian(pos, rpy_xyz)
                return 't'
            except:
                return 'f'
                # exit(1)

        if flag == 2:
            # 169.254.92.39,2,0,0,0,0,0,0
            joint1 = (math.radians(f1), math.radians(f2), math.radians(f3),
                      math.radians(f4), math.radians(f5), math.radians(f6))
            # 关节运动至目标路点1
            try:
                # print('000')
                robot.move_joint(joint1)
                # print(ret)
                # print('111')
                return 't'
            except:
                return 'f'
                # exit(1)
        # print(ret)
        # print("robot move_joint ret is {0}".format(ret))

        if flag == 3:
            # 169.254.92.39,3,2,4,2,2,2,2
            hc = f1
            qs = f2
            jjsd = f3
            jsd = f4
            xjsd = f5
            xsd = f6

            # # 设置机械臂末端最大线加速度
            # robot.set_end_max_line_acc(end_maxacc=0.1)
            # # 设置机械臂末端最大线速度
            # robot.set_end_max_line_velc(end_maxvelc=0.1)
            # 设置关节最大速度
            # robot.set_joint_maxvelc((11.0, 11.0, 11.0, 11.0, 11.0, 11.0))

            # 设置机械臂末端最大角加速度
            robot.set_end_max_angle_acc(end_maxacc=0.1 * jjsd)
            # 设置机械臂末端最大角速度
            robot.set_end_max_angle_velc(end_maxvelc=0.1 * jsd)
            # 设置机械臂末端最大线加速度
            robot.set_end_max_line_acc(end_maxacc=0.1 * xjsd)
            # 设置机械臂末端最大线速度
            robot.set_end_max_line_velc(end_maxvelc=0.1 * xsd)

            n = robot.get_current_waypoint()
            # print("n:\n", n)
            # for key in n:
            #     for i in range(len(n[key])):
            #         n[key][i] = round((n[key][i]), 6)
            # print('-1')
            #   {'joint': [0.46097755432128906, -0.3541720509529114, -1.4865835905075073, 0.49275752902030945, -1.9377663135528564, 0.7270048260688782],
            #   'ori': [0.06093651796916516, -0.6006129829759546, 0.7779998122087032, 0.17397435920773474],
            #   'pos': [-0.22652917691899493, -0.21051319948128083, 0.5475984528527112]}
            joint_n = n['joint']
            # joint_n = [round((joint_n[0]), 6), round((joint_n[1]), 6), round((joint_n[2]), 6), round((joint_n[3]), 6), round((joint_n[4]), 6), round((joint_n[5]), 6)]
            ori_n = n['ori']
            # ori_n = [round((ori_n[0]), 6), round((ori_n[1]), 6), round((ori_n[2]), 6)]
            pos_n = n['pos']
            # pos_n = [round((pos_n[0]), 6), round((pos_n[1]), 6), round((pos_n[2]), 6)]

            pos_1 = [round((pos_n[0] + 0.01 * hc), 6), pos_n[1], pos_n[2]]
            # pos_1 = [0.0, 0.0, 0.0]
            pos_2 = [pos_n[0], round((pos_n[1] + 0.01 * hc), 6), pos_n[2]]
            # print(pos_n)
            # print(pos_1)
            # print(pos_2)
            # print(ori_n)
            # print(joint_n)
            # print('111')
            try:
                joint_1 = joint_n
                # point_1 = robot.ZB_to_joint(pos_1, ori_n)
                point_1 = robot.inverse_kin(joint_radian=joint_1, pos=pos_1, ori=ori_n)
                # print("point_1:\n", point_1)
                # print("joint_n:\n", joint_n)
                # robot.inverse_kin(,)
                # exit(0)
                # print('1121')
                # point_2 = robot.ZB_to_joint(pos_2, ori_n)
                point_2 = robot.inverse_kin(joint_radian=joint_1, pos=pos_2, ori=ori_n)
                # print('1122')
                # print(point_1)
                # for key in point_1:
                #     for i in range(len(point_1[key])):
                #         point_1[key][i] = round((point_1[key][i]), 6)
                # for key in point_2:
                #     for i in range(len(point_2[key])):
                #         point_2[key][i] = round((point_2[key][i]), 6)
                # print('000')
                # print(point_1['joint'])

                robot.add_waypoint(joint_n)
                # print(point_1)
                # print(point_1['joint'])
                # print('333')

                robot.add_waypoint(point_1['joint'])
                robot.add_waypoint(point_2['joint'])
                # robot.add_waypoint([round((point_1['joint'][0]), 6), round((point_1['joint'][1]), 6), round((point_1['joint'][2]), 6), round((point_1['joint'][3]), 6), round((point_1['joint'][4]), 6), round((point_1['joint'][5]), 6)])
                # robot.add_waypoint([round((point_2['joint'][0]), 6), round((point_2['joint'][1]), 6), round((point_2['joint'][2]), 6), round((point_2['joint'][3]), 6), round((point_2['joint'][4]), 6), round((point_2['joint'][5]), 6)])
                # print('111')

                robot.set_circular_loop_times(circular_count=int(qs))
                # print('222')

                robot.move_track(2)
                # print('444')

                robot.remove_all_waypoint()
                # print("rt")

                # 设置机械臂末端最大角加速度
                robot.set_end_max_angle_acc(end_maxacc=0.1)
                # 设置机械臂末端最大角速度
                robot.set_end_max_angle_velc(end_maxvelc=0.1)
                # 设置机械臂末端最大线加速度
                robot.set_end_max_line_acc(end_maxacc=0.1)
                # 设置机械臂末端最大线速度
                robot.set_end_max_line_velc(end_maxvelc=0.1)

                return 't'

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
        #exit(1)


if __name__ == '__main__':
    robot_run("192.168.135.129",2,0,0,0,0,0,0)
    # robot_login()
    # exit(1)
