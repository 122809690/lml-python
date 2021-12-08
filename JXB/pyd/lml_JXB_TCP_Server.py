#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
机械臂控制程序，tcp通信协议

tcp监听ip：192.168.0.214   端口：12345

接收数据包
逗号间隔开的8个str
ip,flag,f1,f2,f3,f4,f5,f6

例如
模式1 将机械臂移动到 f1,f2,f3位置 末端角度f4,f5,f6
192.168.135.129,1,-0.4,-0.1,0.5,180,-0,-90
模式2 将机械臂六个关节扭转为f1,f2,f3,f4,f5,f6度
192.168.135.129,2,0,0,0,0,0,0
模式3 将机械臂末端做水平圆周运动 (摇晃瓶子) 圆周半径f1(cm)，圆周运动圈数f2，圆周长角加速度f3(*0.1)，角速度f4，线加速度f5，线速度f6
192.168.135.129, 3, 2, 4, 2, 2, 5, 5

返回值
tcp传输成功时，实时反馈           [time]"传入的数据"
参数初验成功，移动机械臂           [machine move] start
移动完成                        [machine move] end
移动失败                        [machine move] false
'''

import time
from socket import *
from time import ctime

import lml_JXB_pyd


# import socket


def lml_JXB_TCP_server():
    # 获取计算机名称
    hostname = gethostname()
    # 获取本机IP
    host_ip = gethostbyname(hostname)
    # print(ip)
    # hostip = '192.168.0.214'
    # hostip = ''
    port = 12345
    buffsize = 2048
    ADDR = (host_ip, port)

    tctime = socket(AF_INET, SOCK_STREAM)
    tctime.bind(ADDR)
    tctime.listen(3)

    data = ''
    while data != 'q':
        print(host_ip, " : ", port)
        print('Wait connection ...')
        tctimeClient, addr = tctime.accept()
        print("Connection from :", addr)

        while True:
            data = tctimeClient.recv(buffsize).decode()
            if not data:
                break
            # if data == 'q':
            #     break
            # print(type(data))
            tctimeClient.send(('[%s]\"%s\"  ' % (ctime(), data)).encode())  # str->bytes
            print(data)
            # import struct
            # s = struct.pack("3f",2.23,323.32,32.32323)
            # d = struct.unpack("3f",s)

            data_t = tuple(data.split(","))

            tctimeClient.send("[machine move] start".encode())
            rmsg = lml_JXB_pyd.robot_run(data_t[0], int(data_t[1]), float(data_t[2]), float(data_t[3]),
                                         float(data_t[4]), float(data_t[5]), float(data_t[6]), float(data_t[7]))
            # print("======================")
            print(rmsg)
            if rmsg == 't':
                tctimeClient.send("[machine move] end".encode())
            if rmsg == 'f':
                tctimeClient.send("[machine move] false".encode())
            time.sleep(1)

        tctimeClient.close()


if __name__ == '__main__':
    lml_JXB_TCP_server()
