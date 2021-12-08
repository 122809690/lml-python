#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
机械臂控制程序，tcp通信协议

tcp监听ip：192.168.0.214   端口：12345

接收数据包
逗号间隔开的8个str
ip,flag,f1,f2,f3,f4,f5,f6
例如
192.168.135.129,1,-0.400319,-0.121499,0.547598,179.999588,-0.000081,-89.999641
192.168.135.129,2,0,0,0,0,0,0

返回值
tcp传输成功时，实时反馈           [time]"传入的数据"
参数初验成功，移动机械臂           [machine move] start
移动完成                        [machine move] end
移动失败                        [machine move] false
'''

import time
from socket import *
from time import ctime

from JXB import lml_JXB_test

# host = ''
host = '192.168.0.214'
port = 12345
buffsize = 2048
ADDR = (host, port)

tctime = socket(AF_INET, SOCK_STREAM)
tctime.bind(ADDR)
tctime.listen(3)

data = ''
while data != 'q':
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
        if len(data_t) == 8:
            tctimeClient.send("[machine move] start".encode())
            rmsg = lml_JXB_test.robot_run(data_t[0], int(data_t[1]), float(data_t[2]), float(data_t[3]),
                                          float(data_t[4]), float(data_t[5]), float(data_t[6]), float(data_t[7]))
            # print("======================")
            print(rmsg)
            if rmsg == 't':
                tctimeClient.send("[machine move] end".encode())
            if rmsg == 'f':
                tctimeClient.send("[machine move] false".encode())
            time.sleep(1)

    tctimeClient.close()
