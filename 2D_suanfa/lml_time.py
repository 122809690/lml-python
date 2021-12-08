#!/usr/bin/python
# -*- coding: UTF-8 -*-

import datetime


# t = time.time()
#
#
# localtime = time.asctime(time.localtime(time.time()) )
# print("本地时间为 :", localtime)
#
# localtime = time.localtime(time.time())
# print("本地时间为 :", localtime)
#
# # print(localtime.tm_year)
# # print(localtime.tm_mon)
# # print(localtime.tm_mday)
# # print(localtime.tm_hour)
# # print(localtime.tm_min)
# # print(localtime.tm_sec)
# # print(localtime.tm_wday)
# # print(localtime.tm_yday)
#
# dt    = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') # 含微秒的日期时间
# print(dt)
# print(dt_ms)

def get_time_ymd_hms_ms():
    dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %f')
    return dt_ms


def get_time_ymd():
    dt_ymd = datetime.datetime.now().strftime('%Y-%m-%d')
    return dt_ymd


def get_time_yunsuan(time1, time2):
    t1 = datetime.datetime.strptime(time1, '%Y-%m-%d %H:%M:%S %f')
    t2 = datetime.datetime.strptime(time2, '%Y-%m-%d %H:%M:%S %f')
    if t1 > t2:
        return t1 - t2
    else:
        return t2 - t1
