# 二维码识别

import cv2
import cv2 as cv
import numpy as np
import qrcode as qrcode
from pyzbar import pyzbar
from pyzbar.pyzbar import decode

import lml_img_xuanzhuan
import lml_imread
import lml_puttext
import lml_time


def tupian_bianhuan(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = img

    # cv.imshow("1", img)
    img = img / 255.0  # 归一化转换  灰度值转换为0~1区间
    # img = img / 1.0
    # cv.imshow("2", img)

    # cv.imshow("0", frame_mono)
    # print(frame_mono.dtype)
    # frame_mono = frame_mono.astype('float64')
    # frame_mono = frame_mono.astype('float64')

    # cv.imshow("1", frame_mono)
    # print(frame_mono.shape)
    # cv.waitKey(0)

    # print(len(frame_mono[0])/2)
    # print(len(frame_mono)/2)
    # print(frame_mono[1])
    img[:int(len(img) / 2 - 60), :int(len(img[0]) / 2)] *= 0.5  # 蒙版变更 局部亮度改变
    img[int(len(img) / 2 - 60):, int(len(img[0]) / 2):] *= 2

    img = cv2.blur(img, (3, 3))  # 均值处理 即模糊

    # cv.imshow("3", img)
    img = lml_img_xuanzhuan.img_xuanzhuang(img, 125)
    # cv.imshow("4", img)

    # print(len(img.shape))
    # print(img.shape)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    # cv.imshow("5", img)

    return img


def shibie_zbar(img):
    # cv.imshow("txm", img)
    # cv.waitKey(0)
    # print(len(img.shape))
    # print(img.shape)
    #
    # if len(img.shape) > 2:
    #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # else:
    #     img_gray = img

    # frame_mono = img_gray / 255.0    # 格式转换  灰度值转换为0~1区间

    # cv.imshow("0", frame_mono)
    # print(frame_mono.dtype)
    # frame_mono = frame_mono.astype('float64')
    # frame_mono = frame_mono.astype('float64')

    # cv.imshow("1", frame_mono)
    # print(frame_mono.shape)
    # cv.waitKey(0)

    # print(len(frame_mono[0])/2)
    # print(len(frame_mono)/2)
    # print(frame_mono[1])
    # frame_mono[:int(len(frame_mono)/2-100), :int(len(frame_mono[0])/2)] *= 0.5
    # frame_mono[int(len(frame_mono)/2-100):, int(len(frame_mono[0])/2):] *= 2

    # cv.imshow("2", frame_mono)
    # cv.waitKey(0)

    # start = time.time_ns()
    cv.imshow("yt", img)
    # cv.waitKey(0)
    img = tupian_bianhuan(img)
    cv.imshow("bianhuan", img)
    # cv.imshow("bh", img)

    barcodes = decode(img)
    # barcodes = decode(img, symbols=[ZBarSymbol.QRCODE])    # 确定是QRCODE二维码可以加速
    # print("========")
    if len(barcodes) > 0:
        # end = time.time_ns()
        # print("running time is ", str(end - start))
        print("FIND")
        # qrResult = barcodes[0].data.decode('utf-8')
        # print("QR Code  is: {}".format(qrResult))
        for barcode in barcodes:
            # 提取二维码的边界框的位置
            # 画出图像中条形码的边界框
            (x, y, w, h) = barcode.rect
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # 提取二维码数据为字节对象，所以如果我们想在输出图像上
            # 画出来，就需要先将它转换成字符串
            barcodeData = barcode.data.decode("UTF8")
            barcodeType = barcode.type

            # 绘出图像上条形码的数据和条形码类型
            text = "{} ({})".format(barcodeData, barcodeType)
            # cv.putText(qrcode_img, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 125), 2)
            # print(type(img))
            # print(type(img2))
            # print(img.shape)
            # print(img2.shape)

            img = lml_puttext.lml_puttext(img, text, (x - 10, y - 30), 50, (0, 0, 255, 0))
            # 向终端打印条形码数据和条形码类型
            print(" Found [{}] : {}".format(barcodeType, barcodeData))

            # exit()
    else:
        print("NOT FIND")

    return img


def qr_ewm_sb(qrcode_img):
    barcodes = pyzbar.decode(qrcode_img)

    print("[INFO] Found {} barcode: {}", barcodes)
    for barcode in barcodes:
        # 提取二维码的边界框的位置
        # 画出图像中条形码的边界框
        (x, y, w, h) = barcode.rect
        cv.rectangle(qrcode_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 提取二维码数据为字节对象，所以如果我们想在输出图像上
        # 画出来，就需要先将它转换成字符串
        barcodeData = barcode.data.decode("UTF8")
        barcodeType = barcode.type

        # 绘出图像上条形码的数据和条形码类型
        text = "{} ({})".format(barcodeData, barcodeType)
        # cv.putText(qrcode_img, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 125), 2)
        qrcode_img = lml_puttext.lml_puttext(qrcode_img, text, (x - 10, y - 30), 30, (0, 0, 255, 0))
        # 向终端打印条形码数据和条形码类型
        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
        # cv.imshow("shibie",qrcode_img)
    return qrcode_img


# 生成二维码
def qr_ewm_make():
    # found = set()
    # capture = cv2.VideoCapture(0)
    # PATH = "test.csv"
    # while (1):
    #     ret, frame = capture.read()
    #     test = pyzbar.decode(frame)
    #     for tests in test:
    #         testdate = tests.data.decode('utf-8')
    #         print(testdate)
    #         if testdate not in found:
    #             with open(PATH, 'a+') as f:
    #                 csv_write = csv.writer(f)
    #                 date = [testdate]
    #                 csv_write.writerow(date)
    #             found.add(testdate)
    #     cv2.imshow('Test', frame)
    #     if cv2.waitKey(1) == ord('q'):
    #         break

    qr = qrcode.QRCode(
        version=1,  # 二维码的格子矩阵大小
        error_correction=qrcode.constants.ERROR_CORRECT_Q,
        box_size=10,
        border=4,
    )
    qr.add_data("Hello World 二维码测试")  # 向二维码添加数据
    qr.make(fit=True)
    img = qr.make_image(fill_color="green", back_color="white")  # 更改QR的背景和绘画颜色
    # print(type(img))
    # img.show()  # 显示二维码
    # print("img.size.width",img.size.width)

    # img_oc = np.zeros((img.width, img.height), np.uint8)  # 创建黑色背景布
    # cv.imshow("ewm_bj", img_oc)
    img_oc = np.array(img).astype(np.uint8)

    # cv.imshow("ewm", img_oc)  # 显示二维码
    # print("111111111111111111111111111")
    return img_oc


def qr_ewm_test():
    # # 生成二维码
    # tsc1 = lml_time.get_time_ymd_hms_ms()
    # ewm = qr_ewm_make()
    # tsc2 = lml_time.get_time_ymd_hms_ms()
    # tsc = lml_time.get_time_yunsuan(tsc1, tsc2)
    # print("tsc=", tsc)
    # cv.imshow("ewm", ewm)
    # # print("============")

    ewm = lml_imread.imread("../DATA/img/二维码识别/ewm.jpg")
    # ewm2 = ewm.copy()

    # ewm = tupian_bianhuan(ewm)
    # print(ewm.dtype)
    # 旋转二维码

    # ewm = lml_img_xuanzhuan.img_xuanzhuang(ewm, 125)
    # ewm = cv2.blur(ewm, (5, 5))  # 均值处理 即模糊

    # cv.imshow("ewm", ewm)
    # cv.waitKey()
    # 识别
    tsb1 = lml_time.get_time_ymd_hms_ms()
    # shibie = qr_ewm_sb(mk)
    shibie = shibie_zbar(ewm)
    tsb2 = lml_time.get_time_ymd_hms_ms()
    tsb = lml_time.get_time_yunsuan(tsb1, tsb2)
    print("tsb=", tsb)

    cv.imshow("ewm_shibie", shibie)
    cv.waitKey(0)
    exit(0)


qr_ewm_test()


def qr_txm_test():
    txm = lml_imread.imread("../DATA/img/二维码识别/txm.jpg")
    # 旋转二维码
    # txm = lml_img_xuanzhuan.img_xuanzhuang(txm, 125)
    # txm = cv2.blur(txm, (5, 5))  # 均值处理 即模糊
    # cv.imshow("txm", txm)
    # cv.waitKey(0)
    # exit()

    # 识别
    tsb1 = lml_time.get_time_ymd_hms_ms()
    # shibie = qr_ewm_sb(mk)
    shibie = shibie_zbar(txm)
    tsb2 = lml_time.get_time_ymd_hms_ms()
    tsb = lml_time.get_time_yunsuan(tsb1, tsb2)
    print("time_sb=", tsb)

    cv.imshow("ewm_shibie", shibie)
    cv.waitKey(0)
    exit(0)

# qr_txm_test()

#
# def ewm_shibie():
#     return 1
