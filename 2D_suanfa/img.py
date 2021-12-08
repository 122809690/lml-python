import os

import cv2

inputpath = 'C:\\ylc_data\\6-d4\\'
outputpath = 'C:\\ylc_outimg\\'
converpath = 'C:\\ylc_data\\Annotations\\'
infopath = 'C:\\ylc_data\\label_info\\'
filenames = os.listdir(inputpath)
converfiles = os.listdir(converpath)


def cut_left_right():
    for file_name in filenames:
        img = cv2.imread(inputpath + file_name, cv2.COLOR_BGR2RGB)  # ,cv2.IMREAD_GRAYSCALE)
        img1 = img[0:1024, 0:1024]
        # img2 = img[0:1024, 1024:2048]
        cv2.imwrite(outputpath + "L" + file_name, img1)
        # cv2.imwrite(outputpath + "R" + file_name, img2)
