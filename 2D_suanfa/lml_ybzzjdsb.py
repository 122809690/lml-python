# encoding:utf-8

# 求图中最长直线的角度

import lml_img_to_jiaodu
import lml_imread

if __name__ == '__main__':
    # path = '..DATA/img/仪表识别/yuanbiao.png'

    # img = lml_imread.imread('../DATA/img/轮廓查找/yt.jpg')
    img = lml_imread.imread('../DATA/img/仪表识别/yuanbiao.png')
    # img = lml_imread.imread(path)

    lines = lml_img_to_jiaodu.line_detect_possible_demo(img)
    line = lml_img_to_jiaodu.lines_choose(lines, img)
    print(line)
