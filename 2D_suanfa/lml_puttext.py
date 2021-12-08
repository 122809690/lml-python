#   opencv 图片上加入中文汉字

import numpy as np
# import PILasOPENCV as Image
# import PILasOPENCV as ImageDraw
# import PILasOPENCV as ImageFont
from PIL import ImageFont, ImageDraw, Image


def lml_puttext(img, text, post, zihao=15, yanse=(0, 0, 0, 0)):
    fontpath = "DATA/Fonts/simsun.ttc"  # <== 这里是宋体路径
    font = ImageFont.truetype(fontpath, zihao)  # <== 字号
    img_pil = Image.fromarray(img)  # opencv数据图转PLT数据图
    img_pil = img_pil.convert('RGB')
    draw = ImageDraw.Draw(img_pil)  # 创建绘画对象
    draw.text(post, text, font=font, fill=yanse)
    #   坐标  汉字  字体  颜色
    img_zh = np.array(img_pil)
    return img_zh
