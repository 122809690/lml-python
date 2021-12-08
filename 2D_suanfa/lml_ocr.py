#   OCR字符识别

# Tesseract vs EasyOCR
# Tesseract在字母识别方面表现更好，而EasyOCR在数字方面表现更好。
# 如果图片包含大量字母，可以考虑Tesseract。
# 此外，EasyOCR 的输出是小写的。如果大写对处理很重要，还应该使用Tesseract。
# 另一方面，如果图片中包含大量数字，建议EasyOCR。
# 在速度方面，Tesseract在CPU上的表现优于EasyOCR，而EasyOCR在GPU上的表现更好。

# 第三种 PaddlePaddle/PaddleOCR

###########忽略警告
import warnings

warnings.filterwarnings("ignore")

# 导入easyocr
import easyocr
import cv2
import lml_imread
import lml_time
import numpy as np
# import PILasOPENCV as Image
# import PILasOPENCV as ImageDraw
# import PILasOPENCV as ImageFont
from PIL import ImageFont, ImageDraw, Image
import pytesseract
from paddleocr import PaddleOCR

from aip import AipOcr


def draw_text_easyocr(img, result):
    fontpath = "DATA/Fonts/simsun.ttc"  # <== 这里是宋体路径
    font = ImageFont.truetype(fontpath, 15)  # <== 字号
    img_pil = Image.fromarray(img)  # opencv数据图转PLT数据图
    draw = ImageDraw.Draw(img_pil)  # 创建绘画对象
    for i in result:
        # for k in i:
        #     print(k)
        # print(i[1],i[0][0])
        # print(type(i[0][0]))
        # exit(0)
        # cv2.putText(img2, i[1], (i[0][0][0]+50, i[0][0][1]+50), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2, 8)
        draw.text((i[0][0][0] + 10, i[0][0][1] + 10), i[1], font=font, fill=(0, 0, 0, 0))
        #   坐标  汉字  字体  颜色
    # fontpath = "DATA/Fonts/simsun.ttc" # <== 这里是宋体路径
    # font = ImageFont.truetype(fontpath, 32)
    # img_pil = Image.fromarray(img2)
    # draw = ImageDraw.Draw(img_pil)
    # draw.text((50, 80),  "端午节就要到1。。。", font = font, fill = (0,0,0,0))
    img_zh = np.array(img_pil)
    return img_zh


def draw_text_tesseract(img, result):
    fontpath = "DATA/Fonts/simsun.ttc"  # <== 这里是宋体路径
    font = ImageFont.truetype(fontpath, 15)  # <== 字号
    img_pil = Image.fromarray(img)  # opencv数据图转PLT数据图
    draw = ImageDraw.Draw(img_pil)  # 创建绘画对象
    np.set_printoptions(threshold=np.inf)  # 打印不省略
    # print("result:\n", result)
    # print("type(result)", type(result))
    # print("result.shape", result.shape)
    for i in range(len(result)):
        # i = len(result) - i
        # for k in i:
        #     print(k)
        # print(i[1],i[0][0])
        # print(type(i[0][0]))
        # exit(0)
        # cv2.putText(img2, i[1], (i[0][0][0]+50, i[0][0][1]+50), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2, 8)
        # print("i", i)
        # print("i[0]", i[0])
        # print("i[1]", i[1])
        # print("i[1][1]", i[1][1])
        # print(type(i))
        # exit(0)
        # lml 11-19
        # print(int(i[1]))
        # if i[0] != '':
        #
        # if i > 1 and float(result[i][1]) - float(result[i-1][1]) > 20:   # 字距修正
        #     result[i][1] = str(float(result[i][1] + result[i-1][1])/2)

        # 行距修正
        bzhj = 15  # 标准行距
        if i > 0:
            # if result[i][0] == '' or result[i][0] == ' ':
            #     print("===========================")y
            #     result = np.delete(result, i, axis=0)
            #     continue
            qianhangju = float(result[i][2]) - float(result[i - 2][2])
            hangju = float(result[i][2]) - float(result[i - 1][2])
            #   前项大错位行修正
            if qianhangju < bzhj and hangju > bzhj:
                result[i - 1][2] = result[i - 2][2]
            #   当前项微偏差行修正
            if bzhj > hangju > -bzhj:  # 行距修正    上下偏差小于10像素的进行对齐
                # result[i][2] = str((float(result[i][2]) + float(result[i-1][2]))/2)
                result[i][2] = result[i - 1][2]

            bjzf = 15
            # ziju = float(result[i][1]) - float(result[i - 1][1])     # + float(result[i - 1][3])
            # print("len = ", len(result[i - 1][0]))
            hangju = float(result[i][2]) - float(result[i - 1][2])

            if hangju == 0:  # 字距修正
                # if hangju == 0 and ziju > len(result[i - 1][0]) * bjzf:   # 字距修正
                #     result[i][1] = str((float(result[i][1]) + float(result[i-1][3]))*2)
                # result[i][1] = str(float(result[i-1][1]) + len(result[i - 1][0]) * bjzf)
                # result[i][0] = '死的是个名字12'
                lenTxt = len(result[i - 1][0])  # 英文编码下总长 全为1
                lenTxt_utf8 = len(result[i - 1][0].encode('utf-8'))  # utf-8编码下总长 中文为3 英文数字为1
                size = float((lenTxt_utf8 - lenTxt) / 2 + lenTxt) / 2  # 计算差值
                # print(lenTxt, lenTxt_utf8, size)
                # exit(0)
                result[i][1] = str(float(result[i - 1][1]) + size * bjzf)

        draw.text((float(result[i][1]), (float(result[i][2])) + 10), result[i][0], font=font, fill=(0, 0, 0, 0))
    # print("result:\n", result)
    # print("result: ", type(result))
    #   坐标  汉字  字体  颜色
    # fontpath = "DATA/Fonts/simsun.ttc" # <== 这里是宋体路径
    # font = ImageFont.truetype(fontpath, 32)
    # img_pil = Image.fromarray(img2)
    # draw = ImageDraw.Draw(img_pil)
    # draw.text((50, 80),  "端午节就要到1。。。", font = font, fill = (0,0,0,0))
    img_zh = np.array(img_pil)
    return img_zh


def draw_text_paddle(img, result):
    # print("draw_text_paddle")
    # print(result)
    # print(result.shape)
    # exit(0)
    fontpath = "DATA/Fonts/simsun.ttc"  # <== 这里是宋体路径
    font = ImageFont.truetype(fontpath, 15)  # <== 字号
    img_pil = Image.fromarray(img)  # opencv数据图转PLT数据图
    draw = ImageDraw.Draw(img_pil)  # 创建绘画对象
    for i in result:
        # for k in i:
        #     print(k)
        # print(i[1],i[0][0])
        # print(type(i[0][0]))
        # exit(0)
        # cv2.putText(img2, i[1], (i[0][0][0]+50, i[0][0][1]+50), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2, 8)
        # print(i)
        # print(i[0][0][0])
        # print(i[0])
        # print(i[1])
        # print(i[1][0])
        # exit(0)
        draw.text((i[0][0][0] + 10, i[0][0][1] + 10), i[1][0], font=font, fill=(0, 0, 0, 0))
        #   坐标  汉字  字体  颜色
    # fontpath = "DATA/Fonts/simsun.ttc" # <== 这里是宋体路径
    # font = ImageFont.truetype(fontpath, 32)
    # img_pil = Image.fromarray(img2)
    # draw = ImageDraw.Draw(img_pil)
    # draw.text((50, 80),  "端午节就要到1。。。", font = font, fill = (0,0,0,0))
    img_zh = np.array(img_pil)
    return img_zh


def draw_text_baiduocr(img, result):
    # print("draw_text_paddle")
    # print(result)
    # print(result.shape)
    # exit(0)
    fontpath = "DATA/Fonts/simsun.ttc"  # <== 这里是宋体路径
    font = ImageFont.truetype(fontpath, 15)  # <== 字号
    img_pil = Image.fromarray(img)  # opencv数据图转PLT数据图
    draw = ImageDraw.Draw(img_pil)  # 创建绘画对象
    for i in result:
        draw.text((int(i[1]) + 10, int(i[2]) + 10), i[0], font=font, fill=(0, 0, 0, 0))
        #   坐标  汉字  字体  颜色
    # fontpath = "DATA/Fonts/simsun.ttc" # <== 这里是宋体路径
    # font = ImageFont.truetype(fontpath, 32)
    # img_pil = Image.fromarray(img2)
    # draw = ImageDraw.Draw(img_pil)
    # draw.text((50, 80),  "端午节就要到1。。。", font = font, fill = (0,0,0,0))
    img_zh = np.array(img_pil)
    return img_zh


def easyocr_run(img):
    reader = easyocr.Reader(['ch_sim', 'en'])
    result = reader.readtext(img)
    return result


def tesseract_run(img):
    # text = pytesseract.image_to_string(img, lang="chi_sim")
    dict = pytesseract.image_to_data(img, lang="chi_sim", output_type='dict')
    # tuple = dict.items()
    # # level   page_num    block_num   par_num line_num   word_num  left   top width   height  conf    text
    # # print(len(dict['text']))
    # num = len(dict['text'])
    # # result_t = [[] for i in range(num)]
    # result_t = np.zeros(shape=(num, 3)).astype('str')
    # # print(result_t.shape)
    # # print(result_t)
    # for i in tuple:
    #     if i[0] == 'left':
    #         for j in range(num):
    #             # print(str(i[1][j-1]))
    #             result_t[j-1][1] = str(i[1][j-1])
    #     if i[0] == 'top':
    #         for j in range(num):
    #             # print(str(i[1][j-1]))
    #             result_t[j-1][2] = str(i[1][j-1])
    #     if i[0] == 'text':
    #         for j in range(num):
    #             # print("i[1][j-1]", i[1][j-1])
    #             # print("ty", type(i[1][j-1]))
    #             # print(type(str(i[1][j-1])))
    #             # print(type(result_t[j-1][2]))
    #             # exit(0)
    #             # continue
    #             result_t[j-1][0] = i[1][j-1]
    #     #
    #     # if i[0] == 'top':
    #     #
    #     # if i[0] == 'text':
    # # print(result_t)
    # # exit(0)
    return dict
    # get_tesseract_version返回系统中安装的Tesseract版本。
    # image_to_string将图像上的Tesseract-OCR运行结果返回到字符串
    # image_to_boxes返回包含已识别字符及其框边界的结果
    # image_to_data返回包含框边界，置信度和其他信息的结果。需要Tesseract3.05 +。有关更多信息，请查看Tesseract TSV文档
    # image_to_osd返回包含有关方向和脚本检测的信息的结果。
    # 参数：
    # image_to_data(image, lang=None, config='', nice=0, output_type=Output.STRING)
    # image object　　图像对象
    # lang String，Tesseract　　语言代码字符串
    # config String　　任何其他配置为字符串，例如：config = '--psm 6'
    # nice Integer　　修改Tesseract运行的处理器优先级。Windows不支持。尼斯调整了类似unix的流程的优点。
    # output_type　　类属性，指定输出的类型，默认为string。有关所有支持类型的完整列表，请检查pytesseract.Output类的定义。
    # class Output:
    #     BYTES = 'bytes'
    #     DATAFRAME = 'data.frame'
    #     DICT = 'dict'
    #     STRING = 'string'

    # def tesseract_to_easyor(dict):
    #     tuple = dict.items()
    #     # level   page_num    block_num   par_num line_num   word_num  left   top width   height  conf    text
    #     # print(len(dict['text']))
    #     num = len(dict['text'])
    #     # result_t = [[] for i in range(num)]
    #     result_t = np.zeros(shape=(num, 3)).astype('str')
    #     # print(result_t.shape)
    #     # print(result_t)
    #     for i in tuple:
    #         if i[0] == 'left':
    #             for j in range(num):
    #                 # print(str(i[1][j-1]))
    #                 result_t[j - 1][1] = str(i[1][j - 1])
    #         if i[0] == 'top':
    #             for j in range(num):
    #                 # print(str(i[1][j-1]))
    #                 result_t[j - 1][2] = str(i[1][j - 1])
    #         if i[0] == 'text':
    #             for j in range(num):
    #                 # print("i[1][j-1]", i[1][j-1])
    #                 # print("ty", type(i[1][j-1]))
    #                 # print(type(str(i[1][j-1])))
    #                 # print(type(result_t[j-1][2]))
    #                 # exit(0)
    #                 # continue
    #                 result_t[j - 1][0] = i[1][j - 1]
    #         #
    #         # if i[0] == 'top':
    #         #
    #         # if i[0] == 'text':
    #     # print(result_t)
    #     # exit(0)
    #     return result_t


def paddle_run(img):
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
    # img_path = './imgs/11.jpg'
    # print("=====================================")
    result = ocr.ocr(img, cls=True)
    # for line in result:
    #     print(line)
    return result


def baiduocr_run(img_path):
    # https://ai.baidu.com/ai-doc/OCR/7kibizyfm
    """ 你的 APPID AK SK """
    APP_ID = '25209673'
    API_KEY = 'qcKDF8V60rokLmF31b8mQGut'
    SECRET_KEY = 'VE9Ktl0zFYElFyHEj13OXGNmz2KUnebA'
    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

    def get_file_content(img_path):
        with open(img_path, 'rb') as fp:
            return fp.read()

    img = get_file_content(img_path)

    # options = {}
    # options["recognize_granularity"] = "small"
    # options["detect_direction"] = "true"
    # dict = client.general(img, options)
    # dict = client.accurate(img, options)
    dict = client.accurate(img)

    # num = len(dict['words_result'])
    # print(num)

    # exit(0)
    return dict


def tesseract_result_to_tuple(dict):
    tuple = dict.items()
    # level   page_num    block_num   par_num line_num   word_num  left   top width   height  conf    text
    # print(len(dict['text']))
    num = len(dict['text'])
    # result_t = [[] for i in range(num)]
    result_t = np.zeros(shape=(num, 5)).astype('str')
    # print(result_t.shape)
    # print(result_t)
    for i in tuple:
        if i[0] == 'left':
            for j in range(num):
                # print(str(i[1][j-1]))
                result_t[j - 1][1] = str(i[1][j - 1])
        if i[0] == 'top':
            for j in range(num):
                # print(str(i[1][j-1]))
                result_t[j - 1][2] = str(i[1][j - 1])
        if i[0] == 'width':
            for j in range(num):
                # print(str(i[1][j-1]))
                result_t[j - 1][3] = str(i[1][j - 1])
        if i[0] == 'height':
            for j in range(num):
                # print(str(i[1][j-1]))
                result_t[j - 1][4] = str(i[1][j - 1])
        if i[0] == 'text':
            for j in range(num):
                # print("i[1][j-1]", i[1][j-1])
                # print("ty", type(i[1][j-1]))
                # print(type(str(i[1][j-1])))
                # print(type(result_t[j-1][2]))
                # exit(0)
                # continue
                result_t[j - 1][0] = i[1][j - 1]

        #
        # if i[0] == 'top':
        #
        # if i[0] == 'text':
    # print(result_t)
    # exit(0)
    return result_t


def baiduocr_result_to_tuple(dict):
    # result = client.basicAccurate(img)
    # print(dict)
    # print(type(dict))
    # print(dict['words_result'])
    # print(type(dict['words_result']))
    # print(dict['words_result'][0])
    # print(type(dict['words_result'][0]))
    # print(dict['words_result'][0]['words'])
    # print(type(dict['words_result'][0]['words']))

    # print(result.shape)
    # tuple = dict.items()
    # print(tuple)
    # print(tuple[0])
    # print(tuple[0][0])
    # print(tuple)
    # level   page_num    block_num   par_num line_num   word_num  left   top width   height  conf    text
    # print(len(dict['text']))
    num = len(dict['words_result'])
    # result_t = [[] for i in range(num)]
    result_b = np.zeros(shape=(num, 5)).astype('str')
    # print(result_t.shape)
    # print(dict['words_result'])
    for i in range(len(dict['words_result'])):
        # print(i)
        # print(dict['words_result'][i]['words'])
        # print(dict['words_result'][i])
        # print(dict['words_result'][i]['location'])
        # print(type(dict['words_result'][i]['location']))
        # {'words': '人工智能学院、人工智能研究院建设情况', 'location': {'top': 7, 'left': 4, 'width': 267, 'height': 17}}

        # print(dict['words_result'][i])

        # i[]
        result_b[i][0] = dict['words_result'][i]['words']
        result_b[i][1] = dict['words_result'][i]['location']['left']
        result_b[i][2] = dict['words_result'][i]['location']['top']
        result_b[i][3] = dict['words_result'][i]['location']['width']
        result_b[i][4] = dict['words_result'][i]['location']['height']

        #
        # if i[0] == 'top':
        #
        # if i[0] == 'text':
    # print(result_b)

    # exit(0)
    return result_b


def ocr_test():
    # 创建reader对象
    # reader = easyocr.Reader(['ch_sim', 'en'])
    # 读取图像

    # img_mb = cv.imdecode(np.fromfile("DATA/yt1.jpg", dtype=np.uint8), -1)
    # # tpl = cv.imread("DATA/yt1.jpg")  # 模板

    # img_path = 'DATA/img/字符识别ocr/数字仪表盘.jpg'
    img_path = '../DATA/img/字符识别ocr/长段文字.png'
    img = lml_imread.imread(img_path)
    cv2.imshow("yuantu", img)
    # img = lml_imread.imread('DATA/img/字符识别ocr/数字仪表盘.jpg')
    # t1 = lml_time.get_time_ymd_hms_ms()

    # Paddle
    tp1 = lml_time.get_time_ymd_hms_ms()
    result_p = paddle_run(img)
    tp2 = lml_time.get_time_ymd_hms_ms()
    tp = lml_time.get_time_yunsuan(tp1, tp2)
    img3 = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)  # 创建黑色背景布
    img3.fill(255)  # 填充成白色背景布
    img_zh_p = draw_text_paddle(img3, result_p)
    cv2.imshow("Paddle", img_zh_p)
    print("\nPaddle    time= ", tp)

    # Tesseract
    img3 = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)  # 创建黑色背景布
    img3.fill(255)  # 填充成白色背景布
    tt1 = lml_time.get_time_ymd_hms_ms()
    result_t = tesseract_run(img)
    tt2 = lml_time.get_time_ymd_hms_ms()
    tt = lml_time.get_time_yunsuan(tt1, tt2)
    result_t = tesseract_result_to_tuple(result_t)
    # print(result_t)
    # print(result_t)
    # print(type(result_t))
    # print(np.array(result_t, dtype=object).shape)
    # exit(0)
    img_zh_t = draw_text_tesseract(img3, result_t)
    cv2.imshow("Tesseract", img_zh_t)
    print("Tesseract time= ", tt)

    # easyocr
    img2 = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)  # 创建黑色背景布
    img2.fill(255)  # 填充成白色背景布
    # result = reader.readtext(img)    # , detail=0 可以只保存识别结果，不输出坐标、识别度等其他信息
    te1 = lml_time.get_time_ymd_hms_ms()
    result_e = easyocr_run(img)  # , detail=0 可以只保存识别结果，不输出坐标、识别度等其他信息
    te2 = lml_time.get_time_ymd_hms_ms()
    te = lml_time.get_time_yunsuan(te1, te2)
    # t2 = lml_time.get_time_ymd_hms_ms()
    # print(lml_time.get_time_yunsuan(t1, t2))    # 0:00:01.922000
    # 结果
    # print(np.array(result_e, dtype=object).shape)
    # print(result)
    img_zh_e = draw_text_easyocr(img2, result_e)  # 将中文打印到图片上(带位置)
    cv2.imshow("easyocr", img_zh_e)
    print("easyocr   time= ", te)

    # baidu-ocr
    tb1 = lml_time.get_time_ymd_hms_ms()
    result_b = baiduocr_run(img_path)
    tb2 = lml_time.get_time_ymd_hms_ms()
    tb = lml_time.get_time_yunsuan(tb1, tb2)
    result_b = baiduocr_result_to_tuple(result_b)
    img4 = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)  # 创建黑色背景布
    img4.fill(255)  # 填充成白色背景布
    img_zh_p = draw_text_baiduocr(img4, result_b)
    cv2.imshow("baidu-ocr", img_zh_p)
    print("baidu-ocr time= ", tb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit(0)


ocr_test()
