import os
import xml.dom.minidom as minidom
from shutil import copyfile

# import tree

# xmlpath =       'C:\\Users\\LML-YLC-PC\\Desktop\\xml-img\\xml_check_change\\'
# xmlpath_out =   'C:\\Users\\LML-YLC-PC\\Desktop\\xml-img\\xml_out\\'
#
# imgpath =       'C:\\Users\\LML-YLC-PC\\Desktop\\xml-img\\JPEG2\\'
# imgpath_out =   'C:\\Users\\LML-YLC-PC\\Desktop\\xml-img\\img_out\\'
#
# # outputpath = 'C:\\ylc_outimg\\'
#
# xmlfilenames = os.listdir(xmlpath)
# imgfilenames = os.listdir(imgpath)

if 0:
    def from_img_cp_xml():
        for file_name in imgfilenames:
            # xmlfilepath = os.path.abspath(x_path)
            # print(file_name)
            # file_name = inputpath + file_name
            # print(file_name)
            # print(file_name)
            # print(file_name[-3:])
            file_name = file_name.replace('jpg', 'xml')
            file_name = file_name.replace('bmp', 'xml')
            # file_name[-3:] = 'xml'
            # print(file_name)
            xmlfilepath = os.path.abspath(xmlpath + file_name)
            try:
                copyfile(xmlfilepath, xmlpath_out + file_name)
            except FileNotFoundError:
                print("not found xml =============================", file_name)
            # print(xmlfilepath)
            # exit(1)


    from_img_cp_xml()

if 1:
    def from_xml_cp_img():
        xmlpath = 'C:\\Users\\LML-YLC-PC\\Desktop\\新建文件夹\\xml\\'
        imgpath = 'C:\\Users\\LML-YLC-PC\\Desktop\\新建文件夹\\b_t1\\'
        imgpath_out = 'C:\\Users\\LML-YLC-PC\\Desktop\\新建文件夹\\img_out\\'
        xmlfilenames = os.listdir(xmlpath)

        # imgfilenames = os.listdir(imgpath)
        for file_name in xmlfilenames:
            # xmlfilepath = os.path.abspath(x_path)
            # print(file_name)
            # file_name = inputpath + file_name
            # print(file_name)
            # print(file_name)
            # print(file_name[-3:])
            # file_name[-3:] = 'xml'
            file_name = file_name.replace('xml', 'jpg')
            imgfilepath = os.path.abspath(imgpath + file_name)
            try:
                copyfile(imgfilepath, imgpath_out + file_name)
            except FileNotFoundError:
                file_name = file_name.replace('jpg', 'bmp')
                imgfilepath = os.path.abspath(imgpath + file_name)
                try:
                    copyfile(imgfilepath, imgpath_out + file_name)
                except FileNotFoundError:
                    print("not found img =============================", file_name)


    from_xml_cp_img()

if 0:
    def from_xml_check_box_is_big(xmlpath=xmlpath, xmlpath_out=xmlpath_out, big=20):
        for file_name in xmlfilenames:
            # xmlfilepath = os.path.abspath(x_path)
            # print(file_name)
            # file_name = "0118.xml"
            # print(file_name)
            xmlfilepath = os.path.abspath(xmlpath + file_name)
            # print("xml文件路径：", xmlfilepath)
            # print("=================")
            domobj = minidom.parse(xmlfilepath)
            x1 = domobj.getElementsByTagName("xmin")
            y1 = domobj.getElementsByTagName("ymin")
            x2 = domobj.getElementsByTagName("xmax")
            y2 = domobj.getElementsByTagName("ymax")
            # print(x1, x2, y1, y2)

            if (len(x1) != 0):
                for i in range(len(x1)):
                    # print(x1[i - 1].toxml()[6:-7], x2[i - 1].toxml()[6:-7], y1[i - 1].toxml()[6:-7], y2[i - 1].toxml()[6:-7])
                    #       if (x1[i - 1].toxml() != "<name>黑点</name>" and subElementObj[i - 1].toxml() != "<name>白点</name>"):  #
                    #             print("==============================================", file_name)
                    #             print(subElementObj[i - 1].toxml())
                    if (int(x2[i - 1].toxml()[6:-7]) - int(x1[i - 1].toxml()[6:-7]) >= big or
                            int(y2[i - 1].toxml()[6:-7]) - int(y1[i - 1].toxml()[6:-7]) >= big):
                        copyfile(xmlfilepath, xmlpath_out + file_name)

            # if (len(subElementObj) == 0):
            #     print("***********************************", file_name)
            # print(file_name)
            # print(file_name)
            # exit(1)


    from_xml_check_box_is_big()

if 0:
    def xml_find_kong(xmlpath=xmlpath):
        for file_name in xmlfilenames:
            # xmlfilepath = os.path.abspath(x_path)
            # print(file_name)
            # file_name = inputpath + file_name
            # print(file_name)
            xmlfilepath = os.path.abspath(xmlpath + file_name)
            # print("xml文件路径：", xmlfilepath)
            # print("=================")
            domobj = minidom.parse(xmlfilepath)
            subElementObj = domobj.getElementsByTagName("name")
            # print(len(subElementObj))
            if (len(subElementObj) != 0):
                for i in range(len(subElementObj)):
                    if (subElementObj[i - 1].toxml() != "<name>黑点</name>" and subElementObj[
                        i - 1].toxml() != "<name>白点</name>"):  #
                        print("==============================================", file_name)
                        print(subElementObj[i - 1].toxml())
            if (len(subElementObj) == 0):
                print("***********************************", file_name)

            # img = cv2.imread(inputpath + file_name, cv2.COLOR_BGR2RGB)  # ,cv2.IMREAD_GRAYSCALE)
            # img1 = img[0:1024, 0:1024]
            # img2 = img[0:1024, 1024:2048]
            # cv2.imwrite(outputpath + "L" + file_name, img1)
            # cv2.imwrite(outputpath + "R" + file_name, img2)


    xml_find_kong()

    # x_path = "D:\\WORK\\xml\\01.xml"
    # # nodes = list()
    # xmlfilepath = os.path.abspath(x_path)
    #
    #
    #
    # # def findNodeIndex(node):
    # #     for i in range(len(nodes)):
    # #         if (nodes[i] == node):
    # #             return str(i)  # +":"+nodes[i]
    #
    #
    # print("xml文件路径：", xmlfilepath)
    # print("=================")
    # # 得到文档对象rfg
    # domobj = minidom.parse(xmlfilepath)
    # # root=minidom.parse('sample.xml')
    # # print("xmldom.parse:", type(domobj))
    # # 得到元素对象
    # # elementobj = domobj.documentElement
    # # print(elementobj)
    # # print ("domobj.documentElement:", type(elementobj))
    # subElementObj = domobj.getElementsByTagName("name")
    #
    # print(subElementObj[0].toxml())
    #
    # # tree = minidom.parse(xmlfilepath)
    # # itemgroup = tree.getElementsByTagName('name')
    # # print(itemgroup[0].toxml())
    # # print(itemgroup[0].toxml())
