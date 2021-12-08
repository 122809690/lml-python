# 引入模块
import lml_time
from obs import ObsClient

# import sys
# sys.path.append('../')

# 创建ObsClient实例
obsClient = ObsClient(
    access_key_id='	EUADZYSETKCIM5X5C64F',
    secret_access_key='fNQCZR0R6JZrTf4sGAwQNzfuqJRDzsYZO2aOsxfZ',
    server='obs.cn-north-4.myhuaweicloud.com'
)
# 使用访问OBS

if  0   :
    # 调用createBucket创建桶
    resp = obsClient.createBucket('bucketname')
    if resp.status < 300:
        # 输出请求Id
        print('requestId:', resp.requestId)
    else:
        # 输出错误码
        print('errorCode:', resp.errorCode)
        # 输出错误信息
        print('errorMessage:', resp.errorMessage)

if  0   :
    # 调用putContent接口上传对象到桶内
    resp = obsClient.putContent('bucketname', 'objectname', 'Hello OBS')
    if resp.status < 300:
        # 输出请求Id
        print('requestId:', resp.requestId)
    else:
        # 输出错误码
        print('errorCode:', resp.errorCode)
        # 输出错误信息
        print('errorMessage:', resp.errorMessage)

if  1   :
    try:
        # 文件上传 https://support.huaweicloud.com/sdk-python-devg-obs/obs_22_0904.html

        from obs import PutObjectHeader

        bucketName = "lml-obs"  # 桶名

        dt_ms = lml_time.get_time_ymd_hms_ms()
        dt_ymd = lml_time.get_time_ymd()
        objectKey = 'sdk-t/' + dt_ymd + '/' + dt_ms + '.jpg'  # 对象名，即上传后的路径和文件名
        # print(objectKey)

        file_path = r"C:\Users\LML-YLC-PC\Desktop\1.png"  # 待上传文件 / 文件夹的完整路径，如aa / bb.txt，或aa /。

        # headers = PutObjectHeader()
        # headers.contentType = 'text/plain'

        # resp = obsClient.putFile(bucketName, objectKey, file_path,
        #                          metadata={'meta1': 'value1', 'meta2': 'value2'}, headers=headers)
        resp = obsClient.putFile(bucketName, objectKey, file_path)

        if resp.status < 300:
            print('requestId:', resp.requestId)
            print('etag:', resp.body.etag)
            print('versionId:', resp.body.versionId)
            print('storageClass:', resp.body.storageClass)
        else:
            print('errorCode:', resp.errorCode)
            print('errorMessage:', resp.errorMessage)
    except:
        import traceback

        print(traceback.format_exc())

if  0   :
    try:
        from obs import PutObjectHeader

        bucketName = "lml-obs"  # 桶名
        objectKey = "t1.jpg"  # 对象名，即上传后的文件名
        file_path = "C:\\Users\\LML-YLC-PC\\Desktop\\1.png"  # 待上传文件 / 文件夹的完整路径，如aa / bb.txt，或aa /。

        # headers = PutObjectHeader()
        # headers.contentType = 'text/plain'

        # resp = obsClient.putFile(bucketName, objectKey, file_path,
        #                          metadata={'meta1': 'value1', 'meta2': 'value2'}, headers=headers)
        resp = obsClient.putFile(bucketName, objectKey, file_path)


        if resp.status < 300:
            print('requestId:', resp.requestId)
            print('etag:', resp.body.etag)
            print('versionId:', resp.body.versionId)
            print('storageClass:', resp.body.storageClass)
        else:
            print('errorCode:', resp.errorCode)
            print('errorMessage:', resp.errorMessage)
    except:
        import traceback

        print(traceback.format_exc())


# 关闭obsClient
obsClient.close()