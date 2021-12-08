try:
    from obs import PutObjectHeader

    headers = PutObjectHeader()
    headers.contentType = 'text/plain'

    resp = obsClient.putFile('bucketname', 'objectkey', 'localfile', metadata={'meta1': 'value1', 'meta2': 'value2'},
                             headers=headers)

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