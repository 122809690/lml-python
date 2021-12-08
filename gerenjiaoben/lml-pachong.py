import ssl
from urllib import request, parse

import requests


def pachong_urllib_test():
    # response = urllib.request.urlopen('http://www.baidu.com')
    # print(response.read().decode('utf-8'))
    # urlopen 默认是 Get 请求  当我们传入参数它就为 Post 请求了
    # urllib.request.urlopen(url, data=None, [timeout, ]*)
    # 第一个 url 是请求的链接
    # 第二个参数 data 就是专门给我们 post 请求携带参数的 比如我们在登录的时候 可以把用户名密码封装成 data 传过去
    # 在这里的 data 的值我们可以用 byte 的类型传递
    # 第三个参数 timeout 就是设置请求超时时间

    # urllib.request.Request(url, data=None, headers={}, method=None)
    #  Request 可以让我们自己定义请求的方式   定义请求头信息
    #

    url = 'https://music.liuzhijin.cn/'

    # context = ssl._create_unverified_context()
    context = ssl.create_default_context()

    headers = {
        # 假装自己是浏览器
        'User-Agent': ' Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
    }

    dict = {
        # 'return_url': 'http://zhjw.scu.edu.cn/login',
        'input': 'https://music.163.com/#/song?id=1343434284',
        'filter': 'url',
        'type': '_',
        'page': '1',
        # 'password': '134c4dc7b90fd032a6981d981d768965',
        # # '_post_type': 'ajax',
        # 'j_captcha': 'agw4'
    }

    data = bytes(parse.urlencode(dict), 'utf-8')

    req = request.Request(url, data=data, headers=headers, method='POST')

    response = request.urlopen(req, context=context)
    print(response.read().decode('utf-8'))


# pachong_urllib_test()

def pachong_requests_test():
    # get请求
    r = requests.get('https://api.github.com/events')
    # post请求
    r = requests.post('https://httpbin.org/post', data={'key': 'value'})

    # Http请求
    # r = requests.put('https://httpbin.org/put', data={'key': 'value'})
    # r = requests.delete('https://httpbin.org/delete')
    # r = requests.head('https://httpbin.org/get')
    # r = requests.options('https://httpbin.org/get')

    # 携带请求参数
    # payload = {'key1': 'value1', 'key2': 'value2'}
    # r = requests.get('https://httpbin.org/get', params=payload)