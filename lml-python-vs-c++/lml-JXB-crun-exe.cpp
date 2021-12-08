// lml-vs-c.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <Python.h>
#include <string>
#include <cstdio>
using namespace std;

int main3()
{
    //int a = 0;
    Py_SetPythonHome(L"D:\\Anaconda3\\envs\\py37");
    Py_Initialize(); //调用Python之前要初始化
    //std::cout << "Hello World!2\n";
    PyRun_SimpleString("import sys"); //加入需要Python代码中需要的库，这些库也可以直接添加在Python文件里。
    //PyRun_SimpleString("import cpython"); //加入需要Python代码中需要的库，这些库也可以直接添加在Python文件里。
    PyRun_SimpleString("sys.path.append('./')");//同上
    PyRun_SimpleString("print ('start')");//--打印
    cout << "1\n";
    PyRun_SimpleString("import lml_JXB_pyd"); //加入需要Python代码中需要的库，这些库也可以直接添加在Python文件里。
    //PyRun_SimpleString("sys.path.append('../')");//同上
    cout << "2\n";

    char JXB_msg[100];
    char* JXB_msg1 = "lml_JXB_pyd.robot_run(";
    char* JXB_msg3 = ")";
    //JXB_msg = "lml_JXB_pyd.robot_run('192.168.135.129',2,10,30,10,0,0,0)";
    //cout << strlen(JXB_msg) << "\n";



    char* JXB_msg2 = "'192.168.135.129', 2, 0, 0, 0, 0, 0, -0";



    sprintf_s(JXB_msg, "%s%s%s", JXB_msg1, JXB_msg2, JXB_msg3);
    cout << JXB_msg << "\n";
    PyRun_SimpleString(JXB_msg);



    cout << "3\n";
    Py_Finalize(); //--清理python环境释放资源
    cout << "end\n";

}

