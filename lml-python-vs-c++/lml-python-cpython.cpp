// lml-vs-c.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <Python.h>
#include <string>
using namespace std;

int main2()
{
    //int a = 0;
    Py_SetPythonHome(L"D:\\Anaconda3\\envs\\py37");
    Py_Initialize(); //调用Python之前要初始化
    //std::cout << "Hello World!2\n";
    PyRun_SimpleString("import sys"); //加入需要Python代码中需要的库，这些库也可以直接添加在Python文件里。
    PyRun_SimpleString("sys.path.append('./')");//同上
    //cout << "1";
    PyRun_SimpleString("import lml_hello"); //加入需要Python代码中需要的库，这些库也可以直接添加在Python文件里。
    //PyRun_SimpleString("sys.path.append('../')");//同上
    //cout << "2";
    //PyRun_SimpleString("print ('hw')");//--打印
    PyRun_SimpleString("lml_hello.hw()");//--打印
    //cout << a;
    Py_Finalize(); //--清理python环境释放资源
    //cout << "3";
}



/*
int main()
{
    // 加载DLL文件
    HINSTANCE hDllInst;
    hDllInst = LoadLibrary("lml_hello.pyd"); //调用DLL
    if (hDllInst) {
        typedef long* (*PLUSFUNC)(); //后边为参数，前面为返回值
        PLUSFUNC plus_str = (PLUSFUNC)GetProcAddress(hDllInst, "1000"); //GetProcAddress为获取该函数的地址

        cout << hDllInst << plus_str << "调用完成" << endl;
    }
    else {
        cout << "DLL加载失败" << endl;
    }

    return 0;
}
*/