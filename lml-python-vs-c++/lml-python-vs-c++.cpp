// lml-vs-c.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <Python.h>
#include <string>
using namespace std;

int main1()
{
    //std::cout << "Hello World!1\n";
    //Py_SetPythonHome(L"C:\\Users\\lenovo\\AppData\\Local\\conda\conda\\envs\\python32\\Scripts");
    Py_SetPythonHome(L"D:\\Anaconda3\\envs\\py37");
    Py_Initialize(); //调用Python之前要初始化
    //std::cout << "Hello World!2\n";
    PyRun_SimpleString("import sys"); //加入需要Python代码中需要的库，这些库也可以直接添加在Python文件里。
    PyRun_SimpleString("sys.path.append('../')");//同上
    //PyRun_SimpleString("import os");
    //PyRun_SimpleString("print os.path.exists('./lml_JXB_test.py')");
    //PyRun_SimpleString("print os.path.exists('.lml_robotcontrol.py')");
    //std::cout << "Hello World!3\n";
    PyObject* pModule = NULL;
    PyObject* pFunc = NULL;
    pModule = PyImport_ImportModule("lml_JXB_test");//调用文件名
    std::cout << pModule << "\n";
    pFunc = PyObject_GetAttrString(pModule, "robot_nys"); //加载需要的函数
    if (!pFunc || !PyCallable_Check(pFunc))
    {
        std::cout << "==error1==";
        //return -1;
        //exit(2);
    }
    //std::cout << "Hello World!5\n";
    PyObject* pReturn = PyEval_CallObject(pFunc, NULL); //调用函数
    //此函数有两个参数，而且都是Python对象指针，其中pfunc是要调用的Python 函数，
    //一般说来可以使用PyObject_GetAttrString()获得，pargs是函数的参数列表，通常是使用Py_BuildValue()来构建。
    if (pReturn == NULL)
        std::cout << "==error2==";
        //_sleep(1);
        //return 0;
        //exit(3);
    //int pyri = -1;
    
    //PyArg_ParseTuple(pReturn, "i", &pyri);//char szBuffer[256] = {0};
    //pyri = _PyLong_AsInt(pReturn);
    const char *pyrc;
    pyrc = _PyUnicode_AsString(pReturn);

    //Py_Finalize(); //相当于释放，和Py_Initialize相反。


    std::cout << "pReturn == " << pReturn;
    std::cout << "\npyrc == " << pyrc;
    std::cout << "\nend!\n";
    return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件


 