// lml-vs-c.cpp : ���ļ����� "main" ����������ִ�н��ڴ˴���ʼ��������
//

#include <iostream>
#include <Python.h>
#include <string>
using namespace std;

int main2()
{
    //int a = 0;
    Py_SetPythonHome(L"D:\\Anaconda3\\envs\\py37");
    Py_Initialize(); //����Python֮ǰҪ��ʼ��
    //std::cout << "Hello World!2\n";
    PyRun_SimpleString("import sys"); //������ҪPython��������Ҫ�Ŀ⣬��Щ��Ҳ����ֱ�������Python�ļ��
    PyRun_SimpleString("sys.path.append('./')");//ͬ��
    //cout << "1";
    PyRun_SimpleString("import lml_hello"); //������ҪPython��������Ҫ�Ŀ⣬��Щ��Ҳ����ֱ�������Python�ļ��
    //PyRun_SimpleString("sys.path.append('../')");//ͬ��
    //cout << "2";
    //PyRun_SimpleString("print ('hw')");//--��ӡ
    PyRun_SimpleString("lml_hello.hw()");//--��ӡ
    //cout << a;
    Py_Finalize(); //--����python�����ͷ���Դ
    //cout << "3";
}



/*
int main()
{
    // ����DLL�ļ�
    HINSTANCE hDllInst;
    hDllInst = LoadLibrary("lml_hello.pyd"); //����DLL
    if (hDllInst) {
        typedef long* (*PLUSFUNC)(); //���Ϊ������ǰ��Ϊ����ֵ
        PLUSFUNC plus_str = (PLUSFUNC)GetProcAddress(hDllInst, "1000"); //GetProcAddressΪ��ȡ�ú����ĵ�ַ

        cout << hDllInst << plus_str << "�������" << endl;
    }
    else {
        cout << "DLL����ʧ��" << endl;
    }

    return 0;
}
*/