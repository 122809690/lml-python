// lml-vs-c.cpp : ���ļ����� "main" ����������ִ�н��ڴ˴���ʼ��������
//

#include <iostream>
#include <Python.h>
#include <string>
#include <cstdio>
using namespace std;
#pragma execution_character_set("utf-8")



int JXB(char* JXB_msg2)
{
    //int a = 0;
    Py_SetPythonHome(L"./py37");
    //Py_SetPythonHome(L"D:\\Anaconda3\\envs\\py37");
    //Py_SetPythonHome(L"D:\\WORK\\lml-python\\x64\\Release\\py37");
    Py_Initialize(); //����Python֮ǰҪ��ʼ��
    //std::cout << "Hello World!2\n";
    if (!Py_IsInitialized()) {  //��Ϊ���·������  vs�����лᱨ��  ���ɵ�exe����ֱ����������   ȡ������·����ע�;Ϳ���������vs������
        return -1;
    }
    PyRun_SimpleString("import sys"); //������ҪPython��������Ҫ�Ŀ⣬��Щ��Ҳ����ֱ�������Python�ļ��
    PyRun_SimpleString("import socket"); 
    //PyRun_SimpleString("import os, sys, io, selectors"); 
    //PyRun_SimpleString(" from collections.abc import Mapping"); 
    //PyRun_SimpleString("import cpython"); //������ҪPython��������Ҫ�Ŀ⣬��Щ��Ҳ����ֱ�������Python�ļ��
    PyRun_SimpleString("sys.path.append('./')");//ͬ��
    //PyRun_SimpleString("sys.path.append('D:\\WORK\\lml-python\\x64\\Release)");//ͬ��

    PyRun_SimpleString("print ('start')");//--��ӡ
    cout << "1\n";
    PyRun_SimpleString("import lml_JXB_pyd"); //������ҪPython��������Ҫ�Ŀ⣬��Щ��Ҳ����ֱ�������Python�ļ��
    //PyRun_SimpleString("sys.path.append('../')");//ͬ��
    cout << "2\n";

    char JXB_msg[100];
    char* JXB_msg1 = "lml_JXB_pyd.robot_run(";
    char* JXB_msg3 = ")";
    //JXB_msg = "lml_JXB_pyd.robot_run('192.168.135.129',2,10,30,10,0,0,0)";
    //cout << strlen(JXB_msg) << "\n";



    //char* JXB_msg2 = "'192.168.135.129', 2, 0, 0, 0, 0, 0, -0";



    sprintf_s(JXB_msg, "%s%s%s", JXB_msg1, JXB_msg2, JXB_msg3);
    cout << JXB_msg << "\n";
    PyRun_SimpleString(JXB_msg);



    cout << "3\n";
    Py_Finalize(); //--����python�����ͷ���Դ
    cout << "end\n";

}

int main()
{
    char* msg = "'192.168.135.129', 2, 0, 0, 0, 0, 0, -0";
    JXB(msg);
}
