#!/usr/bin/env python3
#-*- coding:utf-8 -*-

'''  server_2.aubo
client_ip="192.168.135.1" --客户端IP
port=6666 --服务器端口号
tcp.server.listen(port) --开始监听
while tcp.server.is_connected(client_ip)~=true do --等待客户端连接
    	sleep(1) --等待1秒
	--print("wait2")
end
print("client is connected!") --向系统信息打印信息
recv="" --标识符用以保存接收到的客户端数据
while recv~="quit" do--当接收到的数据不为quit时将一直执行循环语句
	--print("no quit")
	recv=tcp.server.recv_str_data(client_ip) --接收客户端数据
	--print(recv)
 	if recv~="" then --如果接收到的数据不为空，执行if下的语句
        	print("recv_data: "..recv) --向系统信息打印接收到的数据
        	tcp.server.send_str_data(client_ip,recv) --向客户端发送接收到的数据
        	print("send_data: "..recv) --向系统信息打印信息
		set_global_variable("V_I_choose",tonumber(recv)) --设置全局变量"V_I_choose"的值
    	end
 	sleep(0.2)--等待0.2秒
end
tcp.server.close()--停止监听
print("server closed!")--向系统信息打印信息
'''

''' move.aubo
init_global_move_profile()--初始化全局运动属性
set_joint_maxvelc({2,2,2,2,2,2})--设置轴动最大速度
set_joint_maxacc({10,10,10,10,10,10})--设置轴动最大加速度
move_joint({0,0,0,0,0,0},true)--轴动至零位
while true do--无限循环
	if ((get_global_variable("V_I_choose"))== (1)) then--如果获取到全局变量"V_l_choose"的值为1，执行以下语句
		move_joint({1,1,1,1,1,1},true)--轴动1关节1弧度
		move_joint({0,0,0,0,0,0},true)--轴动至零位
		set_global_variable("V_I_choose",0)--设置全局变量"V_l_choose""的值为0
	elseif ((get_global_variable("V_I_choose"))== (2)) then--如果获取到全局变量"V_l_choose”的值为2，执行以下语句
		move_joint({-3,-1,-1,0,0,0},true)--轴动2关节1弧度
		move_joint({0,0,0,0,0,0},true)--轴动至零位
		set_global_variable("V_I_choose",0)--设置全局变量"V_l_choose"的值为0
	elseif ((get_global_variable("V_I_choose"))==(3)) then--如果获取到全局变量"V_l_choose”的值为3，执行以下语句
		move_joint({1,1,0,0,-1,-1},true)--轴动3关节1弧度
		move_joint({0,0,0,0,0,0},true)--轴动至零位
		set_global_variable("V_I_choose",0)--设置全局变量""V_l_choose"的值为0
	end
	sleep(1)--等待1秒
end

'''
