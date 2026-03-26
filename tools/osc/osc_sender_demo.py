"""
TouchDesigner OSC API 客户端实现
用于与 TouchDesigner 进行双向通信
"""

import argparse
import time
from typing import Any, List, Optional
from pythonosc import udp_client, dispatcher, osc_server
import threading
import json

class TouchDesignerClient:
    """TouchDesigner API 客户端类"""
    
    def __init__(self, td_ip: str = "127.0.0.1", 
                 td_port: int = 7000,
                 client_port: int = 7001):
        """
        初始化 TouchDesigner 客户端
        
        参数：
        td_ip: TouchDesigner 服务器 IP 地址
        td_port: TouchDesigner 接收端口
        client_port: 客户端接收端口
        """
        self.td_ip = td_ip
        self.td_port = td_port
        self.client_port = client_port
        
        # 创建 OSC 客户端
        self.client = udp_client.SimpleUDPClient(td_ip, td_port)
        
        # 创建 OSC 服务器用于接收响应
        self.dispatcher = dispatcher.Dispatcher()
        self.server = None
        self.server_thread = None
        
        # 存储接收到的数据
        self.received_data = {}
        
    def start_server(self):
        """启动 OSC 服务器以接收来自 TouchDesigner 的消息"""
        self.server = osc_server.ThreadingOSCUDPServer(
            ("127.0.0.1", self.client_port), 
            self.dispatcher
        )
        self.server_thread = threading.Thread(
            target=self.server.serve_forever
        )
        self.server_thread.daemon = True
        self.server_thread.start()
        print(f"OSC 服务器启动，监听端口: {self.client_port}")
        
    def stop_server(self):
        """停止 OSC 服务器"""
        if self.server:
            self.server.shutdown()
            
    def register_handler(self, address: str, handler):
        """
        注册消息处理器
        
        参数：
        address: OSC 地址模式
        handler: 处理函数
        """
        self.dispatcher.map(address, handler)
        
    def send_parameter(self, parameter_name: str, value: Any):
        """
        发送参数到 TouchDesigner
        
        参数：
        parameter_name: 参数名称
        value: 参数值
        """
        address = f"/parameter/{parameter_name}"
        self.client.send_message(address, value)
        print(f"发送: {address} = {value}")
        
    def send_control_command(self, command: str, args: Optional[List] = None):
        """
        发送控制命令到 TouchDesigner
        
        参数：
        command: 命令名称
        args: 命令参数列表
        """
        address = f"/control/{command}"
        if args:
            self.client.send_message(address, args)
        else:
            self.client.send_message(address, [])
        print(f"发送命令: {address}")
        
    def send_data_batch(self, data_dict: dict):
        """
        批量发送数据
        
        参数：
        data_dict: 数据字典
        """
        json_data = json.dumps(data_dict)
        self.client.send_message("/data/batch", json_data)
        print(f"批量发送数据: {len(data_dict)} 项")

# 使用示例
def main():
    """主函数示例"""
    
    # 步骤 1: 创建客户端实例
    td_client = TouchDesignerClient(
        td_ip="127.0.0.1",
        td_port=7000,
        client_port=7001
    )
    
    # 步骤 2: 定义消息处理器
    def handle_response(unused_addr, *args):
        """处理来自 TouchDesigner 的响应"""
        print(f"收到响应: {unused_addr}, 数据: {args}")
        td_client.received_data[unused_addr] = args
    
    # 步骤 3: 注册处理器
    td_client.register_handler("/response/*", handle_response)
    
    # 步骤 4: 启动服务器
    td_client.start_server()
    
    try:
        # 步骤 5: 发送各种类型的消息
        
        # 5.1 发送单个参数
        td_client.send_parameter("opacity", 0.8)
        td_client.send_parameter("scale", 1.5)
        td_client.send_parameter("rotation", 45.0)
        
        # 5.2 发送控制命令
        td_client.send_control_command("play")
        time.sleep(2)
        td_client.send_control_command("pause")
        
        # 5.3 发送复杂数据
        visualization_data = {
            "nodes": [
                {"id": 1, "x": 100, "y": 100, "size": 20},
                {"id": 2, "x": 200, "y": 150, "size": 30},
                {"id": 3, "x": 150, "y": 200, "size": 25}
            ],
            "edges": [
                {"source": 1, "target": 2, "weight": 0.5},
                {"source": 2, "target": 3, "weight": 0.8}
            ]
        }
        td_client.send_data_batch(visualization_data)
        
        # 步骤 6: 保持连接一段时间以接收响应
        time.sleep(5)
        
    finally:
        # 步骤 7: 清理资源
        td_client.stop_server()
        print("客户端关闭")

if __name__ == "__main__":
    main()