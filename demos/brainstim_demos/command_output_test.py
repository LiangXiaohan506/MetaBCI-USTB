import socket


def connect_and_receive_instructions():
    # 服务器配置
    server_ip = 'localhost'  # 目标服务器IP
    server_port = 12345  # 目标服务器端口

    # 创建socket对象
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            # 连接服务器
            print(f"尝试连接 {server_ip}:{server_port}...")
            s.connect((server_ip, server_port))
            print("连接成功，等待接收指令...")

            while True:
                # 接收数据
                data = s.recv(1024)
                if not data:
                    print("连接已关闭")
                    break

                instruction = data.decode('utf-8').strip()
                print(f"收到指令: {instruction}")

                # 收到任何指令后立即返回"arrived"
                s.sendall("arrived".encode('utf-8'))
                print("已发送响应: arrived")

                # 这里可以添加特定指令的处理逻辑
                # if instruction == "特定指令":
                #    执行特定操作

        except ConnectionRefusedError:
            print(f"连接被拒绝，请确保 {server_ip} 正在监听端口 {server_port}")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            print("断开连接")


if __name__ == "__main__":
    connect_and_receive_instructions()