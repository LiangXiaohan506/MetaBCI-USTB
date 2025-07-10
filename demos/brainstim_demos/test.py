import socket

# 创建TCP服务器
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '172.20.10.7'  # 服务器IP
port = 12345        # 端口号
server_socket.bind((host, port))
server_socket.listen(1)

print(f"Server listening on {host}:{port}")

# 接受客户端连接
client_socket, address = server_socket.accept()
print(f"Connected by {address}")

# 手动输入指令
while True:
    command = input("Enter command (1, 2, 3, 4) or 'exit' to quit: ")
    if command.lower() == 'exit':
        break
    if command in ['1', '2', '3', '4']:
        message = command.encode('ascii')  # 转换为字节
        client_socket.send(message)
        print(f"Sent: {command}")
    else:
        print("Invalid command! Use 1, 2, 3, 4, or exit.")

# 关闭连接
client_socket.close()
server_socket.close()
print("Server disconnected.")