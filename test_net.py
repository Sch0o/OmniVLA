import roslibpy
client = roslibpy.Ros(host='10.89.75.54', port=9090)
client.run(timeout=5)
print("连接成功!" if client.is_connected else "连接失败!")
client.terminate()