import socket
import numpy as np
import threading
import time
from collections import deque

class MEGDataReceiver:
    def __init__(self, host='localhost', port=12345, buffer_size=10240000, n_channels=32, fs=1000):
        """
        脑磁数据接收器
        
        参数:
            host (str): 监听主机地址
            port (int): 监听端口号
            buffer_size (int): 接收缓冲区大小
            n_channels (int): 通道数量
            fs (int): 采样率(Hz)
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.n_channels = n_channels
        self.fs = fs
        self.data_buffer = deque(maxlen=fs*10) # 10秒的数据缓冲区
        self.running = False
        self._lock = threading.Lock()
        self._metadata_event = threading.Event()
        
        # 启动信息（最终配置将在收到元数据后打印）
        print("\n=== MEG数据接收器 ===")
        print(f"监听地址: {self.host}")
        print(f"监听端口: {self.port}")
        print(f"缓冲区大小: {self.buffer_size} bytes")
        print("等待客户端数据...\n")

    def start_server(self):
        """启动数据接收服务器"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"服务器已启动，正在监听 {self.host}:{self.port}")
        print("等待客户端连接...")
        
        self.running = True
        self.thread = threading.Thread(target=self._accept_connections)
        self.thread.start()
        
    def _accept_connections(self):
        """接受客户端连接"""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                print(f"Connection from {addr}")
                threading.Thread(target=self._handle_client, args=(client_socket,)).start()
            except Exception as e:
                if self.running:
                    print(f"Connection error: {e}")
    def _recv_exact(self, sock, n):
        buf = b''
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("socket closed")
            buf += chunk
        return buf

    def _handle_client(self, client_socket):
        """处理客户端数据"""
        try:
            # 1) 接收元数据：4字节通道数 + 4字节采样率（均为大端序）
            meta = self._recv_exact(client_socket, 8)
            new_channels = int.from_bytes(meta[:4], 'big')
            new_fs = int.from_bytes(meta[4:8], 'big')
            with self._lock:
                self.n_channels = new_channels
                self.fs = new_fs
                self.data_buffer = deque(maxlen=self.fs*10)
            self._metadata_event.set()
            print(f"[DEBUG] Metadata received - channels: {self.n_channels}, fs: {self.fs}")
            print(">>> 最终配置已生效:\n"
                  f"通道数: {self.n_channels}\n"
                  f"采样率: {self.fs} Hz\n"
                  f"数据缓冲区长度: {self.fs*10} 样本\n")

            # 2) 循环接收帧：4字节 samples(大端) + n_channels*samples*8 字节的 float64(大端)
            while self.running:
                raw_samples = self._recv_exact(client_socket, 4)
                if not raw_samples:
                    break
                samples = int.from_bytes(raw_samples, 'big')
                expected = self.n_channels * samples * 8
                payload = self._recv_exact(client_socket, expected)

                # 解析为二维数组 (n_channels, samples)
                arr = np.frombuffer(payload, dtype='>f8', count=self.n_channels * samples)
                if arr.size != self.n_channels * samples:
                    print(f"[ERROR] Incomplete frame: expected {self.n_channels * samples}, got {arr.size}")
                    break
                arr2d = arr.reshape(self.n_channels, samples)
                # 入缓冲（逐样本向量）
                for sample in arr2d.T:
                    self.data_buffer.append(np.asarray(sample, dtype=np.float64))
        except Exception as e:
            print(f"[ERROR] Client handling error: {e}")
        finally:
            client_socket.close()

    def wait_for_metadata(self, timeout=None):
        """等待客户端发送元数据，成功返回 True，超时返回 False"""
        return self._metadata_event.wait(timeout=timeout)

    def get_latest_data(self, duration=1.0):
        """
        获取最新数据，并对数据不足或空缓冲做保护处理
        
        参数:
            duration (float): 数据时长(秒)
            
        返回:
            ndarray: 最新数据，形状为 (n_channels, n_samples)
        """
        n_samples = int(duration * self.fs)
        # 从缓冲区获取最近 n_samples 帧，每帧是长度为 n_channels 的向量
        buffer_list = list(self.data_buffer)
        latest = buffer_list[-n_samples:] if buffer_list else []

        # 缓冲区完全空
        if not latest:
            # print("Warning: data buffer is empty, returning zeros")
            return np.zeros((self.n_channels, n_samples), dtype=np.float64)  # 指定float64类型

        # 缓冲区中帧数不足，用零向量补齐到 n_samples
        if len(latest) < n_samples:
            miss = n_samples - len(latest)
            print(f"Warning: only have {len(latest)} samples, need {n_samples}, padding with {miss} zero-frames")
            # 每个零帧也是长度为 n_channels
            pad = [np.zeros(self.n_channels, dtype=np.float64) for _ in range(miss)]  # 指定float64类型
            latest = pad + latest

        # 转换为float64数组并转置
        return np.ascontiguousarray(np.array(latest, dtype=np.float64).T)

    def stop_server(self):
        """停止服务器"""
        self.running = False
        try:
            self.server_socket.close()
        except:
            pass
        self.thread.join()

    def get_meg_channels(self):
        """获取通道数"""
        return list(range(self.n_channels))

    def get_meg_samplingrate(self):
        """获取采样率"""
        return self.fs

# 使用示例
if __name__ == "__main__":
    receiver = MEGDataReceiver(host='localhost', port=12345, n_channels=344)
    receiver.start_server()
    
    try:
        while True:
            # 获取最新1秒的数据
            data = receiver.get_latest_data(duration=2.0)
            time.sleep(0.1)
    except KeyboardInterrupt:
        receiver.stop_server()