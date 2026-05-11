import socket
import struct
import threading
import time
from scapy.all import sniff, wrpcap

# --- 配置参数 ---
HOST = '127.0.0.1'
PORT = 18085
PCAP_FILENAME = "gh0st_variant_c2.pcap"
MAGIC_BYTES = b'xy '

def xor_encrypt(data: bytes, key: int = 0x12) -> bytes:
    """简单的单字节异或加密，模拟样本中的异或处理"""
    return bytes([b ^ key for b in data])

# --- 1. 模拟 C2 服务端 (Server) ---
def start_mock_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[Server] 正在监听 {HOST}:{PORT}...")
        
        conn, addr = s.accept()
        with conn:
            print(f"[Server] 接收到来自 {addr} 的连接")
            try:
                # 接收头部数据
                header = conn.recv(15)
                if not header or not header.startswith(MAGIC_BYTES):
                    return
                
                # 解析包头结构
                magic = header[:3]
                total_len, effect_len, fixed_val = struct.unpack('<III', header[3:15])
                print(f"[Server] 解析包头 - Magic: {magic}, 总长度: {total_len}, 有效数据长度: {effect_len}, 固定值: {fixed_val}")
                
                # 接收剩余有效数据
                remaining_len = total_len - 15
                payload = conn.recv(remaining_len)
                
                data_type = payload[0]
                encrypted_data = payload[1:]
                
                print(f"[Server] 数据类型: {data_type}")
                print(f"[Server] 接收到的加密负载 (Hex): {encrypted_data.hex()}")
                
                time.sleep(1)
            except Exception as e:
                print(f"[Server] 错误: {e}")

# --- 2. 模拟被控端 (Client) ---
def start_mock_client():
    # 增加等待时间，确保服务端和嗅探器已经完全启动
    time.sleep(3) 
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            print(f"[Client] 已连接到 C2 {HOST}:{PORT}")
            
            # 构造模拟的主机信息
            mock_host_info = b"WIN-TEST-PC|192.168.1.100|Windows 10|3.2GHz|8GB"
            encrypted_info = xor_encrypt(mock_host_info)
            
            # 组装有效数据: 1字节类型 + 加密数据
            data_type = b'\x01'
            effective_data = data_type + encrypted_info
            
            # 计算长度
            effect_len = len(effective_data)
            total_len = 15 + effect_len 
            
            # 构造完整数据包 (小端序)
            packet = MAGIC_BYTES + struct.pack('<III', total_len, effect_len, 1) + effective_data
            
            print(f"[Client] 发送模拟上线数据包，总长度: {total_len} 字节")
            s.sendall(packet)
            
            # 保持连接，让交互完整完成
            time.sleep(1)
        except Exception as e:
            print(f"[Client] 发生异常: {e}")

# --- 3. 抓包器 (Sniffer) ---
def capture_traffic():
    print(f"[Sniffer] 开始抓取 lo 接口上端口 {PORT} 的流量 (持续 6 秒)...")
    # 去掉 host 限制，避免 loopback 的 BPF 解析问题
    bpf_filter = f"tcp port {PORT}"
    
    # 显式指定 iface='lo'，移除 count 限制，抓取完整交互周期
    packets = sniff(iface='lo', filter=bpf_filter, timeout=6)
    
    if len(packets) > 0:
        wrpcap(PCAP_FILENAME, packets)
        print(f"[Sniffer] 🎉 抓包成功！共捕获 {len(packets)} 个数据包，已保存至当前目录下的 {PCAP_FILENAME}")
    else:
        print("[Sniffer] 未捕获到任何数据包。")

if __name__ == "__main__":
    print("=== 开始模拟 Gh0st 变种 C2 流量 ===")
    
    server_thread = threading.Thread(target=start_mock_server)
    server_thread.daemon = True
    server_thread.start()

    client_thread = threading.Thread(target=start_mock_client)
    client_thread.daemon = True
    client_thread.start()

    # 主线程阻塞抓包
    capture_traffic()
    
    print("=== 模拟结束 ===")