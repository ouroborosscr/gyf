#!/usr/bin/env python3
"""
Smargaft C2 流量模拟器 (研究用途)
===================================
参考: https://blog.xlab.qianxin.com/smargaft_abusing_binance-smart-contracts_en/

模拟 Smargaft 僵尸网络 C2 通信协议, 用于生成 pcap 包进行流量防御智能体的召回测试。

Smargaft = Smart Contract + Gafgyt, 首个使用 EtherHiding (Binance Smart Chain
智能合约托管 C2) 的僵尸网络。

核心流量特征 (3 个阶段):
  阶段 1: EtherHiding — Bot 通过 JSON-RPC (HTTP POST) 调用 BSC 智能合约
           eth_call 方法 0xd7ec3ad7 获取真实 C2 地址
           合约地址: 0xdf2208d4902aa1ec9a0957132ca86a4e1d40455b
           目标: 14 个硬编码 BSC RPC 节点

  阶段 2: C2 通信 — Bot 连接 C2 的 81 端口, 发送 5 字节 ready 包
           协议为纯文本, 支持 15 种命令

  阶段 3: DDoS 指令执行 — 文本指令如:
           "udph <ip> <port> <duration>"
           "tcph <ip> <port> <duration>"
           "http <url> <duration>"
           "shell <cmd>"
           "socks5 <port>"

用法:
  python3 smargaft_c2_sim.py                        # 运行全部模拟并生成 pcap
  python3 smargaft_c2_sim.py --bots 3 --port 81     # 指定 bot 数量和端口
"""

import argparse, hashlib, json, os, socket, struct, sys, threading, time
import traceback, random
from datetime import datetime

try:
    import dpkt
except ImportError:
    print("[!] pip install dpkt --break-system-packages"); sys.exit(1)

# ═══════════════ 全局 PCAP 记录器 ═══════════════
PCAP_LOCK = threading.Lock()
PCAP_RECORDS = []
SEQ_COUNTERS = {}

def record_packet(src_ip, src_port, dst_ip, dst_port, data, flags='PA'):
    with PCAP_LOCK:
        key = (src_ip, src_port, dst_ip, dst_port)
        if key not in SEQ_COUNTERS:
            SEQ_COUNTERS[key] = 1000 + random.randint(0, 50000)
        seq = SEQ_COUNTERS[key]
        rev = (dst_ip, dst_port, src_ip, src_port)
        ack = SEQ_COUNTERS.get(rev, 0)
        PCAP_RECORDS.append((time.time(), src_ip, src_port, dst_ip, dst_port,
                             data, seq, ack, flags))
        SEQ_COUNTERS[key] = seq + max(len(data), 1)

def write_pcap(filename):
    with open(filename, 'wb') as f:
        writer = dpkt.pcap.Writer(f, linktype=dpkt.pcap.DLT_RAW)
        for rec in PCAP_RECORDS:
            ts, sip, sp, dip, dp, data, seq, ack, flags = rec
            tf = 0
            if 'S' in flags: tf |= dpkt.tcp.TH_SYN
            if 'A' in flags: tf |= dpkt.tcp.TH_ACK
            if 'P' in flags: tf |= dpkt.tcp.TH_PUSH
            if 'F' in flags: tf |= dpkt.tcp.TH_FIN
            tcp_p = dpkt.tcp.TCP(sport=sp, dport=dp, seq=seq, ack=ack,
                                 off=5, flags=tf, win=65535, data=data)
            ip_p = dpkt.ip.IP(src=socket.inet_aton(sip), dst=socket.inet_aton(dip),
                              p=dpkt.ip.IP_PROTO_TCP, ttl=64, data=tcp_p)
            ip_p.len = len(ip_p)
            writer.writepkt(bytes(ip_p), ts)
    log(f"[PCAP] 已写入 {len(PCAP_RECORDS)} 个数据包 → {filename}")

# ═══════════════ Recording Socket ═══════════════
class RecSock:
    def __init__(self, sock, lip='127.0.0.1', rip='127.0.0.1'):
        self._s = sock; self._li = lip; self._ri = rip; self._lp = 0; self._rp = 0

    def connect(self, addr):
        self._ri, self._rp = addr[0], addr[1]
        self._s.connect(addr)
        self._lp = self._s.getsockname()[1]
        record_packet(self._li, self._lp, self._ri, self._rp, b'', 'S')
        record_packet(self._ri, self._rp, self._li, self._lp, b'', 'SA')
        record_packet(self._li, self._lp, self._ri, self._rp, b'', 'A')

    def send(self, data):
        if isinstance(data, str): data = data.encode()
        r = self._s.send(data)
        record_packet(self._li, self._lp, self._ri, self._rp, data, 'PA')
        return r

    def sendall(self, data):
        if isinstance(data, str): data = data.encode()
        self._s.sendall(data)
        record_packet(self._li, self._lp, self._ri, self._rp, data, 'PA')

    def recv(self, n):
        data = self._s.recv(n)
        if data:
            record_packet(self._ri, self._rp, self._li, self._lp, data, 'PA')
        return data

    def close(self):
        record_packet(self._li, self._lp, self._ri, self._rp, b'', 'FA')
        self._s.close()

    def settimeout(self, t): self._s.settimeout(t)
    def getsockname(self): return self._s.getsockname()
    def setsockopt(self, *a): self._s.setsockopt(*a)
    def bind(self, a): self._s.bind(a)
    def listen(self, n): self._s.listen(n)

    def accept(self):
        cs, addr = self._s.accept()
        w = RecSock(cs, self._li, addr[0])
        w._lp = self._s.getsockname()[1]; w._rp = addr[1]; w._ri = addr[0]
        record_packet(addr[0], addr[1], self._li, w._lp, b'', 'S')
        record_packet(self._li, w._lp, addr[0], addr[1], b'', 'SA')
        record_packet(addr[0], addr[1], self._li, w._lp, b'', 'A')
        return w, addr

# ═══════════════ 协议常量 ═══════════════
# Smargaft 智能合约地址 (来自博客)
CONTRACT_ADDR = "0xdf2208d4902aa1ec9a0957132ca86a4e1d40455b"
# eth_call 方法签名 — 获取 C2 地址
GET_C2_METHOD = "0xd7ec3ad7"
# SET C2 方法签名
SET_C2_METHOD = "0x61695f0a"

# Smargaft 内嵌的 14 个 BSC RPC 节点地址 (博客提及)
BSC_RPC_NODES = [
    "https://bsc-dataseed.binance.org/",
    "https://bsc-dataseed1.defibit.io/",
    "https://bsc-dataseed1.ninicoin.io/",
    "https://bsc-dataseed2.defibit.io/",
    "https://bsc-dataseed3.binance.org/",
    "https://bsc-dataseed4.binance.org/",
    "https://bsc-dataseed2.ninicoin.io/",
    "https://bsc-dataseed3.defibit.io/",
    "https://bsc-dataseed4.defibit.io/",
    "https://bsc-dataseed3.ninicoin.io/",
    "https://bsc-dataseed4.ninicoin.io/",
    "https://bsc-dataseed1.binance.org/",
    "https://bsc-dataseed2.binance.org/",
    "https://rpc.ankr.com/bsc",
]

# C2 通信端口 (博客: DDoS task connects on port 81)
C2_PORT = 81

# Smargaft 的 15 种命令 (基于 Gafgyt 变体, 文本协议)
# 来自博客 Table 3
COMMANDS = {
    'udph':    'UDP flood with header',
    'tcph':    'TCP flood with header',
    'udp':     'UDP flood plain',
    'tcp':     'TCP flood plain',
    'std':     'STD flood',
    'http':    'HTTP GET flood',
    'httpx':   'HTTP POST flood',
    'hex':     'HEX payload flood',
    'vse':     'Valve Source Engine flood',
    'ack':     'TCP ACK flood',
    'syn':     'TCP SYN flood',
    'shell':   'Execute system command',
    'socks5':  'Start SOCKS5 proxy',
    'stop':    'Stop all attacks',
    'update':  'Update bot binary',
}

# 5 字节 ready 包 (博客: sends a 5-byte ready packet)
READY_PACKET = b'\x00\x01\x00\x00\x01'

# ═══════════════ 日志 ═══════════════
_LL = threading.Lock()
def log(msg):
    with _LL: print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")

# ═══════════════ 模拟 BSC RPC 节点服务器 ═══════════════
class BSCRPCServer:
    """模拟 Binance Smart Chain RPC 节点, 返回 C2 地址"""

    def __init__(self, host='127.0.0.1', port=8545, c2_ip='127.0.0.1', c2_port=81):
        self.host = host
        self.port = port
        self.c2_ip = c2_ip
        self.c2_port = c2_port
        self.running = False

    def start(self):
        self.running = True
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        raw.settimeout(1.0)
        raw.bind((self.host, self.port))
        raw.listen(10)
        self.ssock = RecSock(raw, self.host, self.host)
        log(f"[BSC RPC] 模拟 RPC 节点监听 {self.host}:{self.port}")

        while self.running:
            try:
                cs, addr = self.ssock.accept()
                threading.Thread(target=self._handle, args=(cs, addr), daemon=True).start()
            except socket.timeout: continue
            except OSError: break

    def stop(self): self.running = False

    def _handle(self, s, addr):
        """处理 JSON-RPC eth_call 请求"""
        tag = f"[BSC|{addr[1]}]"
        try:
            s.settimeout(5.0)
            data = s.recv(4096)
            if not data: s.close(); return

            raw = data.decode('utf-8', errors='ignore')
            # 解析 HTTP POST
            if 'POST' not in raw and 'eth_call' not in raw:
                s.close(); return

            # 提取 JSON body
            body_start = raw.find('{')
            if body_start < 0: s.close(); return
            body = raw[body_start:]

            try:
                req = json.loads(body)
            except json.JSONDecodeError:
                s.close(); return

            method = req.get('method', '')
            req_id = req.get('id', 1)

            if method == 'eth_call':
                params = req.get('params', [{}])
                to_addr = params[0].get('to', '') if params else ''
                call_data = params[0].get('data', '') if params else ''

                log(f"{tag} eth_call → 合约 {to_addr[:16]}... 方法 {call_data[:10]}")

                # 将 C2 IP 编码为 ABI 返回值
                # Solidity string 返回格式: offset(32B) + length(32B) + data(padded to 32B)
                c2_str = f"{self.c2_ip}:{self.c2_port}"
                c2_bytes = c2_str.encode()
                # ABI 编码
                offset = (32).to_bytes(32, 'big')
                length = len(c2_bytes).to_bytes(32, 'big')
                padded_data = c2_bytes + b'\x00' * (32 - len(c2_bytes) % 32) if len(c2_bytes) % 32 != 0 else c2_bytes
                result_hex = '0x' + (offset + length + padded_data).hex()

                response = json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": result_hex
                })
                log(f"{tag} 返回 C2 地址: {c2_str}")

            elif method == 'eth_blockNumber':
                response = json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": "0x211f13d"  # block 34734397
                })
                log(f"{tag} eth_blockNumber → 0x211f13d")

            elif method == 'net_version':
                response = json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": "56"  # BSC mainnet chain ID
                })
                log(f"{tag} net_version → 56 (BSC)")

            else:
                response = json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": "Method not found"}
                })

            # HTTP 响应
            http_resp = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(response)}\r\n"
                f"Connection: close\r\n"
                f"\r\n"
                f"{response}"
            )
            s.sendall(http_resp.encode())
            s.close()

        except Exception as e:
            log(f"{tag} 异常: {e}")
            try: s.close()
            except: pass

# ═══════════════ C2 Server (Smargaft C2, port 81) ═══════════════
class SmargaftC2Server:
    """模拟 Smargaft C2 服务端, 文本协议"""

    def __init__(self, host='127.0.0.1', port=81):
        self.host = host
        self.port = port
        self.running = False

    def start(self):
        self.running = True
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        raw.settimeout(1.0)
        raw.bind((self.host, self.port))
        raw.listen(5)
        self.ssock = RecSock(raw, self.host, self.host)
        log(f"[C2] 监听 {self.host}:{self.port}")

        while self.running:
            try:
                cs, addr = self.ssock.accept()
                threading.Thread(target=self._handle, args=(cs, addr), daemon=True).start()
            except socket.timeout: continue
            except OSError: break

    def stop(self): self.running = False

    def _handle(self, s, addr):
        tag = f"[C2|{addr[1]}]"
        try:
            s.settimeout(10.0)

            # ─── 接收 ready 包 ───
            ready = s.recv(256)
            if not ready:
                s.close(); return
            log(f"{tag} 收到 ready 包: {ready.hex()} ({len(ready)}B)")

            # 确认 ready
            s.send(b"ready_ack\n")
            log(f"{tag} 已回复 ready_ack")

            # ─── 接收 Bot 注册信息 ───
            time.sleep(0.3)
            try:
                reg_data = s.recv(4096)
                if reg_data:
                    reg_text = reg_data.decode('utf-8', errors='ignore').strip()
                    log(f"{tag} 注册信息: {reg_text[:80]}...")
            except socket.timeout:
                pass

            # 注册确认
            s.send(b"registered\n")
            time.sleep(0.5)

            # ─── 下发 DDoS 攻击指令序列 ───
            attack_cmds = [
                f"udph 43.249.192.173 17481 30\n",     # 来自博客的真实捕获示例
                f"tcph 192.168.1.100 80 45\n",
                f"syn 10.0.0.50 443 60\n",
                f"http http://172.16.0.10:8080/ 30\n",
                f"hex 203.0.113.50 53 20\n",
            ]
            for cmd in attack_cmds:
                s.send(cmd.encode())
                cmd_name = cmd.split()[0]
                target = cmd.split()[1] if len(cmd.split()) > 1 else ''
                log(f"{tag} ⚡ 下发: {cmd_name} → {target}")
                time.sleep(0.4)

            # 发送 shell 命令
            time.sleep(0.3)
            shell_cmds = [
                "shell cat /proc/cpuinfo | head -5\n",
                "shell uname -a\n",
            ]
            for cmd in shell_cmds:
                s.send(cmd.encode())
                log(f"{tag} 💻 下发: {cmd.strip()}")
                time.sleep(0.3)

            # 发送 SOCKS5 代理启动命令
            time.sleep(0.3)
            s.send(b"socks5 1080\n")
            log(f"{tag} 🔌 下发: socks5 1080")

            # 等待一些心跳
            time.sleep(0.5)
            hb_count = 0
            while self.running and hb_count < 3:
                try:
                    data = s.recv(4096)
                    if not data: break
                    text = data.decode('utf-8', errors='ignore').strip()
                    if text:
                        log(f"{tag} ← Bot: {text[:60]}")
                        hb_count += 1
                except socket.timeout:
                    hb_count += 1

            # 停止攻击
            s.send(b"stop\n")
            log(f"{tag} ■ 下发: stop")
            time.sleep(0.3)

            log(f"{tag} 会话结束")
            s.close()

        except Exception as e:
            log(f"{tag} 异常: {e}")
            try: s.close()
            except: pass

# ═══════════════ Smargaft Bot Client ═══════════════
class SmargaftBotClient:
    """模拟 Smargaft Bot 客户端"""

    def __init__(self, host='127.0.0.1', rpc_port=8545, c2_port=81, bid=0):
        self.host = host
        self.rpc_port = rpc_port
        self.c2_port = c2_port
        self.bid = bid

    def run(self):
        tag = f"[Bot#{self.bid}]"
        try:
            # ═══ 阶段 1: EtherHiding — 通过 BSC 智能合约获取 C2 ═══
            log(f"{tag} ═══ 阶段1: EtherHiding (BSC 智能合约查询) ═══")
            c2_addr = self._query_bsc_for_c2(tag)
            if not c2_addr:
                log(f"{tag} 无法获取 C2, 使用后备地址")
                c2_addr = (self.host, self.c2_port)

            # ═══ 阶段 2: 连接 C2 并注册 ═══
            log(f"{tag} ═══ 阶段2: C2 通信 (端口 {c2_addr[1]}) ═══")
            self._communicate_with_c2(c2_addr, tag)

        except Exception as e:
            log(f"{tag} 错误: {e}")
            traceback.print_exc()

    def _query_bsc_for_c2(self, tag):
        """
        阶段 1: 模拟 EtherHiding
        Bot 通过 JSON-RPC 向 BSC RPC 节点发送 eth_call 请求
        调用合约方法 0xd7ec3ad7 获取 C2 IP
        """
        # 随机选择一个 RPC 节点 (实际中有 14 个)
        rpc_url = random.choice(BSC_RPC_NODES)
        log(f"{tag} 选择 RPC 节点: {rpc_url}")

        # 先查询 net_version 确认是 BSC 网络
        net_version_req = json.dumps({
            "jsonrpc": "2.0",
            "method": "net_version",
            "params": [],
            "id": 1
        })
        resp = self._send_jsonrpc(net_version_req, tag, "net_version")

        # 查询最新区块号
        block_req = json.dumps({
            "jsonrpc": "2.0",
            "method": "eth_blockNumber",
            "params": [],
            "id": 2
        })
        self._send_jsonrpc(block_req, tag, "eth_blockNumber")

        # 核心: eth_call 调用智能合约获取 C2
        eth_call_req = json.dumps({
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{
                "to": CONTRACT_ADDR,
                "data": GET_C2_METHOD
            }, "latest"],
            "id": 3
        })
        c2_resp = self._send_jsonrpc(eth_call_req, tag, "eth_call (获取C2)")

        if c2_resp:
            try:
                result = json.loads(c2_resp)
                hex_data = result.get('result', '')
                if hex_data and hex_data != '0x':
                    raw = bytes.fromhex(hex_data[2:])
                    # ABI 解码: 跳过 offset(32B) + length(32B), 读取数据
                    if len(raw) >= 64:
                        str_len = int.from_bytes(raw[32:64], 'big')
                        c2_str = raw[64:64+str_len].decode('utf-8', errors='ignore')
                        log(f"{tag} ✓ 从智能合约获取 C2: {c2_str}")
                        if ':' in c2_str:
                            ip, port = c2_str.split(':')
                            return (ip, int(port))
                        return (c2_str, self.c2_port)
            except Exception as e:
                log(f"{tag} C2 解析失败: {e}")

        return (self.host, self.c2_port)

    def _send_jsonrpc(self, body, tag, desc):
        """发送 JSON-RPC 请求到模拟 BSC RPC 节点"""
        try:
            s = RecSock(socket.socket(socket.AF_INET, socket.SOCK_STREAM),
                        '127.0.0.1', self.host)
            s.settimeout(5.0)
            s.connect((self.host, self.rpc_port))

            http_req = (
                f"POST / HTTP/1.1\r\n"
                f"Host: bsc-dataseed.binance.org\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"User-Agent: Smargaft/1.0\r\n"
                f"Connection: close\r\n"
                f"\r\n"
                f"{body}"
            )
            s.sendall(http_req.encode())
            log(f"{tag} → {desc}")

            resp_data = b''
            while True:
                try:
                    chunk = s.recv(4096)
                    if not chunk: break
                    resp_data += chunk
                except socket.timeout:
                    break

            s.close()

            if resp_data:
                resp_text = resp_data.decode('utf-8', errors='ignore')
                body_start = resp_text.find('{')
                if body_start >= 0:
                    json_body = resp_text[body_start:]
                    log(f"{tag} ← {desc} 响应 OK")
                    return json_body

        except Exception as e:
            log(f"{tag} RPC 请求失败: {e}")
        return None

    def _communicate_with_c2(self, c2_addr, tag):
        """
        阶段 2-3: 连接 C2, 发送 ready 包, 接收指令
        """
        s = RecSock(socket.socket(socket.AF_INET, socket.SOCK_STREAM),
                    '127.0.0.1', c2_addr[0])
        s.settimeout(10.0)

        try:
            s.connect(c2_addr)
            log(f"{tag} 已连接 C2 {c2_addr[0]}:{c2_addr[1]}")

            # ─── 发送 5 字节 ready 包 ───
            s.send(READY_PACKET)
            log(f"{tag} → ready 包 ({READY_PACKET.hex()})")

            # 等待 ready_ack
            ack = s.recv(256)
            if ack:
                log(f"{tag} ← {ack.decode('utf-8', errors='ignore').strip()}")

            # ─── 发送注册信息 (文本格式) ───
            bot_info = (
                f"BOT_REG "
                f"arch={random.choice(['arm','mips','x86_64'])} "
                f"ip=192.168.{random.randint(1,254)}.{random.randint(2,254)} "
                f"mac={':'.join(f'{b:02x}' for b in os.urandom(6))} "
                f"os=Linux_4.14.0 "
                f"cpu={random.choice(['ARMv7','MIPS32','x86_64'])} "
                f"mem={random.choice(['128','256','512'])}MB "
                f"ver=smargaft_2.1\n"
            )
            s.send(bot_info.encode())
            log(f"{tag} → 注册信息")

            # 等待注册确认
            try:
                reg_ack = s.recv(256)
                if reg_ack:
                    log(f"{tag} ← {reg_ack.decode('utf-8', errors='ignore').strip()}")
            except socket.timeout:
                pass

            # ─── 接收并处理指令 ───
            cmd_count = 0
            while cmd_count < 15:
                try:
                    s.settimeout(3.0)
                    data = s.recv(4096)
                    if not data: break

                    text = data.decode('utf-8', errors='ignore')
                    lines = text.strip().split('\n')

                    for line in lines:
                        line = line.strip()
                        if not line: continue
                        cmd_count += 1
                        parts = line.split()
                        cmd = parts[0]

                        desc = COMMANDS.get(cmd, 'unknown')
                        if cmd in ('udph','tcph','udp','tcp','std','syn','ack','hex','vse'):
                            target = parts[1] if len(parts) > 1 else '?'
                            port = parts[2] if len(parts) > 2 else '?'
                            dur = parts[3] if len(parts) > 3 else '?'
                            log(f"{tag} ⚡ {cmd} ({desc}): {target}:{port} {dur}s")
                            # 发送攻击确认
                            s.send(f"ATK_START {cmd} {target}\n".encode())
                        elif cmd == 'http' or cmd == 'httpx':
                            url = parts[1] if len(parts) > 1 else '?'
                            log(f"{tag} ⚡ {cmd} ({desc}): {url}")
                            s.send(f"ATK_START {cmd} {url}\n".encode())
                        elif cmd == 'shell':
                            shell_cmd = ' '.join(parts[1:])
                            log(f"{tag} 💻 shell: {shell_cmd}")
                            s.send(f"SHELL_OK: output_placeholder\n".encode())
                        elif cmd == 'socks5':
                            proxy_port = parts[1] if len(parts) > 1 else '1080'
                            log(f"{tag} 🔌 socks5 代理启动, 端口 {proxy_port}")
                            s.send(f"SOCKS5_OK {proxy_port}\n".encode())
                        elif cmd == 'stop':
                            log(f"{tag} ■ 收到 stop 指令, 停止所有攻击")
                            s.send(b"ATK_STOPPED\n")
                        elif cmd == 'update':
                            log(f"{tag} 📥 收到 update 指令")
                            s.send(b"UPDATE_OK\n")
                        else:
                            log(f"{tag} ← 未知: {line[:50]}")

                except socket.timeout:
                    # 发送心跳
                    heartbeat = f"PING {int(time.time())}\n"
                    s.send(heartbeat.encode())
                    log(f"{tag} ♥ 心跳")

            log(f"{tag} ✓ 通信结束")
            s.close()

        except Exception as e:
            log(f"{tag} C2 通信异常: {e}")
            try: s.close()
            except: pass

# ═══════════════ Main ═══════════════
def main(host='127.0.0.1', rpc_port=8545, c2_port=81,
         output='smargaft_c2.pcap', bots=2):
    print(f"""
{'='*64}
  Smargaft C2 流量模拟器 — 安全研究 / 召回测试
{'='*64}
  技术: EtherHiding (BSC 智能合约托管 C2)
  合约: {CONTRACT_ADDR}
  方法: eth_call {GET_C2_METHOD} (免费, 无痕)

  RPC 端口: {rpc_port}  |  C2 端口: {c2_port}
  Bots: {bots}  |  输出: {output}

  流量阶段:
    1) BSC JSON-RPC (net_version/eth_blockNumber/eth_call)
    2) C2 连接 + 5B ready 包 + 注册 (文本协议)
    3) DDoS 指令 (udph/tcph/syn/http/hex/...)
    4) Shell 命令 + SOCKS5 代理
{'='*64}
""")

    # 启动 BSC RPC 模拟节点
    rpc_srv = BSCRPCServer(host, rpc_port, host, c2_port)
    threading.Thread(target=rpc_srv.start, daemon=True).start()
    time.sleep(0.3)

    # 启动 C2 Server
    c2_srv = SmargaftC2Server(host, c2_port)
    threading.Thread(target=c2_srv.start, daemon=True).start()
    time.sleep(0.3)

    # 启动 Bots
    ts = []
    for i in range(bots):
        time.sleep(0.3)
        log(f"\n{'─'*40}")
        log(f"启动 Bot #{i+1}")
        log(f"{'─'*40}")
        b = SmargaftBotClient(host, rpc_port, c2_port, i+1)
        t = threading.Thread(target=b.run, daemon=True); t.start(); ts.append(t)

    for t in ts: t.join(timeout=30)
    time.sleep(1)
    rpc_srv.stop(); c2_srv.stop()

    write_pcap(output)

    print(f"""
{'='*64}
  ✓ 完成! pcap: {output}  ({len(PCAP_RECORDS)} packets)
{'='*64}

  核心流量特征 (供召回知识库匹配):

  【阶段1 — EtherHiding】
    • HTTP POST → BSC RPC 节点
    • JSON-RPC 方法: eth_call / net_version / eth_blockNumber
    • 合约地址: {CONTRACT_ADDR}
    • 方法签名: {GET_C2_METHOD}
    • User-Agent: Smargaft/1.0
    • Content-Type: application/json

  【阶段2 — C2 连接】
    • TCP 端口 {c2_port}
    • 5 字节 ready 包: {READY_PACKET.hex()}
    • 文本注册: "BOT_REG arch=... ip=... mac=..."

  【阶段3 — 指令与攻击】
    • 纯文本指令: "udph <ip> <port> <dur>"
    • 15 种命令: udph/tcph/udp/tcp/std/http/httpx/hex/
      vse/ack/syn/shell/socks5/stop/update
    • 攻击确认: "ATK_START <cmd> <target>"
    • 心跳: "PING <timestamp>"
{'='*64}
""")

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='Smargaft C2 流量模拟器')
    pa.add_argument('--host', default='127.0.0.1')
    pa.add_argument('--rpc-port', type=int, default=8545, help='BSC RPC 模拟端口')
    pa.add_argument('--c2-port', type=int, default=8181, help='C2 端口 (默认 8181 避免权限)')
    pa.add_argument('--output', default='smargaft_c2.pcap')
    pa.add_argument('--bots', type=int, default=2)
    a = pa.parse_args()
    main(a.host, a.rpc_port, a.c2_port, a.output, a.bots)
