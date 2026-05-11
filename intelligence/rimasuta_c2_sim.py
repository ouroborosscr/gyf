#!/usr/bin/env python3
"""
Rimasuta C2 流量模拟器 (研究用途)
===================================
参考: https://blog.xlab.qianxin.com/rimasuta-new-variant-switches-to-chacha20-encryption-cn/

模拟 Rimasuta 僵尸网络 C2 通信协议, 用于生成 pcap 包进行流量防御智能体的召回测试。

协议特征 (来自 XLab 博客):
  1. SOCKS5 握手 — 模拟通过 Tor Proxy 连接 C2 隐藏服务
  2. 密钥协商   — Bot 发 8B 随机数, 经 fasthash 派生 session_header
  3. Bot 注册   — ChaCha20 加密用户信息 (IP/CPU/MAC/网速)
  4. 心跳保活   — 加密心跳包, C2 回复 ACK
  5. 指令下发   — 加密的 DDoS 攻击指令 (UDP/TCP/HTTP Flood)

数据包格式: head(2B) + hash(4B) + content(NB)
"""

import argparse, hashlib, os, socket, struct, sys, threading, time, traceback, random
from datetime import datetime

try:
    from Crypto.Cipher import ChaCha20
except ImportError:
    print("[!] pip install pycryptodome --break-system-packages"); sys.exit(1)

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
        PCAP_RECORDS.append((time.time(), src_ip, src_port, dst_ip, dst_port, data, seq, ack, flags))
        SEQ_COUNTERS[key] = seq + max(len(data), 1)

def write_pcap(filename):
    with open(filename, 'wb') as f:
        writer = dpkt.pcap.Writer(f, linktype=dpkt.pcap.DLT_RAW)
        for rec in PCAP_RECORDS:
            ts, src_ip, src_port, dst_ip, dst_port, data, seq, ack, flags = rec
            tf = 0
            if 'S' in flags: tf |= dpkt.tcp.TH_SYN
            if 'A' in flags: tf |= dpkt.tcp.TH_ACK
            if 'P' in flags: tf |= dpkt.tcp.TH_PUSH
            if 'F' in flags: tf |= dpkt.tcp.TH_FIN
            tcp_pkt = dpkt.tcp.TCP(sport=src_port, dport=dst_port, seq=seq, ack=ack, off=5, flags=tf, win=65535, data=data)
            ip_pkt = dpkt.ip.IP(src=socket.inet_aton(src_ip), dst=socket.inet_aton(dst_ip), p=dpkt.ip.IP_PROTO_TCP, ttl=64, data=tcp_pkt)
            ip_pkt.len = len(ip_pkt)
            writer.writepkt(bytes(ip_pkt), ts)
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
        r = self._s.send(data)
        record_packet(self._li, self._lp, self._ri, self._rp, data, 'PA')
        return r

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
CHACHA20_SEED = bytes.fromhex("BEBA4948")
SOCKS5_VER = 0x05
CMD_HB, CMD_UDP, CMD_TCP, CMD_HTTP, CMD_STOP, CMD_UPD, CMD_SH = 1,2,3,4,5,6,7
TOR_DOMAINS = [b"rimasuta2c3d4e5f.onion", b"ptea7g8h9i0j1k2l.onion", b"chacha3m4n5o6p7q.onion"]

# ═══════════════ fasthash ═══════════════
def fasthash64(data, seed=0):
    m = 0x880355f21e6d1965; M = 0xFFFFFFFFFFFFFFFF
    h = (seed ^ (len(data) * m)) & M; pos = 0
    while pos + 8 <= len(data):
        v = struct.unpack_from('<Q', data, pos)[0]
        v = (v * m) & M; v ^= (v >> 23); v = (v * m) & M
        h ^= v; h = (h * m) & M; pos += 8
    rem = len(data) - pos
    if rem > 0:
        v = 0
        for i in range(rem): v |= data[pos + i] << (i * 8)
        h ^= v; h = (h * m) & M
    h ^= (h >> 23); h = (h * m) & M; h ^= (h >> 47)
    return h & M

def fasthash32(data, seed=0):
    h = fasthash64(data, seed); return ((h >> 32) ^ h) & 0xFFFFFFFF

# ═══════════════ ChaCha20 ═══════════════
def cc20_enc(pt, key, nonce):
    if len(nonce) > 8: nonce = nonce[:8]
    return ChaCha20.new(key=key, nonce=nonce).encrypt(pt)

cc20_dec = cc20_enc

# ═══════════════ 协议构造 ═══════════════
def build_pkt(sh, content):
    return sh + struct.pack('<I', fasthash32(content)) + content

def parse_pkt(data):
    if len(data) < 6: return None, None, None
    return data[:2], struct.unpack('<I', data[2:6])[0], data[6:]

def derive_key(rb, sh): return hashlib.sha256(rb + sh + CHACHA20_SEED).digest()
def derive_nonce(rb): return hashlib.md5(rb).digest()[:8]

def build_bot_info():
    fields = {'ip': '192.168.1.' + str(random.randint(2,254)), 'arch': 'arm',
              'mac': ':'.join(f'{b:02x}' for b in os.urandom(6)),
              'speed': '100', 'os': 'Linux 4.14.0',
              'host': f'dvr-{os.urandom(3).hex()}',
              'up': str(int(time.time()) % 100000)}
    parts = []
    for v in fields.values():
        d = v.encode()
        parts.append(struct.pack('<I', fasthash32(d)) + struct.pack('<H', len(d)) + d)
    return b''.join(parts)

# ═══════════════ 日志 ═══════════════
_LL = threading.Lock()
def log(msg):
    with _LL: print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")

# ═══════════════ C2 Server ═══════════════
class C2Server:
    def __init__(self, host='127.0.0.1', port=9050):
        self.host, self.port, self.running = host, port, False

    def start(self):
        self.running = True
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        raw.settimeout(1.0); raw.bind((self.host, self.port)); raw.listen(5)
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
            # ─── 阶段1: SOCKS5 ───
            g = s.recv(256)
            if not g or g[0] != SOCKS5_VER: s.close(); return
            s.send(bytes([SOCKS5_VER, 0x00]))
            cr = s.recv(256)
            if not cr or len(cr) < 4: s.close(); return
            s.send(bytes([SOCKS5_VER,0,0,1, 0,0,0,0, 0,0]))
            log(f"{tag} SOCKS5 ✓")

            # ─── 阶段2: 密钥协商 ───
            nd = s.recv(256)
            if not nd or len(nd) < 14: s.close(); return
            sh, _, br = parse_pkt(nd)
            c2r = os.urandom(8); c2h = fasthash32(c2r)
            s.send(sh + struct.pack('<H', len(c2r)) + struct.pack('<I', c2h) + c2r)
            ck = derive_key(br + c2r, sh); cn = derive_nonce(br + c2r)
            log(f"{tag} 密钥协商 ✓ session={sh.hex()}")

            # ─── 阶段3: 注册 ───
            rd = s.recv(4096)
            if rd:
                _, _, ec = parse_pkt(rd)
                if ec:
                    try: cc20_dec(ec, ck, cn); log(f"{tag} 注册解密 ✓ ({len(ec)}B)")
                    except: log(f"{tag} 注册数据接收 ({len(ec)}B)")
                s.send(build_pkt(sh, b'\x00\x01'))
                log(f"{tag} 注册 ACK ✓")

            # ─── 阶段4: 心跳+指令 ───
            hbc, sent = 0, False
            while self.running and hbc < 5:
                try:
                    d = s.recv(4096)
                    if not d: break
                    _, _, ct = parse_pkt(d)
                    if ct:
                        try:
                            dc = cc20_dec(ct, ck, cn)
                            if dc and dc[0] == CMD_HB:
                                hbc += 1; log(f"{tag} ♥ 心跳 #{hbc}")
                                s.send(build_pkt(sh, cc20_enc(bytes([CMD_HB, 0]), ck, cn)))
                        except: hbc += 1
                    if hbc == 3 and not sent:
                        sent = True; self._cmds(s, sh, ck, cn, tag)
                except socket.timeout: continue
                except: break
            log(f"{tag} 会话结束"); s.close()
        except Exception as e:
            log(f"{tag} 异常: {e}"); traceback.print_exc()
            try: s.close()
            except: pass

    def _cmds(self, s, sh, k, n, tag):
        for ct, tip, tp, dur, desc in [
            (CMD_UDP, "192.168.1.100",80,120, "UDP_Flood"),
            (CMD_TCP, "10.0.0.50",443,60, "TCP_SYN_Flood"),
            (CMD_HTTP,"172.16.0.10",8080,90, "HTTP_Flood"),
        ]:
            p = struct.pack('B', ct) + socket.inet_aton(tip) + struct.pack('>H', tp) + struct.pack('<I', dur) + b'\x01'
            s.send(build_pkt(sh, cc20_enc(p, k, n)))
            log(f"{tag} ⚡ {desc} → {tip}:{tp} ({dur}s)")
            time.sleep(0.2)
        time.sleep(0.2)
        s.send(build_pkt(sh, cc20_enc(bytes([CMD_STOP]), k, n)))
        log(f"{tag} ■ STOP")

# ═══════════════ Bot Client ═══════════════
class BotClient:
    def __init__(self, host='127.0.0.1', port=9050, bid=0):
        self.host, self.port, self.bid = host, port, bid
        self.sh = self.ck = self.cn = None

    def run(self):
        self.s = RecSock(socket.socket(socket.AF_INET, socket.SOCK_STREAM), '127.0.0.1', self.host)
        self.s.settimeout(10.0)
        tag = f"[Bot#{self.bid}]"
        try:
            self.s.connect((self.host, self.port))
            # ─── SOCKS5 ───
            self.s.send(bytes([SOCKS5_VER, 0x01, 0x00]))
            if self.s.recv(2) != bytes([SOCKS5_VER, 0x00]): raise Exception("SOCKS5 fail")
            td = random.choice(TOR_DOMAINS)
            self.s.send(bytes([SOCKS5_VER,1,0,3, len(td)]) + td + struct.pack('>H', 443))
            r = self.s.recv(256)
            if not r or r[1] != 0: raise Exception("SOCKS5 connect fail")
            log(f"{tag} SOCKS5 ✓ → {td.decode()}")

            # ─── 密钥协商 ───
            br = os.urandom(8)
            self.sh = struct.pack('<H', fasthash64(br) & 0xFFFF)
            self.s.send(build_pkt(self.sh, br))
            r = self.s.recv(256)
            if not r or len(r) < 8: raise Exception("Nego fail")
            cl = struct.unpack('<H', r[2:4])[0]; c2r = r[8:8+cl]
            cb = br + c2r; self.ck = derive_key(cb, self.sh); self.cn = derive_nonce(cb)
            log(f"{tag} 密钥协商 ✓")

            # ─── 注册 ───
            bi = build_bot_info()
            self.s.send(build_pkt(self.sh, cc20_enc(bi, self.ck, self.cn)))
            a = self.s.recv(256)
            if a: log(f"{tag} 注册 ✓")

            # ─── 心跳 ───
            for i in range(5):
                time.sleep(0.8)
                hb = bytes([CMD_HB]) + struct.pack('<I', int(time.time()))
                self.s.send(build_pkt(self.sh, cc20_enc(hb, self.ck, self.cn)))
                log(f"{tag} ♥ #{i+1}")
                try:
                    self.s.settimeout(2.0)
                    d = self.s.recv(4096)
                    if d: self._proc(d, tag)
                except socket.timeout: pass
            log(f"{tag} 完成")
        except Exception as e:
            log(f"{tag} 错误: {e}"); traceback.print_exc()
        finally:
            self.s.close()

    def _proc(self, data, tag):
        _, _, ct = parse_pkt(data)
        if not ct: return
        try:
            dc = cc20_dec(ct, self.ck, self.cn)
            if not dc: return
            nm = {CMD_HB:"ACK", CMD_UDP:"DDoS_UDP", CMD_TCP:"DDoS_TCP", CMD_HTTP:"DDoS_HTTP", CMD_STOP:"STOP"}
            c = dc[0]; n = nm.get(c, f"cmd_{c:#x}")
            if c in (CMD_UDP, CMD_TCP, CMD_HTTP) and len(dc) >= 8:
                t = socket.inet_ntoa(dc[1:5]); p = struct.unpack('>H', dc[5:7])[0]
                d = struct.unpack('<I', dc[7:11])[0] if len(dc) >= 11 else 0
                log(f"{tag} ⚡ {n}: {t}:{p} ({d}s)")
            else:
                log(f"{tag} ← {n}")
        except: pass

# ═══════════════ Main ═══════════════
def main(host='127.0.0.1', port=9050, output='rimasuta_c2.pcap', bots=2):
    print(f"""
{'='*60}
  Rimasuta C2 流量模拟器 — 安全研究 / 召回测试
{'='*60}
  协议: SOCKS5 + fasthash + ChaCha20
  端口: {port}  |  Bots: {bots}  |  输出: {output}
{'='*60}
""")
    srv = C2Server(host, port)
    threading.Thread(target=srv.start, daemon=True).start()
    time.sleep(0.5)
    ts = []
    for i in range(bots):
        time.sleep(0.3)
        b = BotClient(host, port, i+1)
        t = threading.Thread(target=b.run, daemon=True); t.start(); ts.append(t)
    for t in ts: t.join(timeout=30)
    time.sleep(1); srv.stop(); write_pcap(output)
    print(f"""
{'='*60}
  ✓ 完成! pcap: {output}  ({len(PCAP_RECORDS)} packets)
{'='*60}
  核心流量特征 (供召回知识库匹配):
    • 端口 {port} (Tor SOCKS5)
    • SOCKS5: 05 01 00 → 05 00 / .onion 域名
    • 协议头: session(2B) + fasthash32(4B) + payload
    • ChaCha20 加密载荷 (高熵)
    • 心跳: ~0.8s 周期, cmd=0x01
    • DDoS 指令: cmd + target_ip(4B) + port(2B) + duration(4B)
{'='*60}
""")

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='Rimasuta C2 流量模拟器')
    pa.add_argument('--host', default='127.0.0.1')
    pa.add_argument('--port', type=int, default=9050)
    pa.add_argument('--output', default='rimasuta_c2.pcap')
    pa.add_argument('--bots', type=int, default=2)
    a = pa.parse_args()
    main(a.host, a.port, a.output, a.bots)
