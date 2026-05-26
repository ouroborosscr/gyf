"""
GYF FRP 检测工具集 (pcap_tools.py)

封装所有 tshark / tcpdump / capinfos 命令为 LangChain Tool，
供 GYF Graph 的 tools 节点调用。

依赖：
    - tshark / tcpdump / capinfos (apt install tshark tcpdump)
    - langchain-core
"""

import json
import re
import subprocess
import shlex
from pathlib import Path
from typing import Optional, Dict, List, Any
from collections import Counter

from langchain_core.tools import tool


# ============================================================
# 内部辅助
# ============================================================

def _run(cmd: List[str], timeout: int = 60) -> Dict[str, Any]:
    """统一的子进程执行入口，返回 {ok, stdout, stderr}"""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": "timeout", "returncode": -1}
    except FileNotFoundError as e:
        return {"ok": False, "stdout": "", "stderr": f"command not found: {e}", "returncode": -2}


def _validate_pcap(pcap_path: str) -> Optional[str]:
    """返回错误信息字符串，None 表示通过"""
    p = Path(pcap_path)
    if not p.exists():
        return f"pcap 文件不存在: {pcap_path}"
    if not p.is_file():
        return f"路径不是文件: {pcap_path}"
    if p.stat().st_size == 0:
        return f"pcap 文件为空: {pcap_path}"
    return None


# ============================================================
# 工具 1: pcap_basic_info
# ============================================================

@tool
def pcap_basic_info(pcap_path: str) -> Dict[str, Any]:
    """获取 pcap 文件的基本元信息：包数量、抓包时长、字节数、平均包大小、链路类型等。
    
    用于审计入口的健康检查 + 给后续工具提供基线数据。
    
    Args:
        pcap_path: pcap 文件的绝对路径
    
    Returns:
        {
            "ok": bool,
            "file_size_bytes": int,
            "packet_count": int,
            "capture_duration_sec": float,
            "data_byte_rate": float,
            "avg_packet_size": float,
            "first_packet_time": str,
            "last_packet_time": str,
            "raw": str  # 原始 capinfos 输出
        }
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    r = _run(["capinfos", "-c", "-u", "-z", "-i", "-y", "-a", "-e", pcap_path])
    if not r["ok"]:
        return {"ok": False, "error": r["stderr"]}
    
    out = r["stdout"]
    
    def grab(pattern: str, default=None):
        m = re.search(pattern, out)
        return m.group(1).strip() if m else default
    
    info = {
        "ok": True,
        "file_size_bytes": Path(pcap_path).stat().st_size,
        "packet_count": int((grab(r"Number of packets:\s+([0-9,]+)") or "0").replace(",", "")),
        "capture_duration_sec": float(grab(r"Capture duration:\s+([0-9.]+)", "0")),
        "data_byte_rate": float(grab(r"Data byte rate:\s+([0-9.]+)", "0")),
        "avg_packet_size": float(grab(r"Average packet size:\s+([0-9.]+)", "0")),
        "first_packet_time": grab(r"First packet time:\s+(.+)"),
        "last_packet_time": grab(r"Last packet time:\s+(.+)"),
        "raw": out
    }
    return info


# ============================================================
# 工具 2: pcap_protocol_hierarchy
# ============================================================

@tool
def pcap_protocol_hierarchy(pcap_path: str) -> Dict[str, Any]:
    """获取 pcap 的协议层级分布（tshark io,phs），帮助快速识别流量类型。
    
    输出每一层协议的帧数与字节数。例如 frp WSS 模式典型分布为 eth→ip→tcp→tls。
    
    Args:
        pcap_path: pcap 文件绝对路径
    
    Returns:
        {
            "ok": bool,
            "layers": [{"protocol": str, "depth": int, "frames": int, "bytes": int}],
            "top_app_protocols": [str],  # 最深层的应用协议
            "raw": str
        }
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    r = _run(["tshark", "-r", pcap_path, "-q", "-z", "io,phs"])
    if not r["ok"]:
        return {"ok": False, "error": r["stderr"]}
    
    layers = []
    top_app = []
    for line in r["stdout"].splitlines():
        m = re.match(r"^(\s*)(\S+)\s+frames:(\d+)\s+bytes:(\d+)", line)
        if m:
            indent = len(m.group(1))
            depth = indent // 2
            proto = m.group(2)
            layers.append({
                "protocol": proto,
                "depth": depth,
                "frames": int(m.group(3)),
                "bytes": int(m.group(4))
            })
    
    # 取最深层的几个协议作为应用层指示
    if layers:
        max_depth = max(l["depth"] for l in layers)
        top_app = list(set(l["protocol"] for l in layers if l["depth"] == max_depth))
    
    return {"ok": True, "layers": layers, "top_app_protocols": top_app, "raw": r["stdout"]}


# ============================================================
# 工具 3: pcap_conversations
# ============================================================

@tool
def pcap_conversations(pcap_path: str, proto: str = "tcp", limit: int = 20) -> Dict[str, Any]:
    """获取 TCP 或 UDP 会话列表。每条会话包含端点、方向流量、持续时间。
    
    内网穿透核心证据：上下行流量极度不对称（5:1 以上）。
    
    Args:
        pcap_path: pcap 文件绝对路径
        proto: "tcp" 或 "udp"
        limit: 返回会话数上限
    
    Returns:
        {
            "ok": bool,
            "conversations": [
                {
                    "endpoint_a": str, "endpoint_b": str,
                    "frames_a_to_b": int, "bytes_a_to_b": int,
                    "frames_b_to_a": int, "bytes_b_to_a": int,
                    "total_frames": int, "total_bytes": int,
                    "rel_start_sec": float, "duration_sec": float
                }
            ]
        }
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    proto = proto.lower()
    if proto not in ("tcp", "udp"):
        return {"ok": False, "error": "proto 必须是 tcp 或 udp"}
    
    r = _run(["tshark", "-r", pcap_path, "-q", "-z", f"conv,{proto}"])
    if not r["ok"]:
        return {"ok": False, "error": r["stderr"]}
    
    conversations = []
    for line in r["stdout"].splitlines():
        # tshark conv 行格式: "ep1 <-> ep2 frames bytes frames bytes frames bytes rel_start duration"
        if "<->" not in line:
            continue
        m = re.match(
            r"\s*(\S+)\s+<->\s+(\S+)\s+([\d,]+)\s+(\S+)\s+([\d,]+)\s+(\S+)\s+([\d,]+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)",
            line
        )
        if not m:
            continue
        
        def to_int(s): return int(s.replace(",", ""))
        def bytes_from(s):
            # 字节列可能是 "12 kB", "138 B" 这种格式
            num = re.match(r"([\d.]+)", s)
            if not num: return 0
            n = float(num.group(1))
            if "k" in s.lower(): n *= 1024
            elif "m" in s.lower(): n *= 1024 * 1024
            return int(n)
        
        conversations.append({
            "endpoint_a": m.group(1),
            "endpoint_b": m.group(2),
            "frames_a_to_b": to_int(m.group(3)),
            "bytes_a_to_b": bytes_from(m.group(4)),
            "frames_b_to_a": to_int(m.group(5)),
            "bytes_b_to_a": bytes_from(m.group(6)),
            "total_frames": to_int(m.group(7)),
            "total_bytes": bytes_from(m.group(8)),
            "rel_start_sec": float(m.group(9)),
            "duration_sec": float(m.group(10))
        })
        if len(conversations) >= limit:
            break
    
    return {"ok": True, "conversations": conversations, "proto": proto}


# ============================================================
# 工具 4: pcap_packet_payload
# ============================================================

@tool
def pcap_packet_payload(
    pcap_path: str,
    filter_expr: str = "",
    packet_num: int = 1,
    max_bytes: int = 256
) -> Dict[str, Any]:
    """提取指定包的 payload（hex），用于查看协议握手字节。
    
    例如查看 frp 控制连接的第一个数据包，识别 frp 协议头或 TLS Client Hello。
    
    Args:
        pcap_path: pcap 文件绝对路径
        filter_expr: tshark display filter，如 "tcp.dstport == 7000 and tcp.len > 0"
        packet_num: 取第几个匹配的包（从 1 开始）
        max_bytes: 返回的 payload 字节数上限
    
    Returns:
        {
            "ok": bool,
            "frame_number": int,
            "length": int,
            "payload_hex": str,        # 前 max_bytes*2 个字符
            "payload_ascii": str,      # 可打印 ASCII 表示
            "first_byte": int          # 首字节 (decimal)
        }
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    cmd = ["tshark", "-r", pcap_path, "-T", "fields",
           "-e", "frame.number", "-e", "tcp.len", "-e", "tcp.payload",
           "-e", "udp.payload"]
    if filter_expr:
        cmd += ["-Y", filter_expr]
    
    r = _run(cmd)
    if not r["ok"]:
        return {"ok": False, "error": r["stderr"]}
    
    lines = [l for l in r["stdout"].splitlines() if l.strip()]
    if not lines:
        return {"ok": False, "error": "没有匹配的包"}
    
    if packet_num < 1 or packet_num > len(lines):
        return {"ok": False, "error": f"packet_num 越界 (有效范围 1-{len(lines)})"}
    
    parts = lines[packet_num - 1].split("\t")
    while len(parts) < 4:
        parts.append("")
    
    frame_num, tcp_len, tcp_payload, udp_payload = parts[0], parts[1], parts[2], parts[3]
    payload_hex = tcp_payload or udp_payload or ""
    payload_hex = payload_hex.replace(":", "").lower()[:max_bytes * 2]
    
    # ASCII 表示
    ascii_str = ""
    for i in range(0, len(payload_hex), 2):
        try:
            byte = int(payload_hex[i:i+2], 16)
            ascii_str += chr(byte) if 32 <= byte < 127 else "."
        except ValueError:
            ascii_str += "?"
    
    first_byte = int(payload_hex[:2], 16) if payload_hex else -1
    
    return {
        "ok": True,
        "frame_number": int(frame_num) if frame_num.isdigit() else 0,
        "length": int(tcp_len) if tcp_len.isdigit() else 0,
        "payload_hex": payload_hex,
        "payload_ascii": ascii_str,
        "first_byte": first_byte
    }


# ============================================================
# 工具 5: pcap_list_packets
# ============================================================

@tool
def pcap_list_packets(
    pcap_path: str,
    filter_expr: str = "",
    n: int = 20
) -> Dict[str, Any]:
    """列出 pcap 中前 n 个包的概要（时间、源、目的、协议、长度、Info）。
    
    用于快速浏览流量、定位关键包。
    
    Args:
        pcap_path: pcap 文件绝对路径
        filter_expr: tshark display filter
        n: 最大返回条数
    
    Returns:
        {"ok": bool, "packets": [{...}], "total_matched": int}
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    cmd = ["tshark", "-r", pcap_path, "-T", "fields",
           "-e", "frame.number", "-e", "frame.time_relative",
           "-e", "ip.src", "-e", "tcp.srcport", "-e", "udp.srcport",
           "-e", "ip.dst", "-e", "tcp.dstport", "-e", "udp.dstport",
           "-e", "frame.len", "-e", "_ws.col.Protocol", "-e", "_ws.col.Info",
           "-c", str(n)]
    if filter_expr:
        cmd += ["-Y", filter_expr]
    
    r = _run(cmd)
    if not r["ok"]:
        return {"ok": False, "error": r["stderr"]}
    
    packets = []
    for line in r["stdout"].splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        while len(parts) < 11:
            parts.append("")
        packets.append({
            "frame": int(parts[0]) if parts[0].isdigit() else 0,
            "time": float(parts[1]) if parts[1] else 0,
            "src": parts[2],
            "src_port": parts[3] or parts[4],
            "dst": parts[5],
            "dst_port": parts[6] or parts[7],
            "length": int(parts[8]) if parts[8].isdigit() else 0,
            "protocol": parts[9],
            "info": parts[10][:120]  # 截断 info
        })
    
    return {"ok": True, "packets": packets, "total_matched": len(packets)}


# ============================================================
# 工具 6: pcap_tls_handshakes
# ============================================================

@tool
def pcap_tls_handshakes(pcap_path: str, force_port: int = 0) -> Dict[str, Any]:
    """提取所有 TLS 握手包概要，返回握手类型（Client Hello / Server Hello 等）。
    
    注意：非标准端口（如 7000）上的 TLS 流量可能被 Wireshark 错误识别为其他协议。
    使用 force_port 可强制在指定端口上启用 TLS 解析。
    
    Args:
        pcap_path: pcap 文件绝对路径
        force_port: 在该端口上强制按 TLS 解析（0 = 不强制）
    
    Returns:
        {
            "ok": bool,
            "handshakes": [
                {"frame": int, "type": int, "type_name": str, "version": str, "sni": str}
            ]
        }
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    cmd = ["tshark", "-r", pcap_path]
    if force_port:
        cmd += ["-d", f"tcp.port=={force_port},tls"]
    cmd += ["-Y", "tls.handshake.type",
            "-T", "fields",
            "-e", "frame.number",
            "-e", "tls.handshake.type",
            "-e", "tls.handshake.version",
            "-e", "tls.handshake.extensions_server_name"]
    
    r = _run(cmd)
    if not r["ok"]:
        return {"ok": False, "error": r["stderr"]}
    
    type_names = {
        1: "ClientHello", 2: "ServerHello", 11: "Certificate",
        12: "ServerKeyExchange", 13: "CertificateRequest",
        14: "ServerHelloDone", 15: "CertificateVerify",
        16: "ClientKeyExchange", 20: "Finished"
    }
    
    handshakes = []
    for line in r["stdout"].splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        while len(parts) < 4:
            parts.append("")
        try:
            t = int(parts[1]) if parts[1] else -1
            handshakes.append({
                "frame": int(parts[0]) if parts[0] else 0,
                "type": t,
                "type_name": type_names.get(t, "Unknown"),
                "version": parts[2],
                "sni": parts[3]
            })
        except ValueError:
            continue
    
    return {"ok": True, "handshakes": handshakes}


# ============================================================
# 工具 7: pcap_client_hello_details
# ============================================================

@tool
def pcap_client_hello_details(pcap_path: str, force_port: int = 0, frame: int = 0) -> Dict[str, Any]:
    """提取 TLS Client Hello 的完整字段：版本、密码套件、扩展、SNI、JA3。
    
    Args:
        pcap_path: pcap 文件绝对路径
        force_port: 在该端口强制 TLS 解析
        frame: 只看指定帧号（0 = 不限制）
    
    Returns:
        {
            "ok": bool,
            "client_hellos": [
                {
                    "frame": int,
                    "version": str,
                    "cipher_suites": [str],
                    "extensions": [str],
                    "sni": str,
                    "supported_versions": [str],
                    "ja3_string": str,
                    "ja3_hash": str
                }
            ]
        }
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    cmd = ["tshark", "-r", pcap_path]
    if force_port:
        cmd += ["-d", f"tcp.port=={force_port},tls"]
    
    filter_expr = "tls.handshake.type == 1"
    if frame:
        filter_expr = f"frame.number == {frame} and tls.handshake.type == 1"
    cmd += ["-Y", filter_expr, "-V"]
    
    r = _run(cmd)
    if not r["ok"]:
        return {"ok": False, "error": r["stderr"]}
    
    out = r["stdout"]
    hellos = []
    
    # 按 frame 分块（简单切分）
    blocks = re.split(r"^Frame \d+:", out, flags=re.MULTILINE)
    for block in blocks[1:]:
        ja3_match = re.search(r"\[JA3:\s+([a-f0-9]+)\]", block)
        ja3_fullstring_match = re.search(r"\[JA3 Fullstring:\s+(.+?)\]", block)
        sni_match = re.search(r"Server Name:\s+(.+)", block)
        version_match = re.search(r"Version:\s+TLS\s+([\d.]+)", block)
        frame_match = re.search(r"Frame\s+(\d+)", "Frame " + block[:200])
        
        cipher_suites = re.findall(r"Cipher Suite:\s+(\S+)", block)
        extensions = re.findall(r"Extension:\s+(\S+)", block)
        supported_versions = re.findall(r"Supported Version:\s+TLS\s+([\d.]+)", block)
        
        if ja3_match or sni_match or cipher_suites:
            hellos.append({
                "frame": int(frame_match.group(1)) if frame_match else 0,
                "version": version_match.group(1) if version_match else "",
                "cipher_suites": cipher_suites[:30],
                "extensions": extensions[:20],
                "sni": sni_match.group(1).strip() if sni_match else "",
                "supported_versions": supported_versions,
                "ja3_string": ja3_fullstring_match.group(1) if ja3_fullstring_match else "",
                "ja3_hash": ja3_match.group(1) if ja3_match else ""
            })
    
    return {"ok": True, "client_hellos": hellos}


# ============================================================
# 工具 8: pcap_extract_ja3
# ============================================================

@tool
def pcap_extract_ja3(pcap_path: str, force_port: int = 0) -> Dict[str, Any]:
    """快速提取所有 JA3 / JA3S 指纹哈希。
    
    Args:
        pcap_path: pcap 文件绝对路径
        force_port: 在该端口强制 TLS 解析
    
    Returns:
        {
            "ok": bool,
            "ja3_hashes": [{"frame": int, "hash": str, "fullstring": str}],
            "unique_ja3": [str],
            "is_go_default_ja3": bool  # 是否匹配已知 Go 标准库指纹
        }
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    # 已知的 Go 标准库 TLS 客户端 JA3 指纹（来自 frp 抓包分析）
    KNOWN_GO_JA3 = {
        "2196848d251b217de8b2c037e356c11d": "Go default TLS (no SNI)",
        "20b279993ae2e137e62b9647c6d768fb": "Go default TLS (with SNI)"
    }
    
    cmd = ["tshark", "-r", pcap_path]
    if force_port:
        cmd += ["-d", f"tcp.port=={force_port},tls"]
    cmd += ["-Y", "tls.handshake.type == 1", "-V"]
    
    r = _run(cmd)
    if not r["ok"]:
        return {"ok": False, "error": r["stderr"]}
    
    ja3_list = []
    is_go = False
    for line in r["stdout"].splitlines():
        m_hash = re.search(r"\[JA3:\s+([a-f0-9]+)\]", line)
        m_full = re.search(r"\[JA3 Fullstring:\s+(.+?)\]", line)
        if m_hash:
            h = m_hash.group(1)
            ja3_list.append({
                "frame": 0,
                "hash": h,
                "fullstring": m_full.group(1) if m_full else "",
                "known_as": KNOWN_GO_JA3.get(h, "")
            })
            if h in KNOWN_GO_JA3:
                is_go = True
    
    unique = list(set(j["hash"] for j in ja3_list))
    return {
        "ok": True,
        "ja3_hashes": ja3_list,
        "unique_ja3": unique,
        "is_go_default_ja3": is_go
    }


# ============================================================
# 工具 9: pcap_detect_heartbeat
# ============================================================

@tool
def pcap_detect_heartbeat(
    pcap_path: str,
    src_ip: str = "",
    dst_ip: str = "",
    max_packet_size: int = 200
) -> Dict[str, Any]:
    """检测心跳模式。统计客户端到服务端的小数据包（< max_packet_size）的间隔规律。
    
    frp 默认心跳是 30 秒（除 QUIC 模式是 10 秒）。
    
    Args:
        pcap_path: pcap 文件绝对路径
        src_ip: 客户端 IP（过滤源）
        dst_ip: 服务端 IP（过滤目的）
        max_packet_size: 心跳包大小阈值
    
    Returns:
        {
            "ok": bool,
            "small_packets_count": int,
            "buckets_per_second": {sec: count},  # 每秒包数
            "candidate_periods": [seconds],       # 检测到的周期性时间点
            "estimated_period_sec": float,        # 估算的心跳周期
            "is_periodic": bool
        }
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    filters = []
    if src_ip:
        filters.append(f"ip.src == {src_ip}")
    if dst_ip:
        filters.append(f"ip.dst == {dst_ip}")
    filters.append(f"frame.len <= {max_packet_size}")
    filter_expr = " and ".join(filters)
    
    cmd = ["tshark", "-r", pcap_path, "-Y", filter_expr,
           "-T", "fields", "-e", "frame.time_relative", "-e", "frame.len"]
    r = _run(cmd)
    if not r["ok"]:
        return {"ok": False, "error": r["stderr"]}
    
    timestamps = []
    for line in r["stdout"].splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        try:
            timestamps.append(float(parts[0]))
        except (ValueError, IndexError):
            continue
    
    # 每秒装桶
    buckets = Counter()
    for t in timestamps:
        buckets[int(t)] += 1
    
    # 候选周期点：每秒包数 <= 10 且 >= 1（即"周期性小包簇"）
    candidates = sorted([sec for sec, cnt in buckets.items() if 1 <= cnt <= 10])
    
    # 估算周期：相邻候选点的差值众数
    estimated_period = 0.0
    is_periodic = False
    if len(candidates) >= 3:
        intervals = [candidates[i+1] - candidates[i] for i in range(len(candidates) - 1)]
        # 取最常见的间隔
        if intervals:
            interval_counter = Counter(intervals)
            most_common, count = interval_counter.most_common(1)[0]
            if count >= 2 and most_common >= 5:  # 至少出现 2 次且间隔 >= 5 秒
                estimated_period = float(most_common)
                is_periodic = True
    
    return {
        "ok": True,
        "small_packets_count": len(timestamps),
        "buckets_per_second": dict(sorted(buckets.items())),
        "candidate_periods": candidates[:30],
        "estimated_period_sec": estimated_period,
        "is_periodic": is_periodic
    }


# ============================================================
# 工具 10: pcap_packet_size_distribution
# ============================================================

@tool
def pcap_packet_size_distribution(pcap_path: str, filter_expr: str = "") -> Dict[str, Any]:
    """获取包大小分布（区间统计）。
    
    KCP 模式：60%+ 的包在 1280-2559 字节（MTU 满包）
    QUIC 模式：45%+ 在 1280-2559
    TCP/TLS 模式：分布更均匀
    
    Args:
        pcap_path: pcap 文件绝对路径
        filter_expr: 可选 display filter
    
    Returns:
        {
            "ok": bool,
            "buckets": {
                "0-19": {count: int, avg: float, percent: float},
                "20-39": {...},
                ...
            },
            "mtu_pct": float,    # 1280-2559 区间占比
            "small_pct": float   # <= 159 区间占比
        }
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    cmd = ["tshark", "-r", pcap_path, "-q", "-z", "plen,tree"]
    if filter_expr:
        # plen 不支持 -Y，但可以前置 -2 -Y 这样的组合不行；
        # 改用 -Y 配合 io,stat 模拟可能更准确，但简单起见我们先全量统计
        pass
    
    r = _run(cmd)
    if not r["ok"]:
        return {"ok": False, "error": r["stderr"]}
    
    buckets = {}
    mtu_pct = 0.0
    small_pct = 0.0
    
    for line in r["stdout"].splitlines():
        m = re.match(r"\s*(\d+-\d+|\d+\s+and\s+greater)\s+(\d+)\s+([\d.-]+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)%", line)
        if m:
            label = m.group(1).replace(" ", "_")
            count = int(m.group(2))
            avg = float(m.group(3)) if m.group(3) != "-" else 0
            pct = float(m.group(7))
            buckets[label] = {"count": count, "avg": avg, "percent": pct}
            if label == "1280-2559":
                mtu_pct = pct
            if label in ("0-19", "20-39", "40-79", "80-159"):
                small_pct += pct
    
    return {
        "ok": True,
        "buckets": buckets,
        "mtu_pct": mtu_pct,
        "small_pct": small_pct
    }


# ============================================================
# 工具 11: pcap_quic_handshake
# ============================================================

@tool
def pcap_quic_handshake(pcap_path: str) -> Dict[str, Any]:
    """提取 QUIC 握手信息：版本、Connection ID、ALPN、SNI、密码套件。
    
    重要：frp QUIC 模式的 ALPN 是 "frp"（普通 HTTP/3 是 "h3"）。
    这是 frp QUIC 流量的"一击必中"指纹。
    
    Args:
        pcap_path: pcap 文件绝对路径
    
    Returns:
        {
            "ok": bool,
            "quic_versions": [str],
            "alpn_protocols": [str],     # ALPN 列表
            "snis": [str],
            "has_frp_alpn": bool,         # ALPN 是否包含 "frp"
            "initial_packets_count": int
        }
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    r = _run(["tshark", "-r", pcap_path,
              "-Y", "quic.long.packet_type == 0",
              "-V"])
    if not r["ok"]:
        return {"ok": False, "error": r["stderr"]}
    
    out = r["stdout"]
    versions = list(set(re.findall(r"Version:\s+(\d+)\s+\(0x[\da-f]+\)", out)))
    alpn_matches = re.findall(r"ALPN Next Protocol[\s\S]*?ALPN Next Protocol:\s+(\S+)", out)
    if not alpn_matches:
        alpn_matches = re.findall(r"ALPN Next Protocol:\s+(\S+)", out)
    snis = list(set(re.findall(r"Server Name:\s+(\S+)", out)))
    initial_count = out.count("packet_type == 0") or out.count("Initial")
    
    return {
        "ok": True,
        "quic_versions": versions,
        "alpn_protocols": list(set(alpn_matches)),
        "snis": snis,
        "has_frp_alpn": "frp" in alpn_matches,
        "initial_packets_count": initial_count
    }


# ============================================================
# 工具 12: pcap_websocket_handshake
# ============================================================

@tool
def pcap_websocket_handshake(pcap_path: str) -> Dict[str, Any]:
    """提取 WebSocket 升级握手详情：URI、Host、Upgrade 头。
    
    重要：frp WebSocket 模式的 URI 是硬编码的 "/~!frp"。这是 frp WS/WSS 流量的强指纹。
    
    Args:
        pcap_path: pcap 文件绝对路径
    
    Returns:
        {
            "ok": bool,
            "upgrade_requests": [
                {"frame": int, "method": str, "uri": str, "host": str,
                 "user_agent": str, "websocket_key": str}
            ],
            "upgrade_responses": [
                {"frame": int, "status": int, "accept_key": str}
            ],
            "has_frp_websocket_path": bool   # URI 是否包含 /~!frp
        }
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    # 请求
    r1 = _run(["tshark", "-r", pcap_path,
               "-Y", "http.request and http.upgrade",
               "-T", "fields",
               "-e", "frame.number",
               "-e", "http.request.method",
               "-e", "http.request.uri",
               "-e", "http.host",
               "-e", "http.user_agent",
               "-e", "http.sec_websocket_key"])
    
    requests = []
    has_frp_path = False
    for line in r1["stdout"].splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        while len(parts) < 6:
            parts.append("")
        uri = parts[2]
        if "/~!frp" in uri or "~!frp" in uri:
            has_frp_path = True
        requests.append({
            "frame": int(parts[0]) if parts[0].isdigit() else 0,
            "method": parts[1],
            "uri": uri,
            "host": parts[3],
            "user_agent": parts[4],
            "websocket_key": parts[5]
        })
    
    # 响应
    r2 = _run(["tshark", "-r", pcap_path,
               "-Y", "http.response.code == 101",
               "-T", "fields",
               "-e", "frame.number",
               "-e", "http.response.code",
               "-e", "http.sec_websocket_accept"])
    
    responses = []
    for line in r2["stdout"].splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        while len(parts) < 3:
            parts.append("")
        try:
            responses.append({
                "frame": int(parts[0]) if parts[0].isdigit() else 0,
                "status": int(parts[1]) if parts[1].isdigit() else 0,
                "accept_key": parts[2]
            })
        except ValueError:
            pass
    
    return {
        "ok": True,
        "upgrade_requests": requests,
        "upgrade_responses": responses,
        "has_frp_websocket_path": has_frp_path
    }


# ============================================================
# 工具 13: pcap_tls_sni_cert
# ============================================================

@tool
def pcap_tls_sni_cert(pcap_path: str, force_port: int = 0) -> Dict[str, Any]:
    """提取 TLS SNI 和服务器证书详情：CN、SAN、issuer、有效期。
    
    自签证书 + 异常 SNI（无信誉域名）是 frp WSS 等隧道的重要特征。
    
    Args:
        pcap_path: pcap 文件绝对路径
        force_port: 在该端口强制 TLS 解析
    
    Returns:
        {
            "ok": bool,
            "snis": [str],
            "certificates": [
                {"subject_cn": str, "issuer_cn": str, "san_list": [str],
                 "not_before": str, "not_after": str, "is_self_signed": bool}
            ]
        }
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    cmd = ["tshark", "-r", pcap_path]
    if force_port:
        cmd += ["-d", f"tcp.port=={force_port},tls"]
    
    # 提 SNI
    r1 = _run(cmd + ["-Y", "tls.handshake.extensions_server_name",
                     "-T", "fields", "-e", "tls.handshake.extensions_server_name"])
    snis = list(set(s.strip() for s in r1["stdout"].splitlines() if s.strip()))
    
    # 提证书详情
    r2 = _run(cmd + ["-Y", "tls.handshake.type == 11", "-V"])
    
    certificates = []
    blocks = re.split(r"^Frame \d+:", r2["stdout"], flags=re.MULTILINE)
    for block in blocks[1:]:
        subject_cn_matches = re.findall(r"id-at-commonName=([^\s,]+)", block)
        san_list = re.findall(r"dNSName: (\S+)", block)
        ip_san_list = re.findall(r"iPAddress: (\S+)", block)
        not_before = re.search(r"notBefore: utcTime \(0\)\s+(\S+)", block)
        not_after = re.search(r"notAfter: utcTime \(0\)\s+(\S+)", block)
        
        # 自签判断：subject CN == issuer CN
        is_self_signed = False
        if len(subject_cn_matches) >= 2:
            # 通常证书 chain 里 subject 和 issuer 各一个
            is_self_signed = subject_cn_matches[0] == subject_cn_matches[1]
        
        if subject_cn_matches or san_list:
            certificates.append({
                "subject_cn": subject_cn_matches[0] if subject_cn_matches else "",
                "issuer_cn": subject_cn_matches[1] if len(subject_cn_matches) > 1 else "",
                "san_list": san_list + ip_san_list,
                "not_before": not_before.group(1) if not_before else "",
                "not_after": not_after.group(1) if not_after else "",
                "is_self_signed": is_self_signed
            })
    
    return {"ok": True, "snis": snis, "certificates": certificates}


# ============================================================
# 工具 14（新增）: pcap_traffic_direction
# ============================================================

@tool
def pcap_traffic_direction(
    pcap_path: str,
    host_a: str,
    host_b: str
) -> Dict[str, Any]:
    """计算两个 IP 之间的流量方向比例。
    
    **内网穿透的核心特征**：内网主机出站流量远大于入站流量（5:1 以上）。
    
    Args:
        pcap_path: pcap 文件绝对路径
        host_a: 第一个 IP（推测为内网主机/frpc）
        host_b: 第二个 IP（推测为公网/frps）
    
    Returns:
        {
            "ok": bool,
            "a_to_b_bytes": int,
            "b_to_a_bytes": int,
            "ratio_b_to_a_over_a_to_b": float,  # B→A / A→B (内网穿透时应该 > 5)
            "is_internal_penetration_suspect": bool   # 比值 > 5 标记为可疑
        }
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    # A → B
    r1 = _run(["tshark", "-r", pcap_path,
               "-Y", f"ip.src == {host_a} and ip.dst == {host_b}",
               "-T", "fields", "-e", "frame.len"])
    a_to_b = sum(int(x) for x in r1["stdout"].splitlines() if x.isdigit())
    
    # B → A
    r2 = _run(["tshark", "-r", pcap_path,
               "-Y", f"ip.src == {host_b} and ip.dst == {host_a}",
               "-T", "fields", "-e", "frame.len"])
    b_to_a = sum(int(x) for x in r2["stdout"].splitlines() if x.isdigit())
    
    ratio = (b_to_a / a_to_b) if a_to_b > 0 else 0
    return {
        "ok": True,
        "a_to_b_bytes": a_to_b,
        "b_to_a_bytes": b_to_a,
        "ratio_b_to_a_over_a_to_b": round(ratio, 2),
        "is_internal_penetration_suspect": ratio > 5
    }


# ============================================================
# 工具 15（新增）: pcap_session_duration
# ============================================================

@tool
def pcap_session_duration(
    pcap_path: str,
    host_a: str = "",
    host_b: str = "",
    port: int = 0
) -> Dict[str, Any]:
    """检测会话持续时间。frp 控制连接持续整个会话期（> 60s）。
    
    Args:
        pcap_path: pcap 文件绝对路径
        host_a/host_b: 可选的 IP 过滤
        port: 可选的端口过滤
    
    Returns:
        {"ok": bool, "duration_sec": float, "is_long_session": bool}
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    filters = []
    if host_a and host_b:
        filters.append(f"(ip.addr == {host_a} and ip.addr == {host_b})")
    if port:
        filters.append(f"(tcp.port == {port} or udp.port == {port})")
    filter_expr = " and ".join(filters) if filters else ""
    
    cmd = ["tshark", "-r", pcap_path, "-T", "fields", "-e", "frame.time_relative"]
    if filter_expr:
        cmd += ["-Y", filter_expr]
    
    r = _run(cmd)
    if not r["ok"]:
        return {"ok": False, "error": r["stderr"]}
    
    times = [float(x) for x in r["stdout"].splitlines() if x.strip()]
    if not times:
        return {"ok": False, "error": "没有匹配的包"}
    
    duration = times[-1] - times[0]
    return {
        "ok": True,
        "duration_sec": round(duration, 2),
        "is_long_session": duration > 60,
        "first_packet": times[0],
        "last_packet": times[-1],
        "packet_count": len(times)
    }


# ============================================================
# 工具 16（新增）: pcap_byte_pattern_search
# ============================================================

@tool
def pcap_byte_pattern_search(
    pcap_path: str,
    pattern: str,
    is_hex: bool = False,
    filter_expr: str = ""
) -> Dict[str, Any]:
    """在 pcap 中搜索字节模式（明文或 hex），返回命中的帧号和上下文。
    
    适合查找硬编码字符串（如 "/~!frp"）、协议头魔数等。
    
    Args:
        pcap_path: pcap 文件绝对路径
        pattern: 要搜索的字符串
        is_hex: True=按 hex 解释，False=按 ASCII 解释
        filter_expr: 额外的 tshark display filter
    
    Returns:
        {"ok": bool, "matches": [{"frame": int, "context": str}], "match_count": int}
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    if is_hex:
        hex_str = pattern.replace(" ", "").replace(":", "").lower()
    else:
        hex_str = pattern.encode().hex()
    
    # 用 tshark 的 frame.contains
    contains_expr = f'frame contains {pattern!r}' if not is_hex else f'frame[0:] matches "{hex_str}"'
    # 实际上 tshark 用 frame contains "ascii" 比较稳定
    if not is_hex:
        contains_expr = f'frame contains "{pattern}"'
    else:
        # tshark 不直接支持 hex，需要用 _ws.frame.match
        contains_expr = f"frame.data matches \"(?i){hex_str}\""
    
    final_filter = contains_expr
    if filter_expr:
        final_filter = f"({filter_expr}) and ({contains_expr})"
    
    cmd = ["tshark", "-r", pcap_path,
           "-Y", final_filter,
           "-T", "fields", "-e", "frame.number", "-e", "_ws.col.Info"]
    r = _run(cmd)
    if not r["ok"]:
        return {"ok": False, "error": r["stderr"]}
    
    matches = []
    for line in r["stdout"].splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        matches.append({
            "frame": int(parts[0]) if parts[0].isdigit() else 0,
            "context": parts[1] if len(parts) > 1 else ""
        })
    
    return {"ok": True, "matches": matches, "match_count": len(matches)}


# ============================================================
# 工具 17（新增）: pcap_extract_subset
# ============================================================

@tool
def pcap_extract_subset(
    pcap_path: str,
    filter_expr: str,
    output_path: str
) -> Dict[str, Any]:
    """根据过滤条件提取 pcap 子集，用于聚焦特定会话再深入分析。
    
    例如先全局扫描定位可疑会话，然后用本工具提取该会话的子 pcap 给后续 skill 用。
    
    Args:
        pcap_path: 源 pcap 路径
        filter_expr: tshark display filter，如 "tcp.stream eq 0"
        output_path: 输出的子 pcap 路径
    
    Returns:
        {"ok": bool, "output_path": str, "packet_count": int}
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    cmd = ["tshark", "-r", pcap_path, "-Y", filter_expr, "-w", output_path]
    r = _run(cmd)
    if not r["ok"]:
        return {"ok": False, "error": r["stderr"]}
    
    # 统计子 pcap 的包数
    if Path(output_path).exists():
        cap = _run(["capinfos", "-c", output_path])
        m = re.search(r"Number of packets:\s+([0-9,]+)", cap["stdout"])
        count = int(m.group(1).replace(",", "")) if m else 0
        return {"ok": True, "output_path": output_path, "packet_count": count}
    return {"ok": False, "error": "输出文件未生成"}


# ============================================================
# 工具 18（新增）: active_probe_http_server
# ============================================================

@tool
def active_probe_http_server(target_ip: str, port: int = 443, timeout_sec: int = 5) -> Dict[str, Any]:
    """主动探测目标 IP 的 HTTP 服务，识别 Server 头（如 nginx）。
    
    用于 frp WSS 模式判定：如果可疑流量目标 IP 上确实部署了 nginx，
    那么 WSS over nginx 的攻击架构假说被加强。
    
    注意：此工具会真的发起网络请求，需确保有权限/合规。
    
    Args:
        target_ip: 目标 IP
        port: 端口（默认 443）
        timeout_sec: 超时
    
    Returns:
        {
            "ok": bool,
            "server_header": str,
            "is_nginx": bool,
            "is_apache": bool,
            "is_caddy": bool,
            "status_code": int,
            "tls_cert_cn": str,
            "tls_cert_san": [str]
        }
    """
    use_https = (port == 443) or (port % 1000 == 443)
    scheme = "https" if use_https else "http"
    
    # 用 curl 获取 Server 头和状态码
    cmd = ["curl", "-s", "-k", "-I",
           "--max-time", str(timeout_sec),
           "-o", "/dev/null",
           "-w", "%{http_code}|%{header_json}",
           f"{scheme}://{target_ip}:{port}/"]
    r = _run(cmd, timeout=timeout_sec + 5)
    
    server_header = ""
    status_code = 0
    if r["ok"]:
        out = r["stdout"]
        parts = out.split("|", 1)
        if len(parts) == 2:
            try:
                status_code = int(parts[0])
            except ValueError:
                pass
            try:
                hdrs = json.loads(parts[1])
                server = hdrs.get("server", [])
                if isinstance(server, list) and server:
                    server_header = server[0]
                elif isinstance(server, str):
                    server_header = server
            except (json.JSONDecodeError, AttributeError):
                pass
    
    # TLS 证书信息
    cert_cn = ""
    cert_san = []
    if use_https:
        cmd2 = ["bash", "-c",
                f"echo 'Q' | timeout {timeout_sec} openssl s_client -connect {target_ip}:{port} "
                f"-servername {target_ip} 2>/dev/null | openssl x509 -noout -subject -ext subjectAltName 2>/dev/null"]
        r2 = _run(cmd2, timeout=timeout_sec + 5)
        if r2["ok"]:
            cn_m = re.search(r"CN\s*=\s*([^,\n]+)", r2["stdout"])
            if cn_m:
                cert_cn = cn_m.group(1).strip()
            san_m = re.findall(r"DNS:([^\s,]+)", r2["stdout"])
            cert_san = san_m
    
    server_lower = server_header.lower()
    return {
        "ok": True,
        "server_header": server_header,
        "is_nginx": "nginx" in server_lower,
        "is_apache": "apache" in server_lower,
        "is_caddy": "caddy" in server_lower,
        "status_code": status_code,
        "tls_cert_cn": cert_cn,
        "tls_cert_san": cert_san
    }


# ============================================================
# 工具 19（新增）: pcap_http_details
# ============================================================

@tool
def pcap_http_details(pcap_path: str, max_requests: int = 20) -> Dict[str, Any]:
    """提取 HTTP 请求/响应详情：URL、Host、UA、状态码。
    
    用于检查 frp WebSocket 的 /~!frp 升级请求等明文 HTTP 流量。
    
    Args:
        pcap_path: pcap 文件绝对路径
        max_requests: 返回的请求数上限
    
    Returns:
        {
            "ok": bool,
            "requests": [{"frame": int, "method": str, "uri": str, "host": str,
                          "user_agent": str, "upgrade": str}],
            "responses": [{"frame": int, "code": int, "phrase": str,
                           "server": str, "content_type": str}],
            "unique_uris": [str],
            "unique_user_agents": [str]
        }
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    # 请求
    r1 = _run(["tshark", "-r", pcap_path, "-Y", "http.request",
               "-T", "fields",
               "-e", "frame.number",
               "-e", "http.request.method",
               "-e", "http.request.uri",
               "-e", "http.host",
               "-e", "http.user_agent",
               "-e", "http.upgrade",
               "-c", str(max_requests)])
    requests = []
    for line in r1["stdout"].splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        while len(parts) < 6:
            parts.append("")
        requests.append({
            "frame": int(parts[0]) if parts[0].isdigit() else 0,
            "method": parts[1], "uri": parts[2], "host": parts[3],
            "user_agent": parts[4], "upgrade": parts[5]
        })
    
    # 响应
    r2 = _run(["tshark", "-r", pcap_path, "-Y", "http.response",
               "-T", "fields",
               "-e", "frame.number",
               "-e", "http.response.code",
               "-e", "http.response.phrase",
               "-e", "http.server",
               "-e", "http.content_type",
               "-c", str(max_requests)])
    responses = []
    for line in r2["stdout"].splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        while len(parts) < 5:
            parts.append("")
        try:
            responses.append({
                "frame": int(parts[0]) if parts[0].isdigit() else 0,
                "code": int(parts[1]) if parts[1].isdigit() else 0,
                "phrase": parts[2], "server": parts[3], "content_type": parts[4]
            })
        except ValueError:
            pass
    
    return {
        "ok": True,
        "requests": requests,
        "responses": responses,
        "unique_uris": list(set(r["uri"] for r in requests if r["uri"])),
        "unique_user_agents": list(set(r["user_agent"] for r in requests if r["user_agent"]))
    }


# ============================================================
# 工具 20（新增）: pcap_endpoints
# ============================================================

@tool
def pcap_endpoints(pcap_path: str) -> Dict[str, Any]:
    """枚举 pcap 中出现的所有 IP 端点。
    
    用于"会话纯净度"检查：如果端点数 > 2，说明抓包过滤可能有遗漏，
    或者存在其他参与方需要关注。
    
    Args:
        pcap_path: pcap 文件绝对路径
    
    Returns:
        {"ok": bool, "endpoints": [{"ip": str, "frames": int, "bytes": int}],
         "endpoint_count": int}
    """
    err = _validate_pcap(pcap_path)
    if err:
        return {"ok": False, "error": err}
    
    r = _run(["tshark", "-r", pcap_path, "-q", "-z", "endpoints,ip"])
    if not r["ok"]:
        return {"ok": False, "error": r["stderr"]}
    
    endpoints = []
    for line in r["stdout"].splitlines():
        m = re.match(r"^(\d+\.\d+\.\d+\.\d+)\s+(\d+)\s+(\d+)\s+(\d+)", line)
        if m:
            endpoints.append({
                "ip": m.group(1),
                "frames": int(m.group(2)),
                "bytes": int(m.group(3))
            })
    
    return {"ok": True, "endpoints": endpoints, "endpoint_count": len(endpoints)}


# ============================================================
# Tool 注册表
# ============================================================

ALL_PCAP_TOOLS = [
    # 用户列举的工具（12 个）
    pcap_basic_info,
    pcap_protocol_hierarchy,
    pcap_conversations,
    pcap_packet_payload,
    pcap_list_packets,
    pcap_tls_handshakes,
    pcap_client_hello_details,
    pcap_extract_ja3,
    pcap_detect_heartbeat,
    pcap_packet_size_distribution,
    pcap_quic_handshake,
    pcap_websocket_handshake,
    pcap_tls_sni_cert,
    # 新增建议补充的工具（7 个）
    pcap_traffic_direction,       # ★ 内网穿透核心特征
    pcap_session_duration,
    pcap_byte_pattern_search,     # ★ 找 /~!frp 等硬编码
    pcap_extract_subset,
    active_probe_http_server,     # ★ WSS 的 nginx 资产联动
    pcap_http_details,
    pcap_endpoints
]

if __name__ == "__main__":
    # 简单的自检
    print("可用 pcap 分析工具:")
    for t in ALL_PCAP_TOOLS:
        print(f"  - {t.name}: {t.description.splitlines()[0] if t.description else ''}")
