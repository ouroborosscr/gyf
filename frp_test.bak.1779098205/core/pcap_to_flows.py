"""
pcap_to_flows.py — 把 pcap 转换为 Zeek 风格的 flow 列表

输出格式与 gyf 现有 conn_collection 一致，包含字段：
    uid, ts, id.orig_h, id.orig_p, id.resp_h, id.resp_p, proto,
    duration, orig_bytes, resp_bytes, orig_pkts, resp_pkts,
    conn_state, history, batch_index, stream_payload_decoded

这样可以直接喂给 rag.py 的 _build_batch_profile / _extract_payload_summary。
"""

import subprocess
import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


def _run_tshark(args: List[str], timeout: int = 60) -> str:
    cmd = ["tshark"] + args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"tshark failed: {r.stderr[:500]}")
    return r.stdout


def _make_uid(src_ip: str, src_port: int, dst_ip: str, dst_port: int, proto: str) -> str:
    """生成会话唯一 ID（Zeek uid 风格）"""
    key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{proto}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return f"C{h}"


def pcap_to_flows(pcap_path: str, max_payload_chars: int = 1024) -> List[Dict[str, Any]]:
    """
    从 pcap 提取所有 TCP/UDP 会话，每个会话作为一条 Zeek 风格的 flow。
    
    Args:
        pcap_path: pcap 文件路径
        max_payload_chars: 每条流提取的 payload 字符上限
    
    Returns:
        flows: list[dict]，每个 dict 一个会话
    """
    pcap_path = str(Path(pcap_path).resolve())
    
    # 1. 拿 conv,tcp 和 conv,udp 的会话统计
    flows = []
    for proto in ("tcp", "udp"):
        out = _run_tshark(["-r", pcap_path, "-q", "-z", f"conv,{proto}"])
        for line in out.splitlines():
            if "<->" not in line:
                continue
            m = re.match(
                r"\s*(\S+?):(\d+)\s+<->\s+(\S+?):(\d+)\s+"
                r"([\d,]+)\s+(\S+\s*\S*)\s+"
                r"([\d,]+)\s+(\S+\s*\S*)\s+"
                r"([\d,]+)\s+(\S+\s*\S*)\s+"
                r"([\d.]+)\s+([\d.]+)",
                line
            )
            if not m:
                continue
            
            def to_int(s): return int(s.replace(",", ""))
            def to_bytes(s):
                num = re.match(r"([\d.]+)", s.strip())
                if not num:
                    return 0
                n = float(num.group(1))
                if "k" in s.lower():
                    n *= 1024
                elif "m" in s.lower():
                    n *= 1024 * 1024
                return int(n)
            
            src_ip = m.group(1)
            src_port = int(m.group(2))
            dst_ip = m.group(3)
            dst_port = int(m.group(4))
            
            flow = {
                "uid": _make_uid(src_ip, src_port, dst_ip, dst_port, proto),
                "ts": float(m.group(11)),
                "id.orig_h": src_ip,
                "id.orig_p": src_port,
                "id.resp_h": dst_ip,
                "id.resp_p": dst_port,
                "proto": proto,
                "duration": float(m.group(12)),
                "orig_bytes": to_bytes(m.group(6)),     # frames a→b → bytes
                "resp_bytes": to_bytes(m.group(8)),
                "orig_pkts": to_int(m.group(5)),
                "resp_pkts": to_int(m.group(7)),
                "conn_state": "SF" if to_int(m.group(9)) > 0 else "OTH",
                "history": "ShADdaFf" if proto == "tcp" else "Dd",
                "stream_payload_decoded": ""
            }
            flows.append(flow)
    
    # 2. 给每条流抓 payload 摘要
    flows_by_key = {(f["id.orig_h"], f["id.orig_p"], f["id.resp_h"], f["id.resp_p"]): f for f in flows}
    
    # 对每条流单独抓 payload (前 N 个有数据的包)
    for key, flow in flows_by_key.items():
        proto = flow["proto"]
        filter_expr = (
            f"{proto}.srcport == {key[1]} and {proto}.dstport == {key[3]} "
            f"and ip.src == {key[0]} and ip.dst == {key[2]} "
            f"and {proto}.len > 0"
        )
        try:
            out = _run_tshark([
                "-r", pcap_path,
                "-Y", filter_expr,
                "-T", "fields",
                "-e", f"{proto}.payload",
                "-c", "5"
            ])
            hex_payloads = [p.replace(":", "") for p in out.splitlines() if p.strip()]
            if hex_payloads:
                combined_hex = "".join(hex_payloads)[:max_payload_chars * 2]
                # hex → ASCII（保留可打印字符 + 特殊标记）
                ascii_chars = []
                for i in range(0, len(combined_hex), 2):
                    try:
                        b = int(combined_hex[i:i+2], 16)
                        if 32 <= b < 127:
                            ascii_chars.append(chr(b))
                        else:
                            ascii_chars.append(".")
                    except ValueError:
                        ascii_chars.append("?")
                flow["stream_payload_decoded"] = "".join(ascii_chars)[:max_payload_chars]
        except Exception as e:
            flow["stream_payload_decoded"] = ""
    
    # 3. 加 batch_index（按 ts 排序）
    flows.sort(key=lambda f: f["ts"])
    for idx, f in enumerate(flows):
        f["batch_index"] = idx
    
    return flows


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python pcap_to_flows.py <pcap_path>")
        sys.exit(1)
    
    flows = pcap_to_flows(sys.argv[1])
    print(f"提取到 {len(flows)} 条流")
    for f in flows[:3]:
        print(json.dumps(f, ensure_ascii=False, indent=2)[:500])
        print("---")
