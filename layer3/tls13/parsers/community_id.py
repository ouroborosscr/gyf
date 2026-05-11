import base64
import hashlib
import socket
import struct

# Community ID v1 简化实现（参考 https://github.com/corelight/community-id-spec）
PROTO_NUM = {"tcp": 6, "udp": 17, "icmp": 1, "icmp6": 58, "sctp": 132}

def _packed_ip(ip: str) -> bytes:
    try:
        return socket.inet_pton(socket.AF_INET, ip)
    except OSError:
        return socket.inet_pton(socket.AF_INET6, ip)

def community_id_v1(proto: str, src: str, sport: int,
                    dst: str, dport: int, seed: int = 0) -> str:
    p = PROTO_NUM[proto.lower()]
    a, b = _packed_ip(src), _packed_ip(dst)
    sp, dp = sport, dport
    # 排序，使方向对称
    if (a, sp) > (b, dp):
        a, b = b, a
        sp, dp = dp, sp
    h = hashlib.sha1()
    h.update(struct.pack(">H", seed))
    h.update(a); h.update(b)
    h.update(struct.pack(">BB", p, 0))
    h.update(struct.pack(">HH", sp, dp))
    return "1:" + base64.b64encode(h.digest()).decode()