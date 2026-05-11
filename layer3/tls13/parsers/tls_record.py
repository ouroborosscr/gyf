from dataclasses import dataclass, field
from typing import Optional

@dataclass
class HelloRecord:
    is_server_hello: bool
    record_offset: int              # 在 payload hex 中的字节偏移
    record_bytes: bytes             # 完整 record（含 5 字节 record header）
    legacy_version: int             # 0x0303
    is_tls13: bool                  # 由 supported_versions ext 决定
    sni: Optional[str] = None       # ClientHello
    cipher_suite: Optional[int] = None  # ServerHello 选中的 cipher
    session_id: Optional[bytes] = None
    extensions: dict = field(default_factory=dict)

def _read_u8(b, i): return b[i], i+1
def _read_u16(b, i): return int.from_bytes(b[i:i+2], "big"), i+2
def _read_u24(b, i): return int.from_bytes(b[i:i+3], "big"), i+3
def _read_bytes(b, i, n): return b[i:i+n], i+n

def find_hello_records(payload_hex: str) -> list[HelloRecord]:
    """扫描整段 payload，找出所有 TLS handshake 中的 ClientHello / ServerHello。"""
    data = bytes.fromhex(payload_hex)
    out = []
    i = 0
    while i + 5 <= len(data):
        if data[i] != 0x16:           # not handshake
            i += 1; continue
        if data[i+1] != 0x03:         # not TLS
            i += 1; continue
        rec_ver = data[i+2]
        rec_len = int.from_bytes(data[i+3:i+5], "big")
        if rec_len == 0 or i + 5 + rec_len > len(data):
            i += 1; continue
        hs = data[i+5 : i+5+rec_len]
        if len(hs) < 4:
            i += 5 + rec_len; continue
        hs_type = hs[0]
        if hs_type not in (0x01, 0x02):   # 1=CH, 2=SH
            i += 5 + rec_len; continue

        try:
            rec = _parse_hello(hs, hs_type)
            rec.record_offset = i
            rec.record_bytes = data[i : i+5+rec_len]
            out.append(rec)
        except Exception:
            pass
        i += 5 + rec_len
    return out

def _parse_hello(hs: bytes, hs_type: int) -> HelloRecord:
    # 4 字节 handshake header (type + 3 字节 length)
    p = 4
    legacy_version, p = _read_u16(hs, p)
    _random, p = _read_bytes(hs, p, 32)
    sid_len, p = _read_u8(hs, p)
    session_id, p = _read_bytes(hs, p, sid_len)

    if hs_type == 0x01:  # ClientHello
        cs_len, p = _read_u16(hs, p)
        _cs, p = _read_bytes(hs, p, cs_len)
        cm_len, p = _read_u8(hs, p)
        _cm, p = _read_bytes(hs, p, cm_len)
        chosen_cipher = None
    else:  # ServerHello
        chosen_cipher, p = _read_u16(hs, p)
        _cm, p = _read_u8(hs, p)

    extensions = {}
    is_tls13 = False
    sni = None
    if p < len(hs):
        ext_total_len, p = _read_u16(hs, p)
        end = p + ext_total_len
        while p + 4 <= end:
            ext_type, p = _read_u16(hs, p)
            ext_len, p = _read_u16(hs, p)
            ext_data, p = _read_bytes(hs, p, ext_len)
            extensions[ext_type] = ext_data
            if ext_type == 0x002b:  # supported_versions
                # CH: list of versions; SH: single 2 bytes
                if hs_type == 0x01:
                    if len(ext_data) >= 1:
                        n = ext_data[0]
                        vers = ext_data[1:1+n]
                        for k in range(0, len(vers), 2):
                            if vers[k:k+2] == b"\x03\x04":
                                is_tls13 = True
                else:
                    if ext_data == b"\x03\x04":
                        is_tls13 = True
            elif ext_type == 0x0000 and hs_type == 0x01:  # SNI
                # server_name list
                try:
                    list_len = int.from_bytes(ext_data[0:2], "big")
                    name_type = ext_data[2]
                    name_len = int.from_bytes(ext_data[3:5], "big")
                    sni = ext_data[5:5+name_len].decode("ascii", errors="replace")
                except Exception:
                    pass

    return HelloRecord(
        is_server_hello=(hs_type == 0x02),
        record_offset=-1,           # 调用方覆盖
        record_bytes=b"",
        legacy_version=legacy_version,
        is_tls13=is_tls13,
        sni=sni,
        cipher_suite=chosen_cipher,
        session_id=session_id,
        extensions=extensions,
    )