from pymongo import MongoClient
from utils import config
from layer3.tls13.parsers.tls_record import find_hello_records
from layer3.tls13.db.ssl_log_dao import find_ssl_by_5tuple
from layer3.tls13.db.payload_dao import fetch_stream_hex
from layer3.tls13.db.conn_dao import conn_id


def _scan_for_record_start(data: bytes, hint_offset: int, look_back: int = 4096) -> int:
    """从 hint_offset 向前找最近的 TLS record header (0x16 0x03 0x01|03)。"""
    start = max(0, hint_offset - look_back)
    for i in range(hint_offset, start - 1, -1):
        if i + 5 <= len(data) and data[i] == 0x16 and data[i+1] == 0x03 \
                and data[i+2] in (0x01, 0x03):
            return i
    return -1


def case2_late_capture(state: dict) -> dict:
    print("进入case2")
    flow = state["current_flow"]
    update = {"case_tried": state.get("case_tried", []) + ["case2"]}

    # 替换原本的 payload 拉取方式，走独立拉流机制
    client = MongoClient(config.DATABASE["mongo"]["uri"])
    db = client[config.DATABASE["mongo"]["db_name"]]
    streams = fetch_stream_hex(db, state["payload_collection"], flow["uid"])
    
    orig_hex = streams.get("orig", "")
    resp_hex = streams.get("resp", "")

    # 1. 直接全量扫一遍（orig 扫 CH, resp 扫 SH）
    ch = next((r for r in find_hello_records(orig_hex) if not r.is_server_hello), None)
    sh = next((r for r in find_hello_records(resp_hex) if r.is_server_hello), None)

    # 2. 如果还没找到 CH，借助 ssl.log 的 server_name 在明文里反查
    if not ch:
        ssl_row = find_ssl_by_5tuple(
            state["ssl_collection"],
            conn_id(flow, "orig_h"), int(conn_id(flow, "orig_p")),
            conn_id(flow, "resp_h"), int(conn_id(flow, "resp_p")),
        )
        if ssl_row and ssl_row.get("server_name"):
            sni = ssl_row["server_name"].encode("ascii", errors="ignore")
            data_orig = bytes.fromhex(orig_hex)
            idx = data_orig.find(sni)
            if idx > 0:
                rec_start = _scan_for_record_start(data_orig, idx)
                if rec_start >= 0:
                    sub_hex = data_orig[rec_start:].hex()
                    sub_records = find_hello_records(sub_hex)
                    ch = next((r for r in sub_records if not r.is_server_hello), None)
                    # 反查只针对 SNI (ClientHello)

    if ch:
        update["client_hello_hex"] = ch.record_bytes.hex()
        update.setdefault("extracted_meta", {}).update(
            {"sni": ch.sni, "is_tls13": ch.is_tls13})
    if sh:
        update["server_hello_hex"] = sh.record_bytes.hex()
        update.setdefault("extracted_meta", {}).update({
            "cipher_suite": sh.cipher_suite,
            "sh_is_tls13": sh.is_tls13,
        })
    if ch and sh:
        update["extracted_from"] = "case2"
        
    return {**state, **update}