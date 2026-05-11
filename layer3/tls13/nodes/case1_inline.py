# layer3/tls13/nodes/case1_inline.py
from pymongo import MongoClient
from utils import config
from layer3.tls13.db.payload_dao import fetch_stream_hex
from layer3.tls13.parsers.tls_record import find_hello_records


def case1_inline(state: dict) -> dict:
    """会话开头直接抓: ClientHello 在 orig 方向开头, ServerHello 在 resp 方向开头。"""
    client = MongoClient(config.DATABASE["mongo"]["uri"])
    db = client[config.DATABASE["mongo"]["db_name"]]

    streams = fetch_stream_hex(db, state["payload_collection"],
                               state["current_flow"]["uid"])

    update = {"case_tried": state.get("case_tried", []) + ["case1"]}

    # 限定前 8KB 防止误吃后续大包
    orig_head = streams["orig"][: 8 * 1024 * 2]
    resp_head = streams["resp"][: 8 * 1024 * 2]

    ch = next((r for r in find_hello_records(orig_head)
               if not r.is_server_hello), None)
    sh = next((r for r in find_hello_records(resp_head)
               if r.is_server_hello), None)

    if ch:
        update["client_hello_hex"] = ch.record_bytes.hex()
        update["extracted_meta"] = {"sni": ch.sni, "is_tls13": ch.is_tls13}
    if sh:
        update["server_hello_hex"] = sh.record_bytes.hex()
        update.setdefault("extracted_meta", {}).update({
            "cipher_suite": sh.cipher_suite,
            "sh_is_tls13":  sh.is_tls13,
        })
    if ch and sh:
        update["extracted_from"] = "case1"
    return {**state, **update}