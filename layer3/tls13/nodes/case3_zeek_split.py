from pymongo import MongoClient
from utils import config
from layer3.tls13.db.conn_dao import find_prior_conn_by_community_id, conn_id
from layer3.tls13.db.ssl_log_dao import find_ssl_by_uid, find_resumed_session_origin
from layer3.tls13.parsers.tls_record import find_hello_records
from layer3.tls13.db.payload_dao import fetch_stream_hex


def case3_zeek_split(state: dict) -> dict:
    flow = state["current_flow"]
    update = {"case_tried": state.get("case_tried", []) + ["case3"]}

    # ---- 3a. 看本流 ssl.log 是不是 resumed=T（PSK / 0-RTT）----
    ssl_row = find_ssl_by_uid(state["ssl_collection"], flow["uid"])
    if ssl_row and ssl_row.get("resumed") is True and ssl_row.get("session_id"):
        origin = find_resumed_session_origin(
            state["ssl_collection"], ssl_row["session_id"], before_uid=flow["uid"])
        if origin:
            # 找到首次完整握手的那条历史流，去 payload 集合捞它的 hex
            update["extracted_from"] = "case3_psk"
            update["extracted_meta"] = {
                "resumed": True,
                "original_uid": origin["uid"],
                "original_server_name": origin.get("server_name"),
            }
            # 这里只标记，真正的 CH/SH 字节由 payload_analyse 阶段决定要不要继续拉
            return {**state, **update}

    # ---- 3b. Zeek 把同一长连接拆成多条：用 community_id 向前追溯 ----
    cid = state.get("community_id") or flow.get("community_id")
    if cid and flow.get("conn_state") in ("OTH", "S2", "S3"):  # 没看到 SYN 握手
        prior = find_prior_conn_by_community_id(
            state["conn_collection"], cid,
            before_ts=flow["ts"],
            must_have_syn=True,           # 状态里含 'S'/'H'
        )
        if prior:
            # 验证 SEQ 衔接：当前流 init_seq ≈ prior.last_seq
            cur_init = flow.get("orig_seq_init")
            prior_last = prior.get("orig_seq_last")
            seq_ok = (cur_init is not None and prior_last is not None
                      and abs(cur_init - prior_last) < 64)  # 容忍小重传
            if seq_ok or cur_init is None:    # 没 SEQ 字段也允许（退化为五元组）
                
                # 改用 fetch_stream_hex 动态拉取 prior 的 payload
                client = MongoClient(config.DATABASE["mongo"]["uri"])
                db = client[config.DATABASE["mongo"]["db_name"]]
                prior_streams = fetch_stream_hex(db, state["payload_collection"], prior["uid"])
                
                # 分别对 orig 和 resp 提取 Hello
                ch = next((r for r in find_hello_records(prior_streams.get("orig", "")) 
                           if not r.is_server_hello), None)
                sh = next((r for r in find_hello_records(prior_streams.get("resp", "")) 
                           if r.is_server_hello), None)
                
                if ch:
                    update["client_hello_hex"] = ch.record_bytes.hex()
                if sh:
                    update["server_hello_hex"] = sh.record_bytes.hex()
                if ch and sh:
                    update["extracted_from"] = "case3_split"
                    update["extracted_meta"] = {
                        "sni": ch.sni,
                        "is_tls13": ch.is_tls13,
                        "cipher_suite": sh.cipher_suite,
                        "merged_from_uid": prior["uid"],
                        "seq_continuity": seq_ok,
                    }
                    
    return {**state, **update}