"""
读取 conn 集合的辅助。兼容两种字段命名:
  (A) 原始 Zeek JSON: id.orig_h, id.orig_p, id.resp_h, id.resp_p
  (B) step3_1_dataclean.py 清洗后: id_orig_h, id_orig_p, id_resp_h, id_resp_p
"""
from pymongo import DESCENDING


def conn_id(doc: dict, part: str):
    """取 conn 五元组分量。part ∈ {'orig_h','orig_p','resp_h','resp_p'}"""
    return doc.get(f"id.{part}") or doc.get(f"id_{part}")


def find_payload_by_uid(db, payload_col, uid):
    """case3_zeek_split 需要按 uid 找 payload（多条 packet 级）"""
    from layer3.tls13.db.payload_dao import fetch_stream_hex
    return fetch_stream_hex(db, payload_col, uid)


def find_prior_conn_by_community_id(db, conn_col, cid, before_ts,
                                    must_have_syn=True):
    """case3_zeek_split: 用 community_id 向前追溯有 SYN 握手的那条流"""
    q = {"community_id": cid, "ts": {"$lt": before_ts}}
    if must_have_syn:
        q["history"] = {"$regex": "[SH]"}
    return db[conn_col].find_one(q, sort=[("ts", DESCENDING)])


def query_conn_by_5tuple(db, conn_col, sip, sport, dip, dport, proto="tcp"):
    """case2 需要按五元组查 conn / ssl 时用"""
    # 优先用带点 schema 查，没命中再用下划线 schema
    for q in [
        {"id.orig_h": sip, "id.orig_p": sport,
         "id.resp_h": dip, "id.resp_p": dport, "proto": proto},
        {"id_orig_h": sip, "id_orig_p": sport,
         "id_resp_h": dip, "id_resp_p": dport, "proto": proto},
    ]:
        doc = db[conn_col].find_one(q)
        if doc:
            return doc
    return None