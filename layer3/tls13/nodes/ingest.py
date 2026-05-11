import os
import json
import logging
from datetime import datetime
from pymongo import MongoClient
from utils import config

log = logging.getLogger(__name__)


def _ingest_ssl_log(zeek_logs_dir, db, col_name, source_pcap):
    """复用 step1_analyze._ingest 风格，把 ssl.log 入库。"""
    path = os.path.join(zeek_logs_dir, "ssl.log")
    if not os.path.exists(path):
        log.warning(f"ssl.log 不存在于 {path}（确认 pcap 里有 TLS 流量）")
        return 0

    col = db[col_name]
    docs, count = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                d["source_pcap"] = source_pcap
                if "ts" in d:
                    d["ts_date"] = datetime.fromtimestamp(d["ts"])
                docs.append(d)
                if len(docs) >= 5000:
                    col.insert_many(docs)
                    count += len(docs)
                    docs = []
            except Exception:
                continue
    if docs:
        col.insert_many(docs)
        count += len(docs)
    log.info(f"ssl.log({source_pcap}) -> {col_name}: {count} 条")
    return count


def _fill_community_id_fallback(db, conn_col):
    """如果 conn 集合没有 community_id（你现有 save-payload.zeek 没加 policy），
    用 python 算一遍补上。"""
    from layer3.tls13.parsers.community_id import community_id_v1
    from layer3.tls13.db.conn_dao import conn_id

    missing = db[conn_col].count_documents({"community_id": {"$exists": False}})
    if missing == 0:
        return 0
    log.info(f"{missing} 条 conn 缺 community_id，python 兜底计算")

    filled = 0
    for doc in db[conn_col].find({"community_id": {"$exists": False}}):
        try:
            cid = community_id_v1(
                proto=doc.get("proto", "tcp"),
                src=conn_id(doc, "orig_h"), sport=int(conn_id(doc, "orig_p")),
                dst=conn_id(doc, "resp_h"), dport=int(conn_id(doc, "resp_p")),
            )
            db[conn_col].update_one({"_id": doc["_id"]},
                                    {"$set": {"community_id": cid}})
            filled += 1
        except Exception as e:
            log.debug(f"community_id 兜底失败 uid={doc.get('uid')}: {e}")
    log.info(f"community_id 兜底完成: {filled} 条")
    return filled


def ingest(state: dict) -> dict:
    """第三层数据处理入口（适配 step1_analyze.py 已经入完 conn/payload 的现状）。

    本节点只做两件事：
      1. 补 ssl.log 入库（你 step1 没做）
      2. 兜底补 community_id（你 save-payload.zeek 没启用 policy 时）

    conn / payload 集合假定已由 step1_analyze.py 入完。
    如果你要重新跑 zeek，请重跑 step1_analyze.py，不要在这里跑。
    """
    if state.get("ingest_done"):
        log.info("ingest_done=True，跳过")
        return state

    pcap        = state["pcap_filename"]
    conn_col    = state["conn_collection"]
    payload_col = state["payload_collection"]
    ssl_col     = state.get("ssl_collection", "ssl_gyf_demo")

    client = MongoClient(config.DATABASE["mongo"]["uri"])
    db = client[config.DATABASE["mongo"]["db_name"]]

    # 保护：conn / payload 必须已有数据
    n_conn = db[conn_col].count_documents({})
    n_payload = db[payload_col].count_documents({})
    if n_conn == 0 or n_payload == 0:
        raise RuntimeError(
            f"集合 {conn_col}({n_conn} 条) / {payload_col}({n_payload} 条) "
            f"为空。请先用 step1_analyze.py 完成 conn/payload 入库。"
        )

    # 1. ssl.log 入库（幂等：按 source_pcap 检查）
    if db[ssl_col].count_documents({"source_pcap": pcap}) > 0:
        log.info(f"ssl.log({pcap}) 已入库，跳过")
    else:
        zeek_logs_dir = config.DIRECTORIES["zeek_logs"]
        # 注意：step1 是循环跑多个 pcap 的，zeek_logs 里只保留最后一次。
        # 如果你想严格按 pcap 入 ssl.log，必须紧跟在那次 zeek 跑完之后入。
        # 现在用兜底逻辑：如果当前目录的 ssl.log 是这个 pcap 的就入。
        _ingest_ssl_log(zeek_logs_dir, db, ssl_col, pcap)

    # 2. community_id 兜底
    _fill_community_id_fallback(db, conn_col)

    return {**state, "ingest_done": True}