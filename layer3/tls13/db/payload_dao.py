"""
读取 payload 集合的辅助（适配 step1_analyze.py 入库出来的 schema B）：
  - 每条记录 = 一个有载荷的 TCP packet
  - 字段: uid, id.*, is_orig, len, payload(hex), payload_decoded,
          ts, ts_date, source_pcap
  - 如果你重跑入库（路 B），save-payload.zeek 已经加了 seq，
    本函数会自动用 seq 排序；没有时退化为 ts 排序。
"""
import logging

log = logging.getLogger(__name__)


def fetch_stream_hex(db, payload_col: str, uid: str) -> dict:
    """按 uid 拉该流所有 packet，按方向 + seq(或 ts) 排序拼接成整流 hex。

    返回:
        {"orig":      <orig 方向整流 hex>,
         "resp":      <resp 方向整流 hex>,
         "combined":  orig + resp,
         "n_packets": 总包数,
         "sort_key":  实际用了哪个字段排序（"seq" / "ts"）}
    """
    col = db[payload_col]
    pkts = list(col.find({"uid": uid}))
    if not pkts:
        return {"orig": "", "resp": "", "combined": "",
                "n_packets": 0, "sort_key": None}

    sort_key = "seq" if any("seq" in p for p in pkts) else "ts"
    if sort_key == "ts":
        log.debug(f"payload 缺 seq 字段，回退到 ts 排序（uid={uid}）")

    orig_pkts = sorted([p for p in pkts if p.get("is_orig") is True],
                       key=lambda p: p.get(sort_key, 0))
    resp_pkts = sorted([p for p in pkts if p.get("is_orig") is False],
                       key=lambda p: p.get(sort_key, 0))

    orig_hex = "".join(p.get("payload", "") for p in orig_pkts)
    resp_hex = "".join(p.get("payload", "") for p in resp_pkts)

    return {
        "orig":      orig_hex,
        "resp":      resp_hex,
        "combined":  orig_hex + resp_hex,
        "n_packets": len(pkts),
        "sort_key":  sort_key,
    }