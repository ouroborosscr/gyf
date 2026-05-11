import logging
from pymongo import MongoClient
from utils import config
from layer3.tls13.db.conn_dao import conn_id  # 引入 conn_id 适配动态字段

log = logging.getLogger(__name__)


def pick_session(state: dict) -> dict:
    """根据 target_flow_idx 从 conn 集合捞当前要审计的 TLS1.3 会话。

    会话编号语义：按 ts 升序后的第 N 条（0-based），与 layer1/layer2 的"序号"约定一致。
    一并把对应的 ssl 记录读出来，写进 state。
    (注意：payload 获取已下放至具体 case 节点，不再此处预加载)
    """
    idx = state["target_flow_idx"]
    conn_col    = state["conn_collection"]
    ssl_col     = state.get("ssl_collection", "ssl_gyf_demo")

    client = MongoClient(config.DATABASE["mongo"]["uri"])
    db = client[config.DATABASE["mongo"]["db_name"]]

    flows = list(db[conn_col].find().sort("ts", 1).skip(idx).limit(1))
    if not flows:
        raise RuntimeError(f"conn 集合 {conn_col} 中找不到第 {idx} 条流量")
    flow = flows[0]

    ssl_row = db[ssl_col].find_one({"uid": flow["uid"]})

    log.info(
        "pick_session: idx=%s uid=%s %s:%s -> %s:%s state=%s resumed=%s ver=%s sni=%s",
        idx, flow.get("uid"),
        conn_id(flow, "orig_h"), conn_id(flow, "orig_p"),
        conn_id(flow, "resp_h"), conn_id(flow, "resp_p"),
        flow.get("conn_state"),
        (ssl_row or {}).get("resumed"),
        (ssl_row or {}).get("version"),
        (ssl_row or {}).get("server_name"),
    )

    return {
        **state,
        "current_flow":         flow,
        "current_payload":      {},  # payload 获取下放，留空防止破坏原数据结构
        "current_ssl":          ssl_row,
        "community_id":         flow.get("community_id"),
        "case_tried":           [],
        "client_hello_hex":     None,
        "server_hello_hex":     None,
        "extracted_from":       None,
        "extracted_meta":       {},
        "extract_ok":           False,
        "extract_confidence":   0.0,
        "skip_remaining_cases": False,
        "packet_incomplete":    False,
    }