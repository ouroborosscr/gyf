"""
独立测试 layer3 TLS1.3 子图，不对接主图。
假定上游已经判定 example.pcap 中目标会话是 TLS1.3。

用法示例：
    # 首次跑（会调用 zeek 入库 + 跑子图，flow_idx 由你给）
    python -m layer3.tls13.run_test --pcap example.pcap --flow-idx 12

    # 已经入过库，只想反复调试 case 逻辑
    python -m layer3.tls13.run_test --flow-idx 12 --use-existing

    # 先看看 mongo 里有哪些候选 TLS 流，再挑一个 flow-idx
    python -m layer3.tls13.run_test --list-candidates

    # 用真实 LLM 跑 verify_extracted（默认走 mock 启发式，方便无 key 调试）
    python -m layer3.tls13.run_test --flow-idx 12 --use-existing --real-llm
"""
import argparse
import json
import logging
import pprint
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 上（让 from utils / from layer3 都能 import）
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pymongo import MongoClient

from utils import config
from layer3.tls13.graph import tls13_graph


def setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - [TLS13Test] - %(levelname)s - %(message)s",
    )


def _db():
    client = MongoClient(config.DATABASE["mongo"]["uri"])
    return client[config.DATABASE["mongo"]["db_name"]]


def clean_collections(conn_col: str, payload_col: str, ssl_col: str):
    """清空三个集合，强制重入库。"""
    db = _db()
    for c in (conn_col, payload_col, ssl_col):
        n = db[c].count_documents({})
        db[c].drop()
        logging.info(f"已清空集合 {c}（删除 {n} 条记录）")


def list_candidates(conn_col: str, ssl_col: str, limit: int = 30):
    """列出 mongo 里看起来像 TLS 的流，帮你挑 --flow-idx。
    优先列 ssl.log 里 version=TLSv1.3 的；没有就退化为 dst_port=443 的 conn 记录。
    """
    db = _db()
    print(f"\n==== TLS 候选会话（前 {limit} 条）====")
    print(f"{'idx':>4}  {'ts':>14}  {'src':22}  {'dst':22}  {'state':6}  {'version':10}  {'sni'}")
    cursor = db[conn_col].find().sort("ts", 1).limit(500)
    shown = 0
    for idx, doc in enumerate(cursor):
        if shown >= limit:
            break
        is_tls_port = int(doc.get("id.resp_p", 0)) in (443, 8443, 9443)
        ssl_row = db[ssl_col].find_one({"uid": doc["uid"]}) or {}
        version = ssl_row.get("version") or ""
        sni = ssl_row.get("server_name") or ""
        if not (is_tls_port or ssl_row):
            continue
        print(f"{idx:>4}  {doc.get('ts', 0):>14.2f}  "
              f"{doc.get('id.orig_h')+':'+str(doc.get('id.orig_p')):22}  "
              f"{doc.get('id.resp_h')+':'+str(doc.get('id.resp_p')):22}  "
              f"{doc.get('conn_state', ''):6}  "
              f"{version:10}  {sni}")
        shown += 1
    print()


def short(s, n=120):
    if not s:
        return "<empty>"
    return s if len(s) <= n else s[:n] + f"...({len(s)} chars total)"


def print_event(node_name: str, state: dict):
    """打印子图每个节点跑完之后的关键状态变化。"""
    if state is None:
        return
    print(f"\n--- [节点完成] {node_name} ---")
    keys = [
        ("ingest_done",          "ingest_done"),
        ("current_flow.uid",     lambda s: (s.get("current_flow") or {}).get("uid")),
        ("community_id",         "community_id"),
        ("case_tried",           "case_tried"),
        ("extracted_from",       "extracted_from"),
        ("extract_ok",           "extract_ok"),
        ("extract_confidence",   "extract_confidence"),
        ("skip_remaining_cases", "skip_remaining_cases"),
        ("packet_incomplete",    "packet_incomplete"),
    ]
    for label, key in keys:
        val = key(state) if callable(key) else state.get(key)
        if val is not None:
            print(f"  {label:22} = {val}")

    ch = state.get("client_hello_hex")
    sh = state.get("server_hello_hex")
    if ch:
        print(f"  client_hello_hex       = {short(ch)}")
    if sh:
        print(f"  server_hello_hex       = {short(sh)}")
    meta = state.get("extracted_meta")
    if meta:
        print(f"  extracted_meta         = "
              f"{json.dumps(meta, ensure_ascii=False, default=str)}")


def main():
    parser = argparse.ArgumentParser(
        description="独立测试 layer3 TLS1.3 子图（不对接主图）")
    parser.add_argument("--pcap", default="example.pcap",
                        help="要分析的 pcap 文件名（默认 example.pcap）")
    parser.add_argument("--flow-idx", type=int,
                        help="目标 TLS1.3 会话在 conn 集合中的编号（按 ts 升序，0-based）")
    parser.add_argument("--conn-col",    default="example_conn")
    parser.add_argument("--payload-col", default="example_payload")
    parser.add_argument("--ssl-col",     default="example_ssl")
    parser.add_argument("--use-existing", action="store_true",
                        help="跳过重入库，直接用 mongo 里已有的数据")
    parser.add_argument("--clean", action="store_true",
                        help="运行前清空集合，强制重入库")
    parser.add_argument("--list-candidates", action="store_true",
                        help="只列出疑似 TLS 流量，不运行子图")
    parser.add_argument("--real-llm", action="store_true",
                        help="使用真实 LLM 判断（默认 mock 启发式）")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # 模式 1：只列候选，不跑图
    if args.list_candidates:
        list_candidates(args.conn_col, args.ssl_col)
        return

    if args.flow_idx is None:
        parser.error("必须指定 --flow-idx（或先用 --list-candidates 看候选）")

    if args.clean:
        clean_collections(args.conn_col, args.payload_col, args.ssl_col)

    init_state = {
        "messages": [],
        "pcap_filename": args.pcap,
        "conn_collection": args.conn_col,
        "payload_collection": args.payload_col,
        "ssl_collection": args.ssl_col,
        "target_flow_idx": args.flow_idx,
        # 测试专用开关
        "_mock_llm": not args.real_llm,
        "ingest_done": args.use_existing,
    }

    print("=" * 64)
    print("Layer3 TLS1.3 子图独立测试")
    print(f"  pcap         = {args.pcap}")
    print(f"  flow_idx     = {args.flow_idx}")
    print(f"  use_existing = {args.use_existing}")
    print(f"  real_llm     = {args.real_llm}")
    print("=" * 64)

    final_state = None
    try:
        for event in tls13_graph.stream(init_state):
            for node_name, node_state in event.items():
                print_event(node_name, node_state)
                final_state = node_state
    except Exception as e:
        logging.exception(f"子图执行抛出异常：{e}")
        sys.exit(2)

    # 最终结论
    print("\n" + "=" * 64)
    print("最终结论：")
    if not final_state:
        print("  ❌ 子图没有产出任何 state")
        sys.exit(3)

    ch = final_state.get("client_hello_hex")
    sh = final_state.get("server_hello_hex")
    if ch and sh:
        print(f"  ✅ 提取成功（来自 {final_state.get('extracted_from')}）")
        print(f"     ClientHello 长度 = {len(ch)//2} 字节")
        print(f"     ServerHello 长度 = {len(sh)//2} 字节")
        meta = final_state.get("extracted_meta", {}) or {}
        if meta.get("sni"):
            print(f"     SNI            = {meta['sni']}")
        if "is_tls13" in meta:
            print(f"     CH is_tls13    = {meta['is_tls13']}")
        if "sh_is_tls13" in meta:
            print(f"     SH is_tls13    = {meta['sh_is_tls13']}")
        if meta.get("cipher_suite") is not None:
            print(f"     选中 cipher    = 0x{meta['cipher_suite']:04x}")
        if meta.get("resumed"):
            print(f"     resumed=True，原始握手 uid = {meta.get('original_uid')}")
        if meta.get("merged_from_uid"):
            print(f"     从拆分流合并   = {meta['merged_from_uid']}  "
                  f"seq_continuity={meta.get('seq_continuity')}")
    elif final_state.get("packet_incomplete"):
        print("  ⚠️  数据包不完整：三种 case 都没找到 ClientHello/ServerHello，"
              "已进入兜底载荷分析。")
        print(f"     case_tried = {final_state.get('case_tried')}")
    else:
        print("  ❌ 异常结束，最终 state：")
        pprint.pprint({k: v for k, v in final_state.items()
                       if k not in ("messages", "current_payload")})
    print("=" * 64)


if __name__ == "__main__":
    main()


# # 先看 mongo 里有哪些 TLS 候选，挑一个 idx
# python -m layer3.tls13.run_test --list-candidates

# # 假设你挑了 12 号会话
# python -m layer3.tls13.run_test --pcap example.pcap --flow-idx 12 --clean

# # 入完库之后，反复调试 case 逻辑只需要：
# python -m layer3.tls13.run_test --flow-idx 12 --use-existing -v


# # 先看看候选 TLS 会话（用现有的 2_true_conn）
# python -m layer3.tls13.run_test --list-candidates \
#        --conn-col 2_true_conn --ssl-col 2_true_ssl

# # 选一个 idx 跑（--use-existing 保证不会尝试重跑 zeek）
# python -m layer3.tls13.run_test \
#        --conn-col example_conn --payload-col example_payload --ssl-col example_ssl \
#        --flow-idx 0 --use-existing -v