from typing_extensions import TypedDict
from typing import Annotated, Optional
from langgraph.graph.message import add_messages

class TLS13State(TypedDict, total=False):
    # 来自上层的输入
    messages: Annotated[list, add_messages]
    pcap_filename: str                  # 当前要分析的 pcap
    conn_collection: str                # conn_gyf_demo
    payload_collection: str             # payload_gyf_demo
    ssl_collection: str                 # ssl_gyf_demo（新增）
    target_flow_idx: int                # 当前要审计的 TLS1.3 会话编号（暂由用户给）

    # 入库阶段产物
    ingest_done: bool

    # 当前流的元信息（pick_session 填）
    current_flow: dict                  # conn_gyf_demo 中的一条记录
    current_payload: dict               # payload_gyf_demo 中的一条记录
    current_ssl: Optional[dict]         # ssl_gyf_demo 中的一条记录（可能没有）
    community_id: Optional[str]

    # 三种 case 的尝试结果
    case_tried: list                    # ["case1", "case2", ...]
    client_hello_hex: Optional[str]     # 提取到的 ClientHello 原始 hex
    server_hello_hex: Optional[str]
    extracted_from: Optional[str]       # "case1" / "case2" / "case3_split" / "case3_psk"
    extracted_meta: dict                # SNI、cipher、resumed、session_id 等

    # 模型判断结果
    extract_ok: bool                    # verify_extracted 写入
    extract_confidence: float
    skip_remaining_cases: bool          # 模型判断"够好了，不必再试"

    # 兜底
    packet_incomplete: bool             # 三种 case 都失败