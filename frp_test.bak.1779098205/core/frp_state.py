"""
frp_state.py — FRP 测试工作流专用 State

独立于现有 state.py（不污染主 graph）。继承部分通用字段，新增双层队列、证据、判定等字段。
"""

from typing_extensions import TypedDict
from typing import Annotated, List, Dict, Any, Optional
from langgraph.graph.message import add_messages


class FRPTestState(TypedDict, total=False):
    # ────── 输入 ──────
    pcap_path: str                                  # 待测 pcap 路径
    flows: List[Dict[str, Any]]                     # 从 pcap 提取的 Zeek-style 流列表
    expected_label: Optional[str]                   # ground truth（用于评估，可选）
    
    # ────── 召回 ──────
    target_vector: Any                              # 当前流量的特征向量（np.ndarray）
    recall_results: List[Dict[str, Any]]            # RAG 召回结果 [{doc_id, score, doc_content}, ...]
    recalled_flows: Dict[str, Any]                  # ★ 新增：补齐的相关流（client hello / 心跳 / 上传流等）
    
    # ────── 外层文档队列 ──────
    doc_queue: List[Dict[str, Any]]                 # 待审计的文档队列
    current_doc: Optional[Dict[str, Any]]           # 当前处理的文档
    doc_decision: str                               # ENTER / DONE
    
    # ────── 内层 skill 队列 ──────
    skill_queue: List[Dict[str, Any]]               # 当前文档下的 skill 队列
    current_skill: Optional[Dict[str, Any]]         # 当前处理的 skill
    skill_decision: str                             # EXECUTE / SKIP / DONE
    
    # ────── 证据 + 判定 ──────
    evidence: List[Dict[str, Any]]                  # 累积的证据片段
    stage_summaries: List[str]                      # 每个文档结束的阶段总结
    final_verdict: Optional[Dict[str, Any]]         # 最终判定
    
    # ────── 流量上下文（推断出来的） ──────
    frpc_ip: str                                    # 推断的 frpc 端 IP
    frps_ip: str                                    # 推断的 frps 端 IP
    suspect_port: int                               # 主要可疑端口
    
    # ────── 兼容现有 State（让代码能在主 graph 集成时通用） ──────
    messages: Annotated[list, add_messages]
