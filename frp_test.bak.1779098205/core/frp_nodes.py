"""
frp_nodes.py — FRP 测试工作流的 10 个节点实现

节点列表：
    1. ingest_pcap         读 pcap，提取 flows
    2. kb_retrieval        向量化 flows + RAG 召回 frp 文档
    3. doc_pop             弹出文档队列首项
    4. doc_gate            LLM 决策 ENTER/DONE
    5. doc_enter           加载子 skill 队列
    6. skill_pop           弹出 skill 队列首项
    7. skill_gate          LLM 决策 EXECUTE/SKIP/DONE
    8. skill_enter         执行 skill 的 detection_workflow
    9. stage_summary       本文档阶段总结
    10. final_verdict      最终判定
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from core.frp_state import FRPTestState
from core.pcap_to_flows import pcap_to_flows
from core.frp_kb import FRPKnowledgeBase, build_target_vector
from core.llm_wrapper import (
    llm_doc_gate, llm_skill_gate, llm_judge_step, llm_final_verdict
)

logger = logging.getLogger(__name__)


# ───────────── 全局 KB 实例（懒加载）─────────────
_KB_INSTANCE = None

def _get_kb() -> FRPKnowledgeBase:
    global _KB_INSTANCE
    if _KB_INSTANCE is None:
        kb_dir = os.environ.get(
            "FRP_KB_DIR",
            str(Path(__file__).parent.parent / "knowledge_base" / "frp")
        )
        _KB_INSTANCE = FRPKnowledgeBase(kb_dir)
        _KB_INSTANCE.load()
    return _KB_INSTANCE


# ───────────── 工具调用辅助 ─────────────

def _call_pcap_tool(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """根据 tool_name 调用对应的 pcap_tools 工具"""
    from core import pcap_tools
    tool = getattr(pcap_tools, tool_name, None)
    if tool is None:
        return {"ok": False, "error": f"未知工具: {tool_name}"}
    try:
        return tool.invoke(params)
    except Exception as e:
        return {"ok": False, "error": f"工具调用异常: {e}"}


def _resolve_params(params: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """把 detection_workflow 里的 $xxx 占位符替换成真实值"""
    resolved = {}
    for k, v in params.items():
        if isinstance(v, str) and v.startswith("$"):
            key = v[1:]
            resolved[k] = state.get(key, "")
        else:
            resolved[k] = v
    resolved.setdefault("pcap_path", state.get("pcap_path", ""))
    return resolved


def _is_hard_fingerprint(skill_id: str, tool_name: str, result: Dict[str, Any]) -> bool:
    """判断是否命中 hard fingerprint"""
    if skill_id == "frp_quic" and result.get("has_frp_alpn"):
        return True
    if skill_id == "frp_ws" and result.get("has_frp_websocket_path"):
        return True
    if tool_name == "pcap_byte_pattern_search" and result.get("match_count", 0) > 0:
        return True
    return False


# ═══════════════════════════ 节点实现 ═══════════════════════════

def node_ingest_pcap(state: FRPTestState) -> Dict[str, Any]:
    """读取 pcap 转 flows"""
    pcap_path = state.get("pcap_path", "")
    if not pcap_path or not Path(pcap_path).exists():
        logger.error(f"pcap 路径无效: {pcap_path}")
        return {"flows": []}
    
    logger.info(f"[ingest_pcap] 读取 {pcap_path}")
    flows = pcap_to_flows(pcap_path)
    logger.info(f"[ingest_pcap] 提取 {len(flows)} 条流")
    
    # 推断 frpc / frps：取流量最大的会话，发起方为 frpc
    frpc_ip, frps_ip, suspect_port = "", "", 0
    if flows:
        # 按总字节数排序，找最大的会话
        top = max(flows, key=lambda f: f.get("orig_bytes", 0) + f.get("resp_bytes", 0))
        # 推断：orig 是发起方 = frpc，resp 是 frps（标准 frp 拓扑）
        frpc_ip = top.get("id.orig_h", "")
        frps_ip = top.get("id.resp_h", "")
        suspect_port = top.get("id.resp_p", 0)
        # 但如果 resp_bytes >> orig_bytes，说明 resp 端发的多，可能是反向（内网穿透特征）
        # 实际上 frpc 主动连出，但数据是 frps→frpc 进入，再 frpc→frps 回传
        # 这里保守起见保留原始判断
        logger.info(f"[ingest_pcap] 推断 frpc={frpc_ip}, frps={frps_ip}, port={suspect_port}")
    
    return {
        "flows": flows,
        "frpc_ip": frpc_ip,
        "frps_ip": frps_ip,
        "suspect_port": suspect_port,
        "evidence": [],
        "stage_summaries": []
    }


def node_kb_retrieval(state: FRPTestState) -> Dict[str, Any]:
    """向量化 flows + RAG 召回"""
    flows = state.get("flows", [])
    if not flows:
        logger.warning("[kb_retrieval] flows 为空，跳过")
        return {"doc_queue": [], "recall_results": []}
    
    # 1. 计算 target_vector
    try:
        target_vec = build_target_vector(flows)
        logger.info(f"[kb_retrieval] target_vector 维度: {target_vec.shape}")
    except ImportError as e:
        logger.error(f"[kb_retrieval] 无法 import rag.py: {e}")
        kb = _get_kb()
        return {
            "target_vector": None,
            "recall_results": [{"doc_id": d["doc_id"], "score": 0.5, "doc_content": d} for d in kb.docs],
            "doc_queue": [{"doc_id": d["doc_id"], "score": 0.5, "doc_content": d} for d in kb.docs]
        }
    
    # 2. 召回
    kb = _get_kb()
    recall = kb.recall_docs(target_vec, top_k=3)
    logger.info(f"[kb_retrieval] 召回 {len(recall)} 个文档:")
    for r in recall:
        logger.info(f"  - {r['doc_id']} (score={r['score']:.4f})")
    
    return {
        "target_vector": target_vec.tolist() if hasattr(target_vec, "tolist") else target_vec,
        "recall_results": recall,
        "doc_queue": recall
    }


def node_flow_recall(state: FRPTestState) -> Dict[str, Any]:
    """
    ★ 新节点：补齐相关流
    
    现状：kb_retrieval 只算了批次级向量召回文档；批次里可能"缺"几类典型流：
        - TLS Client Hello（决定 JA3 / SNI）
        - 心跳包（决定周期与 bursts）
        - 文件上传/下载流（决定流量方向反转）
    
    本节点：
        1. 识别 anchor flow（控制连接，最大字节 + 最长时长）
        2. 主动调用 pcap 工具补齐三类典型流
        3. 把结果挂在 state.recalled_flows 供 skill_enter 引用
        4. 校准 frpc_ip / frps_ip / suspect_port（比 ingest_pcap 阶段更准）
    """
    flows = state.get("flows", [])
    pcap_path = state.get("pcap_path", "")
    if not flows:
        return {"recalled_flows": {}}
    
    logger.info(f"[flow_recall] 输入 {len(flows)} 条流，开始补齐相关流")
    
    # 1. 识别 anchor flow
    def _score(f):
        return f.get("duration", 0) * 1.0 + (f.get("orig_bytes", 0) + f.get("resp_bytes", 0)) / 1e5
    anchor = max(flows, key=_score)
    a_ip = anchor.get("id.orig_h")
    b_ip = anchor.get("id.resp_h")
    b_port = anchor.get("id.resp_p")
    logger.info(f"  anchor: {a_ip}:{anchor.get('id.orig_p')} -> {b_ip}:{b_port} ({anchor.get('proto')})  "
                f"orig={anchor.get('orig_bytes')}, resp={anchor.get('resp_bytes')}, dur={anchor.get('duration')}")
    
    # 2. 同 IP 对的所有流（含反向 / 短连接）
    related = [
        f for f in flows
        if {f.get("id.orig_h"), f.get("id.resp_h")} == {a_ip, b_ip}
    ]
    logger.info(f"  同 IP 对相关流: {len(related)}")
    
    recalled = {
        "anchor_flow": {
            "uid": anchor.get("uid"),
            "src": f"{a_ip}:{anchor.get('id.orig_p')}",
            "dst": f"{b_ip}:{b_port}",
            "proto": anchor.get("proto"),
            "duration": anchor.get("duration"),
            "orig_bytes": anchor.get("orig_bytes"),
            "resp_bytes": anchor.get("resp_bytes"),
        },
        "related_flows_count": len(related),
        "tls_handshake": None,
        "heartbeat": None,
        "data_streams": [],
    }
    
    # 3a. 补齐 TLS Client Hello
    try:
        tls = _call_pcap_tool("pcap_client_hello_details", {"pcap_path": pcap_path})
        if tls.get("ok") and tls.get("client_hellos"):
            ch_list = tls.get("client_hellos") if isinstance(tls.get("client_hellos"), list) else []
            recalled["tls_handshake"] = {
                "count": len(ch_list) if ch_list else 0,
                "snis": tls.get("snis", [])[:5],
                "ja3_hashes": tls.get("ja3_hashes", [])[:3],
                "sample_client_hello": ch_list[0] if ch_list else None,
            }
            logger.info(f"  ✓ 补齐 TLS Client Hello: {len(ch_list)} 次，SNI={tls.get('snis', [])[:2]}")
        else:
            logger.info(f"  - 无 TLS Client Hello")
    except Exception as e:
        logger.warning(f"  TLS handshake 补齐异常: {e}")
    
    # 3b. 补齐心跳（基于 anchor IP 对）
    try:
        hb = _call_pcap_tool("pcap_detect_heartbeat", {
            "pcap_path": pcap_path,
            "src_ip": a_ip,
            "dst_ip": b_ip,
        })
        if hb.get("ok") and hb.get("is_periodic"):
            recalled["heartbeat"] = {
                "period_sec": hb.get("estimated_period_sec"),
                "burst_count": hb.get("burst_count"),
                "packets_per_burst": hb.get("avg_packets_per_burst"),
            }
            logger.info(f"  ✓ 补齐心跳: 周期={hb.get('estimated_period_sec')}s, bursts={hb.get('burst_count')}")
        else:
            logger.info(f"  - 未检出周期性心跳")
    except Exception as e:
        logger.warning(f"  心跳补齐异常: {e}")
    
    # 3c. 补齐数据流候选（大字节会话）
    data_candidates = sorted(
        [f for f in flows if f.get("orig_bytes", 0) > 10240 or f.get("resp_bytes", 0) > 10240],
        key=lambda f: max(f.get("orig_bytes", 0), f.get("resp_bytes", 0)),
        reverse=True
    )[:5]
    recalled["data_streams"] = [{
        "uid": f.get("uid"),
        "src": f"{f.get('id.orig_h')}:{f.get('id.orig_p')}",
        "dst": f"{f.get('id.resp_h')}:{f.get('id.resp_p')}",
        "orig_bytes": f.get("orig_bytes"),
        "resp_bytes": f.get("resp_bytes"),
    } for f in data_candidates]
    if data_candidates:
        logger.info(f"  ✓ 补齐数据流候选: {len(data_candidates)} 个")
    
    return {
        "recalled_flows": recalled,
        # 校准这三个推断字段
        "frpc_ip": a_ip,
        "frps_ip": b_ip,
        "suspect_port": b_port,
    }


def _build_flow_summary(state: dict) -> dict:
    """构造流量特征摘要，供 LLM 给 sub_skills 排序时使用"""
    flows = state.get("flows", [])
    recalled = state.get("recalled_flows", {}) or {}
    
    proto_count = {"tcp": 0, "udp": 0}
    dst_ports = []
    for f in flows:
        p = (f.get("proto") or "").lower()
        proto_count[p] = proto_count.get(p, 0) + 1
        dp = f.get("id.resp_p")
        if dp:
            dst_ports.append(dp)
    
    return {
        "flow_count": len(flows),
        "proto_dist": proto_count,
        "dst_ports": sorted(set(dst_ports)),
        "has_tls": bool(recalled.get("tls_handshake")),
        "tls_snis": (recalled.get("tls_handshake") or {}).get("snis", []),
        "has_heartbeat": bool(recalled.get("heartbeat")),
        "heartbeat_period": (recalled.get("heartbeat") or {}).get("period_sec"),
        "anchor_dst_port": (recalled.get("anchor_flow") or {}).get("dst", "").split(":")[-1],
        "data_stream_count": len(recalled.get("data_streams", [])),
    }


def node_doc_pop(state: FRPTestState) -> Dict[str, Any]:
    queue = state.get("doc_queue", [])
    if not queue:
        return {"current_doc": None}
    return {"current_doc": queue[0], "doc_queue": queue[1:]}


def node_doc_gate(state: FRPTestState) -> Dict[str, Any]:
    current = state.get("current_doc")
    if not current:
        return {"doc_decision": "DONE"}
    
    doc_content = current.get("doc_content", current)
    decision = llm_doc_gate(state.get("evidence", []), doc_content)
    logger.info(f"[doc_gate] {current.get('doc_id')} → {decision}")
    return {"doc_decision": decision}


def node_doc_enter(state: FRPTestState) -> Dict[str, Any]:
    """
    加载当前文档的子 skill 队列。
    ★ 改造：不再用 JSON 写死的 priority，改成调用 LLM/启发式根据流量特征排序。
    """
    current = state.get("current_doc")
    if not current:
        return {"skill_queue": []}
    
    doc_content = current.get("doc_content", current)
    sub_skills = doc_content.get("sub_skills", [])
    
    if not sub_skills:
        logger.info("[doc_enter] 文档无 sub_skills")
        return {"skill_queue": []}
    
    # 1. 构造流量摘要喂给 LLM
    flow_summary = _build_flow_summary(state)
    logger.info(f"[doc_enter] flow_summary: {flow_summary}")
    
    # 2. 用 LLM/启发式排序，返回 skill_id 列表
    from core.llm_wrapper import llm_rank_skills
    ordered_ids = llm_rank_skills(flow_summary, sub_skills)
    logger.info(f"[doc_enter] LLM 排序结果: {ordered_ids}")
    
    # 3. 按排序结果填充队列
    kb = _get_kb()
    skill_queue = []
    for rank, sid in enumerate(ordered_ids):
        skill = kb.get_skill(sid)
        if skill:
            skill_queue.append({
                "skill_id": sid,
                "priority": rank,
                "content": skill
            })
    
    logger.info(f"[doc_enter] 最终 skill_queue: {[s['skill_id'] for s in skill_queue]}")
    return {"skill_queue": skill_queue}


def node_skill_pop(state: FRPTestState) -> Dict[str, Any]:
    queue = state.get("skill_queue", [])
    if not queue:
        return {"current_skill": None}
    return {"current_skill": queue[0], "skill_queue": queue[1:]}


def node_skill_gate(state: FRPTestState) -> Dict[str, Any]:
    skill = state.get("current_skill")
    current_doc = state.get("current_doc")
    if not skill or not current_doc:
        return {"skill_decision": "DONE"}
    
    doc_content = current_doc.get("doc_content", current_doc)
    decision = llm_skill_gate(state.get("evidence", []), doc_content, skill["content"])
    logger.info(f"[skill_gate] {skill['skill_id']} → {decision}")
    return {"skill_decision": decision}


def node_skill_enter(state: FRPTestState) -> Dict[str, Any]:
    """执行 skill 的 detection_workflow"""
    skill = state.get("current_skill")
    if not skill:
        return {}
    
    workflow = skill["content"].get("detection_workflow", [])
    new_evidence = []
    
    for step in workflow:
        tool_name = step.get("tool")
        if not tool_name:
            continue
        
        params = _resolve_params(dict(step.get("params", {})), state)
        result = _call_pcap_tool(tool_name, params)
        
        expect = step.get("expect", "")
        judgment = llm_judge_step(tool_name, expect, result)
        is_hard = _is_hard_fingerprint(skill["skill_id"], tool_name, result)
        
        evidence_entry = {
            "skill_id": skill["skill_id"],
            "step_id": step.get("step_id"),
            "step_name": step.get("step_name"),
            "tool": tool_name,
            "tool_result": result,
            "expected": expect,
            "judgment": judgment,
            "hard_fingerprint": is_hard,
            "summary": f"[{skill['skill_id']}] step{step.get('step_id')} {step.get('step_name')}: {judgment}" + (" (HARD)" if is_hard else "")
        }
        new_evidence.append(evidence_entry)
        logger.info(f"  {evidence_entry['summary']}")
        
        if judgment == "FAIL" and step.get("next_on_fail") == "abort":
            logger.info(f"  [skill_enter] {skill['skill_id']} step{step.get('step_id')} 失败且配置 abort")
            break
    
    existing = state.get("evidence", [])
    return {"evidence": existing + new_evidence}


def node_stage_summary(state: FRPTestState) -> Dict[str, Any]:
    current_doc = state.get("current_doc", {})
    doc_id = current_doc.get("doc_id", "unknown") if isinstance(current_doc, dict) else "unknown"
    
    evidence = state.get("evidence", [])
    pass_count = sum(1 for e in evidence if e.get("judgment") == "PASS")
    hard_count = sum(1 for e in evidence if e.get("hard_fingerprint"))
    
    summary = f"[stage_summary] 文档 {doc_id} 审计完毕：PASS={pass_count}, HARD={hard_count}"
    logger.info(summary)
    
    existing = state.get("stage_summaries", [])
    return {"stage_summaries": existing + [summary]}


def node_final_verdict(state: FRPTestState) -> Dict[str, Any]:
    evidence = state.get("evidence", [])
    audited = list(set(e.get("skill_id", "") for e in evidence if e.get("skill_id")))
    
    verdict = llm_final_verdict(evidence, audited)
    logger.info(f"[final_verdict] {json.dumps(verdict, ensure_ascii=False, indent=2)}")
    
    return {"final_verdict": verdict}


# ─────────── 路由函数 ───────────

def route_after_doc_pop(state: FRPTestState):
    return "doc_gate" if state.get("current_doc") else "final_verdict"


def route_after_doc_gate(state: FRPTestState):
    return "doc_enter" if state.get("doc_decision") == "ENTER" else "final_verdict"


def route_after_skill_gate(state: FRPTestState):
    d = state.get("skill_decision", "")
    if d == "EXECUTE":
        return "skill_enter"
    if d == "SKIP":
        return "skill_pop" if state.get("skill_queue") else "stage_summary"
    return "stage_summary"  # DONE


def route_after_skill_enter(state: FRPTestState):
    return "skill_pop" if state.get("skill_queue") else "stage_summary"


def route_after_stage_summary(state: FRPTestState):
    return "doc_pop" if state.get("doc_queue") else "final_verdict"
