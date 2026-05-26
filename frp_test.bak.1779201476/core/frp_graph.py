"""
frp_graph.py — FRP 测试工作流的独立 LangGraph

完全独立于 gyf 现有 graph.py，不修改任何现有节点。
"""

from langgraph.graph import StateGraph, START, END

from core.frp_state import FRPTestState
from core.frp_nodes import (
    node_ingest_pcap, node_kb_retrieval, node_flow_recall,
    node_doc_pop, node_doc_gate, node_doc_enter,
    node_skill_pop, node_skill_gate, node_skill_enter,
    node_stage_summary, node_final_verdict,
    route_after_doc_pop, route_after_doc_gate,
    route_after_skill_gate, route_after_skill_enter,
    route_after_stage_summary,
)


def build_frp_test_graph():
    """构建 FRP 测试工作流"""
    g = StateGraph(FRPTestState)
    
    g.add_node("ingest_pcap", node_ingest_pcap)
    g.add_node("kb_retrieval", node_kb_retrieval)
    g.add_node("flow_recall", node_flow_recall)   # ★ Feature 1
    g.add_node("doc_pop", node_doc_pop)
    g.add_node("doc_gate", node_doc_gate)
    g.add_node("doc_enter", node_doc_enter)        # ★ Feature 2（内部已改用 LLM 排序）
    g.add_node("skill_pop", node_skill_pop)
    g.add_node("skill_gate", node_skill_gate)
    g.add_node("skill_enter", node_skill_enter)
    g.add_node("stage_summary", node_stage_summary)
    g.add_node("final_verdict", node_final_verdict)
    
    # 主流：ingest → kb_retrieval → flow_recall → doc_pop
    g.add_edge(START, "ingest_pcap")
    g.add_edge("ingest_pcap", "kb_retrieval")
    g.add_edge("kb_retrieval", "flow_recall")       # ★ 新增
    g.add_edge("flow_recall", "doc_pop")            # ★ 新增
    
    # 外层 doc 循环
    g.add_conditional_edges("doc_pop", route_after_doc_pop, {
        "doc_gate": "doc_gate", "final_verdict": "final_verdict"
    })
    g.add_conditional_edges("doc_gate", route_after_doc_gate, {
        "doc_enter": "doc_enter", "final_verdict": "final_verdict"
    })
    g.add_edge("doc_enter", "skill_pop")
    
    # 内层 skill 循环
    g.add_edge("skill_pop", "skill_gate")
    g.add_conditional_edges("skill_gate", route_after_skill_gate, {
        "skill_enter": "skill_enter",
        "skill_pop": "skill_pop",
        "stage_summary": "stage_summary"
    })
    g.add_conditional_edges("skill_enter", route_after_skill_enter, {
        "skill_pop": "skill_pop",
        "stage_summary": "stage_summary"
    })
    
    # 阶段总结回到外层
    g.add_conditional_edges("stage_summary", route_after_stage_summary, {
        "doc_pop": "doc_pop", "final_verdict": "final_verdict"
    })
    
    g.add_edge("final_verdict", END)
    
    return g.compile()


# 单例
frp_test_graph = None

def get_frp_test_graph():
    global frp_test_graph
    if frp_test_graph is None:
        frp_test_graph = build_frp_test_graph()
    return frp_test_graph
