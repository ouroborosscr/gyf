from langgraph.graph import StateGraph, START, END
from layer3.tls13.state import TLS13State
from layer3.tls13.nodes.ingest import ingest
from layer3.tls13.nodes.pick_session import pick_session
from layer3.tls13.nodes.locate_dispatch import locate_dispatch, route_after_dispatch
from layer3.tls13.nodes.case1_inline import case1_inline
from layer3.tls13.nodes.case2_late_capture import case2_late_capture
from layer3.tls13.nodes.case3_zeek_split import case3_zeek_split
from layer3.tls13.nodes.verify_extracted import verify_extracted, route_after_verify
from layer3.tls13.nodes.threat_intel import threat_intel
from layer3.tls13.nodes.payload_analyse import payload_analyse
from layer3.tls13.nodes.mark_incomplete import mark_incomplete

g = StateGraph(TLS13State)

g.add_node("ingest", ingest)
g.add_node("pick_session", pick_session)
g.add_node("dispatch", locate_dispatch)
g.add_node("case1", case1_inline)
g.add_node("case2", case2_late_capture)
g.add_node("case3", case3_zeek_split)
g.add_node("verify", verify_extracted)
g.add_node("threat_intel", threat_intel)
g.add_node("payload_analyse", payload_analyse)
g.add_node("mark_incomplete", mark_incomplete)

g.add_edge(START, "ingest")
g.add_edge("ingest", "pick_session")
g.add_edge("pick_session", "dispatch")

# 调度器 → case1/case2/case3 / 兜底 / 进入情报库
g.add_conditional_edges("dispatch", route_after_dispatch, {
    "case1": "case1",
    "case2": "case2",
    "case3": "case3",
    "threat_intel": "threat_intel",
    "mark_incomplete": "mark_incomplete",
})

# 每个 case → verify
g.add_edge("case1", "verify")
g.add_edge("case2", "verify")
g.add_edge("case3", "verify")

# verify → 回到 dispatch / 进 threat_intel / 走 incomplete
g.add_conditional_edges("verify", route_after_verify, {
    "dispatch": "dispatch",
    "threat_intel": "threat_intel",
    "mark_incomplete": "mark_incomplete",
})

# 情报库 → 载荷分析；incomplete 也直接进载荷分析（兜底）
g.add_edge("threat_intel", "payload_analyse")
g.add_edge("mark_incomplete", "payload_analyse")
g.add_edge("payload_analyse", END)

tls13_graph = g.compile()



# 对接现有主图
# 在主图的 route_tools（或 update_state 后）增加一个分支：当 preliminary_judgment 判定可疑流是 TLS1.3 时，把控制权交给 tls13_graph：
# 在 graph.py 主图里
# from layer3.tls13.graph import tls13_graph

# def enter_tls13(state):
#     sub_out = tls13_graph.invoke({
#         "pcap_filename": "example.pcap",
#         "conn_collection": state["conn_collection"],
#         "payload_collection": state["payload_collection"],
#         "ssl_collection": "ssl_gyf_demo",
#         "target_flow_idx": state["suspicious_flows_start"],  # 暂由你确定
#     })
#     # 把子图结论合回主图（client_hello / server_hello / extract_ok / packet_incomplete）
#     return {**state, "tls13_result": sub_out}

# graph_builder.add_node("enter_tls13", enter_tls13)