from langgraph.graph import StateGraph, START, END
from state import State
import pprint
from utils.utils import process_message_content

from nodes.preliminary_judgment import preliminary_judgment
from nodes.init import init
from nodes.tool_use import tool_node
from nodes.tool_use import route_tools
from nodes.update_state import update_state
from nodes.print_state import print_state

from utils.config import ENABLE_THINK_OUTPUT, ENABLE_PRINT



graph_builder = StateGraph(State)

# add node
graph_builder.add_node("init", init)
graph_builder.add_node("preliminary_judgment", preliminary_judgment)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("update_state", update_state)
graph_builder.add_node("print_state", print_state)


# add edge
graph_builder.add_edge(START, "init")
graph_builder.add_edge("init", "preliminary_judgment")
# graph_builder.add_edge("preliminary_judgment", END)
graph_builder.add_conditional_edges(
    "preliminary_judgment",
    route_tools,
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "update_state")
graph_builder.add_edge("update_state", "print_state")
graph_builder.add_edge("print_state", END)

# 编译
graph = graph_builder.compile()

def stream_graph_updates(
    user_input: str, 
    skip: int = 100, 
    limit: int = 30, 
    output_filename: str = "dataset.json",
    conn_collection: str = "conn_gyf_demo",
    payload_collection: str = "payload_gyf_demo"
    ):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}], "skip": skip, "limit": limit, "conn_collection": conn_collection, "payload_collection": payload_collection, "output_filename": output_filename}):
        for value in event.values():
            if value is None or "messages" not in value:
                continue
            if ENABLE_PRINT:
                print("\n=== 模型传出的内容 ===")
                pprint.pprint(value["messages"])
                print("====================\n")
            # 处理助手回复内容
            processed_content = process_message_content(value["messages"][-1].content, ENABLE_THINK_OUTPUT)
            print("Assistant:", processed_content)

if __name__ == "__main__":
    # 测试入口
    user_input = """我在进行流量检测,现在我将给你数条json格式的流量数据,这些流量数据中可能存在多个攻击,也可能不存在攻击。
    攻击的类型为”内网穿透“”渗透隐蔽信道“”加密通信“。
    我希望你能找到最靠前的攻击流量的编号和这个攻击包含连续的几条流量,并返回其攻击类型。
    我将向你介绍流量数据中各参数的含义,如果有存在我未告知含义的参数,请通知我：
    1.ts:Unix格式时间戳,表示连接开始的时间。
    2.uid:连接的唯一标识符,用于追踪特定的连接会话。
    3.id.orig_h:源IP地址。
    4.id.orig_p:源端口号。
    5.id.resp_h:目标IP地址。
    6.id.resp_p:目标端口号。
    7.proto:协议类型,如TCP、UDP等。
    8.duration:流量持续时间,单位为秒。
    9.orig_bytes:源端发送的字节数。
    10.resp_bytes:目标端发送的字节数。
    11.conn_state:连接状态,
        'S0'表示'只有SYN,没有SYN+ACK',
        'S1'表示'连接已建立,但还没结束',
        'SF'表示'正常结束,有SYN-FIN',
        'REJ'表示'连接被拒绝(RST)',
        'S2'表示'有SYN+ACK,没有FIN',
        'S3'表示'有FIN,没有ACK',
        'RSTO'表示'发起方发送RST重置连接',
        'RSTR'表示'响应方发送RST重置连接',
        'OTH'表示'其他状态,未见SYN包'。
    12.local_orig:是否为本地发起的流量。
    13.local_resp:是否为本地接收的流量。
    14.missed_bytes:丢失的字节数。
    15.history:这是一个由字母组成的字符串,记录了数据包的交互顺序,其中大写是发起方发送的包,小写是响应方发送的包，
        'S'和's'表示'SYN(请求连接)',
        'H'和'h'表示'SYN+ACK(同意连接)',
        'F'和'f'表示'FIN(结束连接)',
        'A'和'a'表示'ACK(确认)',
        'D'和'd'表示'Data(有负载数据)',
        'R'和'r'表示'RST(重置/强制断开)',
        '^'表示'方向翻转'。
    16.orig_pkts:从源发送的数据包数量。
    17.orig_ip_bytes:从源发送的IP层字节数。
    18.resp_pkts:从目标发送的数据包数量。
    19.resp_ip_bytes:从目标发送的IP层字节数。
    20.ip_proto:IP协议号,6表示TCP。
    21.ts_date:格式化的时间戳,便于人类阅读。
    22.mongo_id:MongoDB 文档的唯一ID。
    23.batch_index:该流在本次截取的流量数据中的编号。
    24.packet_count_captured:该流在整个pcap流量包中的编号。
    25.stream_payload_hex:流负载的十六进制表示。
    26.stream_payload_decoded:流负载的解码表示。
    """
    print("User: " + user_input)
    stream_graph_updates(user_input, skip=60, limit=30)