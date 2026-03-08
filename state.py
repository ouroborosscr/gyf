from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    ts: float #时间戳
    skip: int #跳过的流量数量
    back: int #无事回滚的流量数量
    limit: int #流量列表限制数量
    flows: list #流量列表
    is_suspicious: bool #是否存在可疑流量
    suspicious_flows_start: int #可疑流量列表开始索引
    suspicious_flows_end: int #可疑流量列表结束索引
    suspicious_flows: list #可疑流量列表

    output_filename: str #输出文件名
    conn_collection: str #连接记录集合名
    payload_collection: str #负载记录集合名
    