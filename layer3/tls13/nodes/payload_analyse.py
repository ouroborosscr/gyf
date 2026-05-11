# layer3/tls13/nodes/payload_analyse.py

def payload_analyse(state: dict) -> dict:
    """
    Payload 的拉取和提取逻辑已经下放至 case1/2/3 节点内部。
    此节点当前仅作为占位符透传 state，确保 graph.py 能正常 import 并流转。
    """
    # 什么都不改，直接返回原状态
    return state