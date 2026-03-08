from tools.getflow import export_flows_tool
import time
import logging
from state import State
from langchain_core.messages import ToolMessage

def update_state(state: State):
    """
    更新状态，从最后一个工具结果中提取数据
    
    Args:
        state: 当前状态
        
    Returns:
        dict: 更新后的状态
    """
    logging.info("进入 end_state 节点")
    messages = state.get("messages", [])
    
    # 直接取最后一个消息
    if not messages:
        return state
        
    last_message = messages[-1]
    
    # 检查是否为 ToolMessage 并提取 tool_result
    if isinstance(last_message, ToolMessage):
        tool_result = last_message.tool_result
            
        # 如果工具返回的是字典，提取相关信息
        if isinstance(tool_result, dict):
            # 处理 report_suspicious_traffic_tool 的结果
            if "is_suspicious" in tool_result:
                state["is_suspicious"] = tool_result["is_suspicious"]
                state["suspicious_flows_start"] = tool_result.get("suspicious_flows_start", -1)
                state["suspicious_flows_end"] = tool_result.get("suspicious_flows_end", -1)
                state["suspicious_flows"] = state["flows"][state["suspicious_flows_start"]:state["suspicious_flows_end"]+1]
                    
                # 打印提取的信息
                print(f"[update_state] 提取到工具结果:")
                print(f"  - is_suspicious: {state['is_suspicious']}")
                print(f"  - suspicious_flows_start: {state['suspicious_flows_start']}")
                print(f"  - suspicious_flows_end: {state['suspicious_flows_end']}")
    
    return state