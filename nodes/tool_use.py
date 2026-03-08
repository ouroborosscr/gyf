from tools.tools import tools
from langchain_core.messages import ToolMessage
import json
import logging
from state import State
from langgraph.graph import END 

class BasicToolNode:
    """运行上一个 AIMessage 中请求的工具的节点。"""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools} # 创建一个 tool.name -> tool 的映射字典

    def __call__(self, inputs: dict):
        logging.info("进入 tool_use 节点")
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("在输入中未找到消息 (No message found in input)")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    # ensure_ascii=False 可以让 JSON 中的中文正常显示，而不是转义符
                    content=json.dumps(tool_result, ensure_ascii=False),
                    tool_result = tool_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# 实例化工具节点
# 注意：原代码是 tools=[tool]，但 tool 变量未定义，这里假设你是想使用导入的 tools 列表
tool_node = BasicToolNode(tools=tools)

def route_tools(
    state: State,
):
    """
    用于条件边（conditional_edge）：
    如果最后一条消息包含工具调用（tool calls），则路由到 ToolNode（工具节点）。
    否则，路由到结束（END）。
    """
    logging.info("进入 route_tools 节点")
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"在 tool_edge 的输入状态中未找到消息: {state}")
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END