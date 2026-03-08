from langchain_core.messages import SystemMessage
import json
import logging
from state import State
from llm import llm
from tools.tools import tools
from utils.config import ENABLE_THINK_OUTPUT, ENABLE_PRINT

llm_with_tools = llm.bind_tools(tools)

def preliminary_judgment(state: State):
    logging.info("进入 preliminary_judgment 节点")
    # 1. 获取流量数据
    flows_data = state.get("flows", [])
    # 2. 将数据转换为 JSON 字符串形式，避免直接拼接 list 和 dict 报错
    flows_str = json.dumps(flows_data, ensure_ascii=False, indent=2)
    # 3. 构建上下文消息
    context_message = SystemMessage(content=f"以下是待分析的流量数据详情：\n{flows_str}")
    # 4. 组合消息列表
    input_messages = [context_message] + state["messages"]

    if ENABLE_PRINT:
        print("\n=== 传给模型的内容 (部分展示) ===")
        print(f"System Message (Data): {str(flows_str)[:200]}...") 
        print(f"User Message: {state['messages'][-1].content}")
        print("====================\n")
    
    # 5. 调用模型 (普通对话模式)
    response = llm_with_tools.invoke(input_messages)
    
    return {"messages": [response]}