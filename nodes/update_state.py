from tools.getflow import export_flows_tool
import time
import logging
from state import State
from langchain_core.messages import ToolMessage
from pymongo import MongoClient
import os
import sys

# --- 配置引入部分 ---
try:
    from utils import config
    MONGO_URI = config.DATABASE["mongo"]["uri"]
    HISTORY_DB_NAME = config.GYF_SETTINGS.get("history_db_name", "gyf_history")
    STATUS_COL_NAME = config.GYF_SETTINGS.get("status_collection", "global_status")
except ImportError:
    MONGO_URI = "mongodb://localhost:27017/"
    HISTORY_DB_NAME = "gyf_history"
    STATUS_COL_NAME = "global_status"

def update_state(state: State):
    """
    更新状态，从最后一个工具结果中提取数据
    1. 将可疑流量作为新文档插入数据库 (新建 _id)
    2. 更新主控文档的 current_id
    
    Args:
        state: 当前状态
        
    Returns:
        dict: 更新后的状态
    """
    logging.info("进入 update_state 节点")
    messages = state.get("messages", [])
    
    if not messages:
        return state
        
    last_message = messages[-1]
    
    if isinstance(last_message, ToolMessage):
        tool_result = last_message.tool_result
            
        if isinstance(tool_result, dict):
            if "is_suspicious" in tool_result:
                # --- 1. 提取信息到 State ---
                state["is_suspicious"] = tool_result["is_suspicious"]
                state["suspicious_flows_start"] = tool_result.get("suspicious_flows_start", -1)
                state["suspicious_flows_end"] = tool_result.get("suspicious_flows_end", -1)
                
                # 提取流量切片
                if state["is_suspicious"] and state["suspicious_flows_start"] != -1:
                    local_s = state["suspicious_flows_start"]
                    local_e = state["suspicious_flows_end"]
                    state["suspicious_flows"] = state["flows"][local_s : local_e+1]
                else:
                    state["suspicious_flows"] = []
                
                # --- 2. 数据库操作 ---
                if state["is_suspicious"]:
                    try:
                        client = MongoClient(MONGO_URI)
                        db = client[HISTORY_DB_NAME]
                        collection = db[STATUS_COL_NAME]
                        
                        # 计算全局位置
                        current_skip = state.get("skip", 0)
                        global_start = current_skip + state["suspicious_flows_start"]
                        global_end = current_skip + state["suspicious_flows_end"]
                        
                        # A. 插入一条全新的记录 (Alert Log)
                        # 这是一个新文档，会自动生成新的 _id
                        alert_doc = {
                            "type": "suspicious_alert", # 标记类型，方便区分
                            "is_suspicious": True,
                            "suspicious_flows_start": global_start,
                            "suspicious_flows_end": global_end,
                            "suspicious_flows": state["suspicious_flows"],
                            "created_at": time.time(),
                            "from_batch_skip": current_skip
                        }
                        insert_result = collection.insert_one(alert_doc)
                        print(f"[update_state] 新增可疑记录 _id: {insert_result.inserted_id}")

                        # B. 更新"总的那条"控制记录 (Control Record)
                        # 我们假设总记录是包含 "current_id" 字段的那条
                        # 更新 current_id 到本次攻击流量之后
                        next_id = global_end + 1
                        
                        update_result = collection.update_one(
                            {"current_id": {"$exists": True}}, # 过滤条件：查找包含 current_id 的主记录
                            {
                                "$set": {
                                    "current_id": next_id,
                                    "updated_at": time.time()
                                }
                            }
                        )
                        
                        if update_result.modified_count > 0:
                            print(f"[update_state] 主控记录已更新 current_id -> {next_id}")
                        else:
                            print(f"[update_state] 警告: 未找到主控记录 (current_id字段) 进行更新")
                        
                        client.close()
                        
                    except Exception as e:
                        print(f"[update_state] MongoDB 操作失败: {e}")

    return state