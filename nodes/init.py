from tools.getflow import export_flows_tool
import time
import logging
from state import State

def init(state: State):
    logging.info("进入 init 节点")
    current_ts = time.time()
    skip = state.get("skip", 100) 
    limit = state.get("limit", 30)
    conn_collection = state.get("conn_collection", "conn_gyf_demo")
    payload_collection = state.get("payload_collection", "payload_gyf_demo")
    
    
    # 调用工具获取数据
    flows_data = export_flows_tool.invoke({
        "skip": skip, 
        "limit": limit, 
        "output_filename": "debug_dataset.json",
        "conn_collection": conn_collection,
        "payload_collection": payload_collection
    })
    
    # 必须 return 更新的内容以更新 State
    return {
        "ts": current_ts,
        "skip": skip,
        "limit": limit,
        "flows": flows_data
    }