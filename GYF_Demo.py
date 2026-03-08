import os
import sys
import logging
import time
from datetime import datetime
from pymongo import MongoClient

# --- 1. 环境与路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from utils import config
    from tools.zeek import analyze_pcap_tool
    from graph import stream_graph_updates
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [GYF_Demo] - %(levelname)s - %(message)s')

ANALYSIS_PROMPT = """我在进行流量检测...（此处省略提示词，保持不变）..."""

class GYFController:
    def __init__(self):
        self.mongo_uri = config.DATABASE["mongo"]["uri"]
        self.history_db_name = config.GYF_SETTINGS.get("history_db_name", "gyf_history")
        self.status_col_name = config.GYF_SETTINGS.get("status_collection", "global_status")
        self.zeek_db_name = config.DATABASE["mongo"]["db_name"]
        
        self.client = MongoClient(self.mongo_uri)
        self.history_db = self.client[self.history_db_name]
        self.zeek_db = self.client[self.zeek_db_name]

    def get_control_doc(self):
        """
        辅助函数：专门获取主控记录
        使用 {'current_id': {'$exists': True}} 确保不会读到报警记录
        """
        status_col = self.history_db[self.status_col_name]
        return status_col.find_one({"current_id": {"$exists": True}})

    def init_database(self):
        collection = self.history_db[self.status_col_name]
        default_hop = config.GYF_SETTINGS.get("hop_length", 30)
        default_back = config.GYF_SETTINGS.get("back", 5)
        
        # 为了防止误删报警记录，我们只重置主控记录
        # 1. 尝试查找主控记录
        existing_control = collection.find_one({"current_id": {"$exists": True}})
        
        init_record = {
            "flow_count": 0,
            "hop_length": default_hop,
            "back": default_back,
            "current_id": 0,
            "status": "initialized",
            "updated_at": datetime.now()
        }

        if existing_control:
            # 如果存在，更新它
            collection.update_one({"_id": existing_control["_id"]}, {"$set": init_record})
        else:
            # 如果不存在，清空集合并创建（为了纯净环境，也可以选择保留这行 delete_many）
            collection.delete_many({}) 
            collection.insert_one(init_record)
            
        logging.info(f"数据库初始化完成: {init_record}")

    def initialize_flow(self, pcap_file="clean.pcap"):
        logging.info("Step 1: 开始运行 Zeek 流量分析...")
        conn_col = "conn_gyf_demo"
        payload_col = "payload_gyf_demo"

        # ... Zeek 运行代码保持不变 ...
        try:
            analyze_pcap_tool.invoke({
                "pcap_filename": pcap_file,
                "conn_collection": conn_col,
                "payload_collection": payload_col
            })
        except Exception as e:
            logging.error(f"Zeek 运行失败: {e}")
            return

        generated_flow_count = self.zeek_db[conn_col].count_documents({})
        logging.info(f"检测到生成的流数量: {generated_flow_count}")

        # 这里的 update_one 也要加过滤条件
        self.history_db[self.status_col_name].update_one(
            {"current_id": {"$exists": True}},
            {"$set": {"flow_count": generated_flow_count, "status": "flow_ready"}}
        )

    def start_analysis_loop(self):
        logging.info("Step 2: 开始 AI 分析循环...")
        status_col = self.history_db[self.status_col_name]

        while True:
            # 【修改点 1】 使用精确查找获取主控记录
            status_doc = self.get_control_doc()
            if not status_doc:
                logging.error("未找到 global_status 主控记录，退出。")
                break

            current_id = int(status_doc.get("current_id", 0))
            hop_length = int(status_doc.get("hop_length", 30))
            back = int(status_doc.get("back", 5))
            flow_count = int(status_doc.get("flow_count", 0))

            logging.info(f"当前进度检测: current_id={current_id} / total={flow_count}")

            if current_id >= flow_count:
                logging.info(f"分析循环结束：当前编号 {current_id} 已达到或超过总数 {flow_count}")
                break

            logging.info(f">>> 调用 Graph 分析 batch: start={current_id}, limit={hop_length}")
            
            try:
                stream_graph_updates(
                    user_input=ANALYSIS_PROMPT,
                    skip=current_id,
                    limit=hop_length,
                    conn_collection=config.GYF_SETTINGS.get("conn_collection", "conn_gyf_demo"),
                    payload_collection=config.GYF_SETTINGS.get("payload_collection", "payload_gyf_demo"),
                    output_filename="dataset.json"
                )
                
                # 给数据库一点写入缓冲时间
                time.sleep(1) 
                
                # 【修改点 2】 再次精确查找，对比 ID 变化
                new_status = self.get_control_doc()
                new_id = int(new_status.get("current_id", 0))
                
                if new_id == current_id:
                    # 这种情况通常意味着：Graph 运行成功了，但 AI 没有发现攻击，所以没有更新 DB。
                    # 这不是错误，而是正常的"滑动窗口"前进。
                    step_size = hop_length - back
                    logging.info(f"[正常步进] 未检测到攻击，手动推进窗口: {current_id} -> {current_id + step_size}")
                    
                    status_col.update_one(
                        {"current_id": {"$exists": True}}, # 确保只更新主控记录
                        {"$inc": {"current_id": step_size}}
                    )
                else:
                    logging.info(f"[攻击跳跃] 检测到攻击，Graph 已自动更新进度: {current_id} -> {new_id}")
            
            except Exception as e:
                logging.error(f"Graph 执行出错: {e}")
                break

    def run(self):
        print("====== GYF Demo Start ======")
        self.init_database()
        self.initialize_flow()
        self.start_analysis_loop()
        print("====== GYF Demo Finished ======")

if __name__ == "__main__":
    controller = GYFController()
    controller.run()   