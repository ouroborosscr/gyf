import os
import sys
import logging
import time
from datetime import datetime
from pymongo import MongoClient

# --- 1. 环境与路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# ================= 配置区 =================
TARGET_DATE = "3_1"         # 目标数据集日期后缀，例如 "3_1", "3_2"
FORCE_RESET = False         # 是否强制重置进度，从 0 开始重新测试
# ==========================================

try:
    from utils import config
    
    # 【关键修改】在导入 Graph 及其节点之前，动态覆写目标数据表名！
    # 这样底层 update_state 节点在保存可疑流量始末位置时，就会自动写进对应的测试状态表
    config.GYF_SETTINGS["status_collection"] = f"test_status_{TARGET_DATE}"
    
    from graph import graph
    from utils.utils import process_message_content
    from utils.config import ENABLE_THINK_OUTPUT
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [GYF_Test] - %(levelname)s - %(message)s')

ANALYSIS_PROMPT = """我在进行流量检测,现在我将给你数条json格式的流量数据,这些流量数据中可能存在多个攻击,也可能不存在攻击。
攻击的类型为”内网穿透“”渗透隐蔽信道“”加密通信“。
我希望你能找到最靠前的攻击流量的编号和这个攻击包含连续的几条流量,并返回其攻击类型。"""

class GYFTestController:
    def __init__(self, target_date=TARGET_DATE, reset_progress=FORCE_RESET):
        self.mongo_uri = config.DATABASE["mongo"]["uri"]
        self.history_db_name = config.GYF_SETTINGS.get("history_db_name", "gyf_history")
        self.zeek_db_name = config.DATABASE["mongo"]["db_name"]
        
        self.conn_col_name = f"{target_date}_conn"
        self.payload_col_name = f"{target_date}_payload"
        self.status_col_name = f"test_status_{target_date}"
        self.reset_progress = reset_progress

        self.client = MongoClient(self.mongo_uri)
        self.history_db = self.client[self.history_db_name]
        self.zeek_db = self.client[self.zeek_db_name]

    def get_control_doc(self):
        """专门获取用于记录滑动窗口进度的主要控制记录"""
        status_col = self.history_db[self.status_col_name]
        return status_col.find_one({"current_id": {"$exists": True}})

    def init_database(self):
        collection = self.history_db[self.status_col_name]
        default_hop = config.GYF_SETTINGS.get("hop_length", 30)
        default_back = config.GYF_SETTINGS.get("back", 5)
        
        if self.conn_col_name not in self.zeek_db.list_collection_names():
            logging.error(f"未找到目标集合: {self.conn_col_name}")
            sys.exit(1)
            
        total_flow_count = self.zeek_db[self.conn_col_name].count_documents({})
        logging.info(f"目标测试数据集 [{self.conn_col_name}] 共有 {total_flow_count} 条流量记录。")

        existing_control = self.get_control_doc()
        init_record = {
            "flow_count": total_flow_count,
            "hop_length": default_hop,
            "back": default_back,
            "current_id": 0,
            "status": "testing",
            "updated_at": datetime.now()
        }

        if existing_control and not self.reset_progress:
            current_id = existing_control.get("current_id", 0)
            logging.info(f"✅ 断点重连：将从第 {current_id} 条流继续测试。")
            collection.update_one({"_id": existing_control["_id"]}, {"$set": {"flow_count": total_flow_count, "updated_at": datetime.now()}})
        else:
            if self.reset_progress:
                logging.warning("⚠️ 收到强制重置指令，清空历史进度。")
            collection.delete_many({}) 
            collection.insert_one(init_record)

    def start_analysis_loop(self):
        logging.info("Step: 开始自动化批量测试...")
        status_col = self.history_db[self.status_col_name]

        while True:
            status_doc = self.get_control_doc()
            if not status_doc:
                break

            current_id = int(status_doc.get("current_id", 0))
            hop_length = int(status_doc.get("hop_length", 30))
            back = int(status_doc.get("back", 5))
            flow_count = int(status_doc.get("flow_count", 0))

            if current_id >= flow_count:
                logging.info(f"🎉 自动化测试完成！({self.conn_col_name})")
                break
            
            logging.info(f"== 进度: [ {current_id} / {flow_count} ] ==")

            try:
                inputs = {
                    "messages": [{"role": "user", "content": ANALYSIS_PROMPT}],
                    "skip": current_id,
                    "limit": hop_length,
                    "conn_collection": self.conn_col_name,
                    "payload_collection": self.payload_col_name,
                    "output_filename": f"test_dataset_{self.conn_col_name}.json"
                }
                
                ai_full_response = ""
                is_suspicious = False
                
                for event in graph.stream(inputs):
                    for node_name, value in event.items():
                        if not value or not isinstance(value, dict):
                            continue
                            
                        # 1. 捕获状态流中是否判定为异常
                        if "is_suspicious" in value:
                            is_suspicious = value.get("is_suspicious", False)
                            
                        # 2. 提取并拼接模型的完整输出文本
                        if "messages" in value and value["messages"]:
                            last_msg = value["messages"][-1]
                            if getattr(last_msg, "type", "") == "ai" and last_msg.content:
                                # 完整保留换行、<think>标签及内部工具调用文本
                                raw_text = str(last_msg.content)
                                ai_full_response += f"{raw_text}"
                                
                                # 保持控制台的实时打印不变
                                processed_content = process_message_content(last_msg.content, ENABLE_THINK_OUTPUT)
                                print(f"Assistant ({node_name}):", processed_content)

                # ================= 3. 核心合并与落库逻辑 =================
                if is_suspicious:
                    # 此时 update_state 节点已经生成了一条 type="suspicious_alert" 的记录
                    # 我们找到这条记录，把刚刚拼接的完整大模型输出用 $set 追加进去
                    result = status_col.update_one(
                        {"type": "suspicious_alert", "from_batch_skip": current_id},
                        {"$set": {
                            "model_raw_output": ai_full_response.strip(),
                            "batch_limit": hop_length,
                            "timestamp": datetime.now()
                        }}
                    )
                    if result.modified_count > 0:
                        logging.info("📝 模型的完整思考过程已成功合并至悬疑报警记录中。")
                    else:
                        logging.warning("⚠️ 未能找到对应的 suspicious_alert 记录来合并日志。")
                else:
                    # 如果没有发现攻击，单独存一条 normal_log，保留大模型判定正常的思考过程
                    log_doc = {
                        "type": "normal_log", 
                        "from_batch_skip": current_id,
                        "batch_limit": hop_length,
                        "is_suspicious": False,
                        "model_raw_output": ai_full_response.strip(),
                        "timestamp": datetime.now()
                    }
                    status_col.insert_one(log_doc)
                    logging.info("📝 本批次（未发现异常）模型输出已归档。")
                # ========================================================

                time.sleep(1) 
                
                # 检查进度是否需要步进
                new_status = self.get_control_doc()
                new_id = int(new_status.get("current_id", 0))
                
                if new_id == current_id:
                    step_size = hop_length - back
                    logging.info(f"|> [正常步进] 未发现攻击，窗口前进: {current_id} -> {current_id + step_size}")
                    status_col.update_one({"current_id": {"$exists": True}}, {"$inc": {"current_id": step_size}})
                else:
                    logging.info(f"|> [捕获报警] 智能体识别到攻击并已自动截断更新进度: {current_id} -> {new_id}")

            except KeyboardInterrupt:
                logging.info("\n🛑 收到中断信号 (Ctrl+C)，退出...")
                break
            except Exception as e:
                # 捕获到 Pydantic 校验错误或其他幻觉报错
                logging.error(f"❌ 测试执行出错 (batch_skip={current_id}): {e}")
                
                # 强制将窗口步进，跳过这个引发模型幻觉的毒性批次，而不是直接 break 退出
                step_size = hop_length - back
                logging.info(f"|> [容错跳过] 强制窗口前进，避免卡死: {current_id} -> {current_id + step_size}")
                status_col.update_one({"current_id": {"$exists": True}}, {"$inc": {"current_id": step_size}})
                
                # 稍微等待一下，继续下一个循环
                time.sleep(2)
                continue

    def run(self):
        print(f"====== GYF Test Start (Target: {self.conn_col_name}) ======")
        self.init_database()
        self.start_analysis_loop()
        print("====== GYF Test Finished ======")

if __name__ == "__main__":
    controller = GYFTestController()
    controller.run()