import os
import sys
from pymongo import MongoClient

# --- 1. 动态导入配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from utils import config
    print("成功加载 utils.config")
except ImportError:
    print("错误: 无法加载 utils.config，请检查路径")
    sys.exit(1)

# --- 2. 获取数据库参数 ---
MONGO_URI = config.DATABASE["mongo"]["uri"]
DB_NAME = config.DATABASE["mongo"]["db_name"]
# 这里请确认你要处理的集合名称
COLLECTION_NAME = "3_1_conn" 

def fix_schema_and_index():
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        col = db[COLLECTION_NAME]
        
        print(f"正在连接数据库: {DB_NAME} -> {COLLECTION_NAME}")
        print("注意：正在执行字段重命名 (id.resp_p -> id_resp_p)...")
        
        # 使用 Aggregation Pipeline 进行批量更新
        # 它可以引用自身字段值，即使字段名里带点
        pipeline = [
            {
                "$set": {
                    # 将旧字段的值赋给新字段 (下划线命名)
                    "id_resp_p": { "$getField": "id.resp_p" },
                    "id_orig_p": { "$getField": "id.orig_p" },
                    "id_orig_h": { "$getField": "id.orig_h" },
                    "id_resp_h": { "$getField": "id.resp_h" },
                    
                    # 确保其他关键字段保留
                    "proto": "$proto",
                    "ts": "$ts",
                    "orig_pkts": "$orig_pkts",
                    "resp_pkts": "$resp_pkts",
                    "duration": "$duration",
                    "uid": "$uid"
                }
            },
            {
                # 删除旧的带点字段，给数据库“瘦身”
                "$unset": ["id.resp_p", "id.orig_p", "id.orig_h", "id.resp_h"] 
            }
        ]

        # 执行更新 (这可能需要一些时间，取决于数据量)
        result = col.update_many({}, pipeline)
        print(f"字段修复完成！匹配文档: {result.matched_count}, 修改文档: {result.modified_count}")

        # --- 重建索引 ---
        print("正在重建索引 (id_resp_p + ts)...")
        # 删除旧索引
        col.drop_indexes()
        
        # 建立新索引：目的端口 + 时间
        col.create_index([
            ("id_resp_p", 1), 
            ("ts", 1)
        ])
        print("索引重建完成！现在的数据库结构已支持高速查询。")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    fix_schema_and_index()