import pandas as pd
from pymongo import MongoClient
import logging
import sys

# ================= 配置区域 =================

MONGO_URI = "mongodb://admin:gyf424201@localhost:62015/"
DB_NAME = "zeek_analysis"
SOURCE_COLLECTION = "conn_test_auto"      # 原始数据
TARGET_COLLECTION = "conn_labels_only"    # 【新】只存 ID 和 Label

CSV_PATH = "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"

# 【参数】(沿用之前成功的参数)
TIME_OFFSET = 42216.0
TOLERANCE_SEC = 3.0
DURATION_DIFF_LIMIT = 15.0

# ===========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class LabelSaver:
    def __init__(self):
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            self.src_col = self.db[SOURCE_COLLECTION]
            self.tgt_col = self.db[TARGET_COLLECTION]
        except Exception as e:
            logging.error(f"❌ MongoDB 连接失败: {e}")
            sys.exit(1)
            
        self.proto_map = {6: "tcp", 17: "udp", 1: "icmp"}

    def prepare_target(self):
        # 可选：清空旧数据
        # self.tgt_col.drop()
        # 创建索引方便查询
        self.tgt_col.create_index("uid")
        self.tgt_col.create_index("label")

    def load_csv(self):
        logging.info(f"📂 读取 CSV: {CSV_PATH} ...")
        try:
            df = pd.read_csv(CSV_PATH)
            df.rename(columns=lambda x: x.strip(), inplace=True)
            
            # 基础清洗
            df['Dst Port'] = pd.to_numeric(df['Dst Port'], errors='coerce').fillna(-1).astype(int)
            
            # 时间对齐
            try:
                df['ts_csv'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %H:%M:%S").astype('int64') // 10**9
            except:
                df['ts_csv'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %H:%M:%S.%f").astype('int64') // 10**9
            
            df['ts_target'] = df['ts_csv'] + TIME_OFFSET
            
            # 辅助字段
            df['dur_csv'] = pd.to_numeric(df['Flow Duration'], errors='coerce').fillna(0) / 1_000_000.0
            df['proto_str'] = df['Protocol'].map(self.proto_map).fillna("unknown")
            
            df = df.sort_values('ts_target')
            return df
            
        except Exception as e:
            logging.error(f"❌ CSV 错误: {e}")
            sys.exit(1)

    def process_batch(self, mongo_docs, csv_df):
        if not mongo_docs: return 0
        
        # 建立映射以获取原始 uid
        doc_map = {d['_id']: d for d in mongo_docs}
        
        zeek_df = pd.DataFrame(mongo_docs)
        
        # 提取匹配字段
        if 'id.resp_p' in zeek_df.columns:
            zeek_df['dst_port'] = zeek_df['id.resp_p']
        elif 'id' in zeek_df.columns:
            zeek_df['dst_port'] = zeek_df['id'].apply(lambda x: x.get('resp_p') if isinstance(x, dict) else None)
        else:
            return 0
            
        if 'proto' not in zeek_df.columns: zeek_df['proto'] = 'unknown'
        
        zeek_df['ts'] = pd.to_numeric(zeek_df['ts'], errors='coerce')
        zeek_df['dst_port'] = pd.to_numeric(zeek_df['dst_port'], errors='coerce').fillna(-1).astype(int)
        zeek_df['proto'] = zeek_df['proto'].astype(str).str.lower()
        
        zeek_df = zeek_df.dropna(subset=['ts']).sort_values('ts')
        
        if zeek_df.empty: return 0
        
        # 核心匹配
        try:
            merged = pd.merge_asof(
                zeek_df, csv_df,
                left_on='ts', right_on='ts_target',
                left_by=['dst_port', 'proto'],
                right_by=['Dst Port', 'proto_str'],
                direction='nearest', tolerance=TOLERANCE_SEC
            )
        except:
            return 0
            
        # 筛选
        merged['duration'] = pd.to_numeric(merged['duration'], errors='coerce').fillna(0)
        merged['dur_diff'] = (merged['duration'] - merged['dur_csv']).abs()
        
        valid = merged[
            (merged['Label'].notna()) & 
            (merged['dur_diff'] <= DURATION_DIFF_LIMIT)
        ]
        
        if valid.empty: return 0
        
        # --- 只构建精简文档 ---
        docs_to_insert = []
        for _, row in valid.iterrows():
            original = doc_map.get(row['_id'])
            if original:
                # 只保留 _id, uid, label
                lite_doc = {
                    "_id": original['_id'],  # 保持 MongoDB ID 一致
                    "label": row['Label']
                }
                
                # 如果原数据有 Zeek 的 uid，也带上
                if 'uid' in original:
                    lite_doc['uid'] = original['uid']
                    
                docs_to_insert.append(lite_doc)
                
        if docs_to_insert:
            try:
                self.tgt_col.insert_many(docs_to_insert, ordered=False)
            except Exception:
                pass # 忽略重复键错误
                
        return len(docs_to_insert)

    def run(self):
        self.prepare_target()
        csv_df = self.load_csv()
        
        total = self.src_col.count_documents({})
        logging.info(f"📊 开始扫描 {total} 条记录...")
        
        # 【核心修复】移除 projection 参数 (即删除了中间的那个 {...} 字典)
        # 这样会读取所有字段，避免 id 和 id.resp_p 的冲突
        cursor = self.src_col.find({}).batch_size(10000)
        
        batch = []
        count = 0
        for doc in cursor:
            batch.append(doc)
            if len(batch) >= 10000:
                count += self.process_batch(batch, csv_df)
                batch = []
                logging.info(f"⏳ 已存储标签: {count}")
                
        if batch:
            count += self.process_batch(batch, csv_df)
            
        logging.info(f"✅ 完成！共存入 {count} 条标签到集合: {TARGET_COLLECTION}")

if __name__ == "__main__":
    LabelSaver().run()