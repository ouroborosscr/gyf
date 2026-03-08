import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import logging
from collections import Counter

# --- 配置 ---
MONGO_URI = "mongodb://admin:gyf424201@localhost:62015/"
DB_NAME = "zeek_analysis"
COLLECTION = "conn_test_auto"
CSV_PATH = "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"

# 限制扫描范围以提高速度
# 既然我们知道大概是 12小时 (43200秒) 左右
# 我们只关心 40000 到 46000 之间的 Offset
SEARCH_MIN = 36000 
SEARCH_MAX = 50000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def find_offset_by_histogram():
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION]
    
    # 1. 获取 Mongo 数据 (仅取特征鲜明的端口，如 80, 443, 22)
    # 这样可以减少计算量，提高信噪比
    target_ports = [80, 443, 22, 53, 8080]
    
    logging.info("正在加载 MongoDB 数据...")
    # 构造查询：端口在列表中
    # 注意兼容扁平/嵌套结构
    mongo_docs = []
    cursor = col.find({})
    
    for doc in cursor:
        try:
            p = None
            if 'id.resp_p' in doc: p = doc['id.resp_p']
            elif 'id' in doc: p = doc['id'].get('resp_p')
            
            if p in target_ports:
                mongo_docs.append({
                    'ts': doc['ts'],
                    'port': int(p),
                    'proto': doc.get('proto', 'unknown').lower()
                })
        except:
            continue
            
    logging.info(f"MongoDB 加载完成，提取关键端口记录: {len(mongo_docs)} 条")
    if not mongo_docs:
        print("❌ MongoDB 里没有 80/443/22 等常见端口的数据，无法进行直方图对齐。")
        return

    # 2. 加载 CSV 数据
    logging.info("正在加载 CSV 数据...")
    df = pd.read_csv(CSV_PATH)
    # 清洗列名
    df.rename(columns=lambda x: x.strip(), inplace=True)
    
    # 筛选 CSV (只看对应端口)
    df = df[df['Dst Port'].isin(target_ports)].copy()
    
    # 处理时间
    try:
        df['ts'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %H:%M:%S").astype('int64') // 10**9
    except:
        df['ts'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %H:%M:%S.%f").astype('int64') // 10**9
    
    # 协议映射
    proto_map = {6: 'tcp', 17: 'udp', 1: 'icmp'}
    df['proto'] = df['Protocol'].map(proto_map)
    
    logging.info(f"CSV 筛选完成，剩余 {len(df)} 条潜在匹配项")

    # 3. 暴力计算直方图
    logging.info("开始计算时间差直方图 (这可能需要一分钟)...")
    
    diff_counter = Counter()
    
    # 为了性能，我们将 DataFrame 转为字典列表
    csv_records = df[['ts', 'Dst Port', 'proto']].to_dict('records')
    
    # 创建一个简单的索引： Port -> List of timestamps
    csv_index = {}
    for row in csv_records:
        key = (row['Dst Port'], row['proto'])
        if key not in csv_index:
            csv_index[key] = []
        csv_index[key].append(row['ts'])

    # 核心循环
    for m_doc in mongo_docs:
        key = (m_doc['port'], m_doc['proto'])
        m_ts = m_doc['ts']
        
        if key in csv_index:
            for c_ts in csv_index[key]:
                # 计算差值
                diff = m_ts - c_ts
                
                # 只记录在合理范围内的 Diff
                if SEARCH_MIN <= diff <= SEARCH_MAX:
                    # 取整到秒
                    diff_counter[int(diff)] += 1

    # 4. 分析结果
    if not diff_counter:
        print("❌ 未找到任何落在此范围内的重叠。可能 Offset 超出了 36000-50000 范围。")
        return

    # 找到出现频率最高的 Diff
    best_offset, count = diff_counter.most_common(1)[0]
    
    print("\n" + "="*50)
    print(f"🔥 发现最佳 OFFSET: {best_offset} 秒")
    print(f"🔥 命中次数: {count}")
    print("="*50)
    
    # 打印前 5 名，看看是否聚焦
    print("\nTop 5 候选 Offset:")
    for offset, cnt in diff_counter.most_common(5):
        print(f"Offset: {offset} s | Hits: {cnt}")
        
    print("\n💡 验证:")
    print(f"CSV 时间 + {best_offset} = Zeek 时间")
    print(f"例如: 1518656400 (CSV Start) + {best_offset} = {1518656400 + best_offset}")

if __name__ == "__main__":
    find_offset_by_histogram()