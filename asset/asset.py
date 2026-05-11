import csv
import json
from neo4j import GraphDatabase

class IPAssetImporter:
    def __init__(self, driver):
        self.driver = driver

    def import_from_csv(self, file_path):
        # 使用 'utf-8-sig' 自动过滤掉 Windows CSV 可能带有的 BOM 头 (\ufeff)
        with open(file_path, mode='r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            
            with self.driver.session() as session:
                # 建议先为 ip 属性创建唯一约束（只需执行一次），这样查询效率极高
                session.run("CREATE CONSTRAINT ip_unique_idx IF NOT EXISTS FOR (i:IP) REQUIRE i.ip IS UNIQUE")

                for row in reader:
                    # 此时 row.get('ip') 获取到的键名已经不再带有 \ufeff
                    ip_value = row.get('ip')
                    port_key = row.get('port')
                    
                    if not ip_value or not port_key:
                        print(f"跳过无效行: {row}")
                        continue
                    
                    # 将这一行的完整原始数据转为 JSON 字符串
                    row_json = json.dumps(row, ensure_ascii=False)
                    
                    # 构造 Cypher 语句：
                    # 1. MERGE 确保 IP 节点唯一
                    # 2. SET n.name 方便在 Neo4j UI 界面直接看到 IP 名字
                    # 3. SET n.`{port}` 动态设置端口属性名，值为该行 JSON
                    cypher_query = f"""
                    MERGE (n:IP {{ip: $ip}})
                    SET n.name = $ip
                    SET n.`{port_key}` = $json_data
                    """
                    
                    session.run(cypher_query, ip=ip_value, json_data=row_json)
                    print(f"入库成功: IP={ip_value}, 端口属性=`{port_key}`")

# ================= 使用示例 =================
if __name__ == "__main__":
    # 【重点】这里的连接方式，请直接参考或引用您 rag.py 中的 driver 实例
    URI = "bolt://localhost:7865"
    AUTH = ("neo4j", "gyf_password")
    
    # 实例化 driver (与 rag.py 保持一致)
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    importer = IPAssetImporter(driver)
    try:
        # 替换为您实际的 csv 文件路径
        importer.import_from_csv("1777260550.csv")
    except Exception as e:
        print(f"入库过程中发生错误: {e}")
    finally:
        driver.close()