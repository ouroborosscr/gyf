import os
import sys
import time
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md  # 🌟 新增：引入 markdownify
import logging
from pymongo import MongoClient
from datetime import datetime

# --- 配置与环境（兼容您的项目结构） ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from utils import config
    MONGO_URI = config.DATABASE["mongo"]["uri"]
    DB_NAME = config.GYF_SETTINGS.get("history_db_name", "gyf_history")
except ImportError:
    # 兜底本地配置，找不到 utils 也能正常跑
    MONGO_URI = "mongodb://localhost:27017/"
    DB_NAME = "gyf_history"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [XLab-Spider] - %(levelname)s - %(message)s')

# X-Lab 博客基础 URL
BASE_URL = "https://blog.xlab.qianxin.com"
# 我们将数据存入这个集合中，供 RAG 检索
COLLECTION_NAME = "xlab_threat_intel" 

def get_article_links():
    """获取博客主页及分页中的所有文章链接"""
    links = set()
    page = 1
    
    while True:
        url = f"{BASE_URL}/page/{page}/" if page > 1 else f"{BASE_URL}/"
        logging.info(f"正在扫描文章列表: {url}")
        
        try:
            # 伪装请求头
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logging.info("分页扫描结束或已无更多页面。")
                break
                
            soup = BeautifulSoup(response.text, 'html.parser')
            # 奇安信博客目前使用的是类似 Ghost 博客的结构，文章链接一般在 post-card 类中
            articles = soup.find_all('article')
            if not articles:
                break
                
            for article in articles:
                a_tag = article.find('a', href=True)
                if a_tag:
                    href = a_tag['href']
                    # 确保是文章路径而不是外链
                    if href.startswith('/') and len(href) > 2:
                        links.add(BASE_URL + href)
            page += 1
            time.sleep(1) # 礼貌延时
            
        except Exception as e:
            logging.error(f"扫描页面 {url} 时出错: {e}")
            break
            
    return list(links)

def scrape_and_save_article(url, collection):
    """爬取单篇文章并存入 MongoDB"""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            logging.error(f"请求失败 [{res.status_code}]: {url}")
            return False
            
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 提取标题
        title_tag = soup.find('h1')
        title = title_tag.text.strip() if title_tag else "Unknown Title"
        
        # 提取正文内容
        content_div = soup.find('div', class_='gh-content') or soup.find('article')
        if not content_div:
            logging.warning(f"未找到正文内容: {url}")
            return False
            
        # 🌟 核心修改：用 markdownify 替换掉原先的 find_all 手动提取
        # strip=['script', 'style'] 可以过滤掉无用的代码标签
        # heading_style="ATX" 保证标题是 # 格式，大模型最容易懂
        content_md = md(str(content_div), heading_style="ATX", strip=['script', 'style'])
        
        # 构建文档
        doc = {
            "source": "XLab",
            "url": url,
            "title": title,
            "content": content_md,  # 🌟 这里存入保留了表格、图片链接的 Markdown 文本
            "updated_at": datetime.now()
        }
        
        # 存入 MongoDB（根据 URL 去重更新）
        collection.update_one(
            {"url": url},
            {"$set": doc},
            upsert=True
        )
        logging.info(f"✅ 成功入库: {title}")
        return True
        
    except Exception as e:
        logging.error(f"解析文章 {url} 失败: {e}")
        return False

def main():
    logging.info("开始连接 MongoDB...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    col = db[COLLECTION_NAME]
    
    logging.info("第一步：获取所有文章 URL 列表...")
    article_links = get_article_links()
    logging.info(f"共发现 {len(article_links)} 篇文章链接。")
    
    logging.info("第二步：逐篇解析并存入数据库...")
    success_count = 0
    for idx, url in enumerate(article_links, 1):
        logging.info(f"进度 [{idx}/{len(article_links)}]...")
        if scrape_and_save_article(url, col):
            success_count += 1
        time.sleep(2) # 防止请求过快被封 IP
        
    logging.info(f"爬取任务完成！成功入库 {success_count}/{len(article_links)} 篇文章。")

if __name__ == "__main__":
    main()