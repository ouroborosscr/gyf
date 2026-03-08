import os
import logging
from langchain_tavily import TavilySearch

# 设置 TAVILY_API_KEY 环境变量
os.environ["TAVILY_API_KEY"] = "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"
# 查询工具
logging.info("正在加载 search 工具")
search_tool = TavilySearch(max_results=2)