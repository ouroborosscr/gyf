import logging
from tools.search_tool import search_tool
from tools.report import report_suspicious_traffic_tool

logging.info("正在加载工具集合")
# tools = [search_tool, report_suspicious_traffic_tool] # 暂时先不联网
tools = [report_suspicious_traffic_tool]
