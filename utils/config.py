import os

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
ZEEK_ROOT = os.path.join(PROJECT_ROOT, "zeek-mongo-lab")

# 目录信息
DIRECTORIES = {
    "zeek_logs": os.path.join(ZEEK_ROOT, "zeek-logs"),
    "tools": os.path.join(PROJECT_ROOT, "tools"),
    "data": os.path.join(PROJECT_ROOT, "data")
}

# 数据库登录信息
DATABASE = {
    "mongo": {
        "uri": "mongodb://admin:gyf424201@localhost:62015/",
        "db_name": "zeek_analysis"
    }
}

# Docker 容器信息
DOCKER = {
    "zeek_container": "my-zeek",
    "mongodb_container": "my-mongodb"
}

# Zeek 配置
ZEEK = {
    "default_pcap": "clean.pcap",
    "default_script": "save-payload.zeek"
}

# 分析配置
GYF_SETTINGS = {
    "hop_length": 30,
    "back": 5,
    "history_db_name": "gyf_history",
    "status_collection": "global_status"
}

# 确保必要的目录存在
for dir_path in DIRECTORIES.values():
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

ENABLE_THINK_OUTPUT = True

ENABLE_PRINT = False