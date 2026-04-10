from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# 1. 核心大语言模型 (Qwen3.5-9B)
# 运行在 8223 端口
llm = ChatOpenAI(
    base_url="http://localhost:8223/v1",
    api_key="EMPTY",
    model="Qwen3.5/Qwen3.5-9B",
)

# 2. 向量特征提取模型 (Qwen3-Embedding-4B)
# 运行在 8002 端口
embeddings = OpenAIEmbeddings(
    base_url="http://localhost:8002/v1",
    api_key="EMPTY",
    model="qwen3-embed-4b",      # 对应你 vLLM 启动时的 --served-model-name
    check_embedding_ctx_length=False # 禁用本地长度校验，交由服务端处理
)