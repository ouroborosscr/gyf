from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage

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

# 3. 补充用于节点调用的快捷函数
def chat_once(prompt: str, response_format: str = None) -> str:
    """
    单轮对话封装，自动处理 langchain 的 invoke 和消息格式
    """
    messages = [HumanMessage(content=prompt)]
    
    kwargs = {}
    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}
        
    try:
        response = llm.invoke(messages, **kwargs)
        return response.content
    except Exception:
        # 如果 vLLM 服务端不支持强制 JSON 约束，则退级到普通调用
        response = llm.invoke(messages)
        return response.content