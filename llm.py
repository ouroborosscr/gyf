from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="Qwen3/Qwen3-8B",
)