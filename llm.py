from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="Qwen3.5/Qwen3.5-9B",
)