import re

def process_message_content(content: str, enable_think_output: bool = False) -> str:
    """处理消息内容，根据参数决定是否移除 <think> 标签及其内容
    
    Args:
        content: 原始消息内容
        enable_think_output: 是否保留 <think> 标签及其内容，默认为 False
    
    Returns:
        处理后的消息内容
    """
    if not enable_think_output:
        # 移除 <think> 标签及其内容
        # 匹配 <think> 标签及其内容
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        # 移除多余的空白字符
        content = ' '.join(content.split())
    return content
