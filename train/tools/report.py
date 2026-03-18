from langchain.tools import tool
import logging

@tool
def report_suspicious_traffic_tool(
    is_suspicious: bool, 
    suspicious_flows_start: int = -1, 
    suspicious_flows_duration: int = -1
) -> dict:
    """
    用于报告流量检测的最终结果。
    模型在分析完流量数据后，必须调用此工具来提交结论。
    
    Args:
        is_suspicious (bool): 本次分析是否发现了可疑攻击流量。
        suspicious_flows_start (int): 可疑流量片段的开始编号 (batch_index)。如果是正常流量或未找到，请填 -1。
        suspicious_flows_duration (int): 可疑流量片段的持续条数。如果是正常流量或未找到，请填 -1。
    
    Returns:
        dict: 包含检测结果的字典
    """
    logging.info("正在使用 report 工具")
    
    # 为了在控制台醒目地显示，我们加一些格式化
    print("\n" + "!" * 40)
    print(" [REPORT] 收到模型检测报告")
    print("!" * 40)
    
    print(f" > 是否存在可疑行为: {is_suspicious}")
    
    if is_suspicious:
        print(f" > 攻击流量范围: [ batch_index: {suspicious_flows_start} ] 到 [ batch_index: {suspicious_flows_start+suspicious_flows_duration-1} ]")
        # 这里你可以加额外的逻辑，比如触发报警邮件、记录数据库等
    else:
        print(" > 结论: 流量正常 (No Threat Detected)")
        
    print("!" * 40 + "\n")
    
    # 返回结构化的字典结果
    return {
        "is_suspicious": is_suspicious,
        "suspicious_flows_start": suspicious_flows_start,
        "suspicious_flows_end": suspicious_flows_start + suspicious_flows_duration - 1 if is_suspicious else -1,
        "suspicious_flows_duration": suspicious_flows_duration,
        "message": f"Report processed. Suspicious: {is_suspicious}, Range: {suspicious_flows_start}-{suspicious_flows_start+suspicious_flows_duration-1}."
    }