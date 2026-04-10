import requests
import json
import re

# 你的 vLLM 部署的 RRM 端口和模型路径
RRM_API_URL = "http://localhost:8123/v1/completions"
AGENT_RRM_PATH = "/date/sunchengrui/models/Agent-RRM"

# ========================================================
# 伪造一个完美的模型生成结果 (包含思考过程 + XML 工具调用)
# ========================================================
mock_completion = """
通过分析这批流量数据，我观察到了以下特征：
1. 在 batch_index 4 开始，存在连续多条目标端口为 3389 (RDP) 的 TCP 连接。
2. 这些连接的 stream_payload_decoded 中包含了典型的 'mstshash=hello' 字符串。
3. 这种行为高度符合“内网穿透”攻击的特征。
4. 该攻击行为从 batch_index 4 开始，一直持续到 batch_index 21，共计 18 条流量。

因此，我判定该段流量存在攻击行为，逻辑自洽。我将调用 report_suspicious_traffic_tool 工具进行上报。

<tool_call>
<function=report_suspicious_traffic_tool>
<parameter=is_suspicious>
True
</parameter>
<parameter=suspicious_flows_start>
4
</parameter>
<parameter=suspicious_flows_duration>
18
</parameter>
</function>
</tool_call>
"""

# ========================================================
# 构造发给 RRM 裁判的 Prompt (要求它只评判逻辑一致性)
# ========================================================
rm_prompt = (
    "Please evaluate the logic and internal consistency of the following agent reasoning and tool call:\n\n"
    f"{mock_completion}\n\n"
    "Focus only on whether the reasoning logically leads to the tool call. "
    "Provide a score between 0.0 and 1.0. You must output the score in the format: <score>0.x</score>"
)

payload = {
    "model": AGENT_RRM_PATH,
    "prompt": rm_prompt,
    "max_tokens": 150,
    "temperature": 0.0,  # 裁判不需要发散思维
}

print("🚀 正在向 RRM 裁判模型发送测试请求...")
try:
    response = requests.post(RRM_API_URL, json=payload, timeout=30)
    response.raise_for_status()
    
    result = response.json()
    rm_output = result["choices"][0]["text"]
    
    print("\n" + "="*50)
    print("🤖 [RRM 裁判原始返回内容]:")
    print(rm_output)
    print("="*50 + "\n")
    
    # 测试正则提取
    score_match = re.search(r'<score>\s*([0-9.]+)\s*</score>', rm_output)
    if score_match:
        print(f"✅ 成功提取到分数: {score_match.group(1)}")
    else:
        print("❌ 正则提取失败！RRM 输出了上面的内容，但没有严格包含 <score>x.x</score> 标签。")
        
except Exception as e:
    print(f"❌ 请求失败，请检查 8123 端口是否存活: {e}")