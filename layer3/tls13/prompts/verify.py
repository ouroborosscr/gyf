VERIFY_PROMPT = """你是 TLS1.3 协议审计助手。下面是某次提取尝试的结果，请判断：
1) 是否成功拿到完整的 ClientHello 和 ServerHello（二者都需要 TLS1.3，即 supported_versions 含 0x0304）；
2) 如果已经足够好，是否可以跳过剩余的提取尝试。

当前尝试的 case：{case}
ClientHello hex（前 200 字符）：{ch_hex_short}
ClientHello 长度（字节）：{ch_len}
ServerHello hex（前 200 字符）：{sh_hex_short}
ServerHello 长度（字节）：{sh_len}
提取附带元信息：{meta_json}

只返回严格 JSON：
{{"extract_ok": true|false, "confidence": 0.0-1.0, "skip_remaining": true|false, "reason": "..."}}
"""