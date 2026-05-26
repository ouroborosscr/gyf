import json
from llm import chat_once   # 使用我们刚刚封装的便捷函数
from layer3.tls13.prompts.verify import VERIFY_PROMPT

def verify_extracted(state: dict) -> dict:
    ch_hex = state.get("client_hello_hex") or ""
    sh_hex = state.get("server_hello_hex") or ""
    meta   = state.get("extracted_meta", {}) or {}

    # ---- 测试 mock：不调用 LLM，用启发式判断 ----
    if state.get("_mock_llm"):
        is_tls13 = bool(meta.get("is_tls13")) and bool(meta.get("sh_is_tls13", True))
        ok = bool(ch_hex and sh_hex
                  and len(ch_hex) >= 40 and len(sh_hex) >= 40
                  and is_tls13)
        return {
            **state,
            "extract_ok":           ok,
            "extract_confidence":   0.95 if ok else 0.1,
            "skip_remaining_cases": ok,   # mock 模式拿到就停
        }

    # ---- 正常路径：走 LLM ----
    tried = state.get("case_tried", [])
    user = VERIFY_PROMPT.format(
        case=tried[-1] if tried else "none",
        ch_hex_short=ch_hex[:200], ch_len=len(ch_hex)//2,
        sh_hex_short=sh_hex[:200], sh_len=len(sh_hex)//2,
        meta_json=json.dumps(meta, ensure_ascii=False, default=str),
    )
    
    # 改回使用 chat_once 进行字符串交互
    resp = chat_once(user, response_format="json")
    
    try:
        parsed = json.loads(resp)
    except Exception:
        parsed = {"extract_ok": bool(ch_hex and sh_hex),
                  "confidence": 0.5, "skip_remaining": False}
                  
    return {
        **state,
        "extract_ok":           bool(parsed.get("extract_ok")),
        "extract_confidence":   float(parsed.get("confidence", 0.0)),
        "skip_remaining_cases": bool(parsed.get("skip_remaining")),
    }


def route_after_verify(state: dict) -> str:
    if state.get("extract_ok") and state.get("skip_remaining_cases"):
        return "threat_intel"
    if state.get("extract_ok") and len(state.get("case_tried", [])) == 3:
        return "threat_intel"
    if not state.get("extract_ok") and len(state.get("case_tried", [])) >= 3:
        return "mark_incomplete"
    return "dispatch"   # 回到调度器，继续下一个 case