"""
llm_wrapper.py — LLM 调用封装

提供两种模式：
    1. real: 用 llm.py 里的 ChatOpenAI 接 vLLM
    2. mock: 用规则模拟（用于无 LLM 环境测试工作流逻辑）
"""

import os
import json
import logging
import re
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

LLM_MODE = os.environ.get("FRP_TEST_LLM_MODE", "real")  # real | mock

# ───── LLM 调用追踪（调试用）─────
_llm_call_counter = 0

def _is_verbose() -> bool:
    """检查是否开启 LLM verbose 打印（每次都查环境变量，方便程序化切换）"""
    return os.environ.get("FRP_TEST_VERBOSE_LLM", "0").lower() in ("1", "true", "yes", "on")


def _try_real_llm(prompt: str, caller: str = "?") -> str:
    """
    调用真实 LLM。
    
    Args:
        prompt: 完整 prompt
        caller: 调用来源标识（如 "doc_gate(frp)" / "judge_step(frp_wss.step1)"），
                仅用于 verbose 打印时显示
    
    Returns:
        清理掉 <think>...</think> 后的纯净响应
    """
    global _llm_call_counter
    _llm_call_counter += 1
    call_id = _llm_call_counter
    verbose = _is_verbose()
    
    if verbose:
        bar = "═" * 78
        print(f"\n{bar}", flush=True)
        print(f"  LLM Call #{call_id}  —  {caller}", flush=True)
        print(f"{bar}", flush=True)
        print(f"【发送 Prompt】({len(prompt)} 字符)", flush=True)
        print(prompt, flush=True)
        print(f"\n【调用中...】", flush=True)
    
    t0 = time.time()
    from llm import llm
    response = llm.invoke(prompt)
    elapsed = time.time() - t0
    
    raw_text = response.content if hasattr(response, "content") else str(response)
    
    # 抽取 <think> 思考部分
    think_match = re.search(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    clean = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    
    if verbose:
        print(f"【LLM 完成】耗时 {elapsed:.2f}s", flush=True)
        if thinking:
            print(f"\n【模型思考过程】", flush=True)
            print(thinking, flush=True)
        else:
            print(f"\n【无 <think> 标签（模型未输出思考链）】", flush=True)
        print(f"\n【模型最终输出】", flush=True)
        print(clean, flush=True)
        print(f"{bar}\n", flush=True)
    else:
        # 非 verbose 模式仍记一条 DEBUG 日志，方便从日志里追
        logger.debug(f"LLM Call #{call_id} [{caller}] in {elapsed:.2f}s, "
                     f"prompt={len(prompt)} chars, response={len(clean)} chars")
    
    return clean


# ─────────────────────────────── 决策门：doc_gate ───────────────────────────────

DOC_GATE_PROMPT = """你正在做攻击审计的"是否进入下一阶段"决策。

【已收集的证据】
{evidence_summary}

【当前候选文档】
- ID: {doc_id}
- 描述: {doc_desc}
- 关键特征: {signatures}

【你的任务】
判断是否需要进入该文档的详细审计，输出之一：
- "ENTER" 表示需要进入该文档（前面证据不足以下结论）
- "DONE"  表示已有足够证据下最终结论，无需再审计

【判定原则】
1. 如果已经命中 hard fingerprint（ALPN=frp / URI=/~!frp 等），输出 DONE。
2. 否则输出 ENTER。

只输出 ENTER 或 DONE 一个词。
"""


def llm_doc_gate(evidence: list, doc: Dict[str, Any]) -> str:
    """返回 ENTER 或 DONE"""
    if LLM_MODE == "mock":
        # 规则版：看证据里是否有 hard fingerprint
        for e in evidence:
            if e.get("hard_fingerprint"):
                return "DONE"
        return "ENTER"
    
    evidence_str = _format_evidence(evidence)
    prompt = DOC_GATE_PROMPT.format(
        evidence_summary=evidence_str,
        doc_id=doc.get("doc_id", ""),
        doc_desc=doc.get("matching_features", {}).get("text_description", "")[:300],
        signatures=json.dumps(doc.get("common_signatures", {}), ensure_ascii=False)
    )
    try:
        resp = _try_real_llm(prompt, caller=f"doc_gate({doc.get('doc_id', '?')})")
        return "DONE" if "DONE" in resp.upper() else "ENTER"
    except Exception as e:
        logger.warning(f"doc_gate LLM 调用失败: {e}，退化为 ENTER")
        return "ENTER"


# ─────────────────────────────── 决策门：skill_gate ───────────────────────────────

SKILL_GATE_PROMPT = """你正在子 skill 级别的审计闸门做决策。

【当前文档】{doc_id}
【当前 skill】{skill_id} — {skill_title}

【已收集证据】
{evidence_summary}

【当前 skill 的特征】
{skill_signatures}

【你的任务】
判断是否需要执行该 skill 的工作流，输出之一：
- "EXECUTE" 需要执行
- "SKIP"    跳过（该模式已确定不匹配，或已被其他证据替代）
- "DONE"    整个文档可以收尾

只输出 EXECUTE / SKIP / DONE 一个词。
"""


def llm_skill_gate(evidence: list, doc: Dict[str, Any], skill: Dict[str, Any]) -> str:
    """返回 EXECUTE / SKIP / DONE"""
    if LLM_MODE == "mock":
        # 规则版：
        # - 如果之前已经命中该 skill 的 hard fingerprint，SKIP
        # - 如果其他 skill 已经命中 hard fingerprint，DONE
        sid = skill.get("skill_id", "")
        for e in evidence:
            if e.get("hard_fingerprint"):
                if e.get("skill_id") == sid:
                    return "SKIP"
                return "DONE"
        return "EXECUTE"
    
    evidence_str = _format_evidence(evidence)
    prompt = SKILL_GATE_PROMPT.format(
        doc_id=doc.get("doc_id", ""),
        skill_id=skill.get("skill_id", ""),
        skill_title=skill.get("title", ""),
        evidence_summary=evidence_str,
        skill_signatures=json.dumps(skill.get("specific_signatures", {}), ensure_ascii=False)
    )
    try:
        resp = _try_real_llm(prompt, caller=f"skill_gate({skill.get('skill_id', '?')})").upper()
        for kw in ("EXECUTE", "SKIP", "DONE"):
            if kw in resp:
                return kw
        return "EXECUTE"
    except Exception as e:
        logger.warning(f"skill_gate LLM 调用失败: {e}，退化为 EXECUTE")
        return "EXECUTE"


# ─────────────────────────────── 判断工具结果是否符合期望 ───────────────────────────────

JUDGE_PROMPT = """你是网络流量审计专家，判断单条工具的执行结果是否满足预期条件。

【当前 skill】{skill_id}
【步骤说明】{step_name}
【调用工具】{tool}
【预期条件】{expect}
【工具实际输出】
{result}

【判定规则 - 严格遵守】
1. 如果工具输出中 ok=false 或包含 error 字段，输出 FAIL
2. 仔细读取实际输出中的字段值，与预期条件比对
3. 数值条件（如 "duration > 60"）：在输出里找到对应字段（如 conversations[*].duration_sec），存在 1 条满足即 PASS
4. 布尔条件（如 "has_frp_alpn == true"）：直接读取对应布尔字段
5. 列表非空条件（如 "snis 不为空"）：检查列表长度
6. 「可疑域名」判定：SNI 域名以 .test/.local/.lab/ 结尾，或是动态 DNS（duckdns/no-ip/freedns），或是看起来明显是攻击者生造的（如 c2.xxx），即可视为可疑

【输出格式】
只输出一个词：PASS 或 FAIL，不要任何其他文字、解释、标点。
"""


def llm_judge_step(tool: str, expect: str, result: Dict[str, Any], skill_id: str = "", step_name: str = "") -> str:
    """单步判断：工具结果是否符合 expect。返回 PASS 或 FAIL。"""
    if LLM_MODE == "mock":
        # 工具调用本身失败 → FAIL
        if not result.get("ok", True):
            return "FAIL"
        
        if not expect:
            return "PASS"
        
        e_lower = expect.lower()
        
        # ===== hard fingerprint =====
        if "has_frp_alpn" in e_lower:
            return "PASS" if result.get("has_frp_alpn") else "FAIL"
        if "has_frp_websocket_path" in e_lower:
            return "PASS" if result.get("has_frp_websocket_path") else "FAIL"
        
        # ===== JA3 / TLS =====
        if "is_go_default_ja3" in e_lower:
            return "PASS" if result.get("is_go_default_ja3") else "FAIL"
        if "存在 client hello" in e_lower or "client_hellos" in e_lower:
            return "PASS" if result.get("client_hellos") else "FAIL"
        if "snis 包含可疑域名" in e_lower or ("snis" in e_lower and "可疑" in expect):
            snis = result.get("snis", [])
            if not snis:
                return "FAIL"
            # 公认大公司主流域名（白名单）
            KNOWN_PUBLIC = (
                "google.com", "googleapis.com", "gstatic.com", "youtube.com",
                "microsoft.com", "office.com", "azure.com", "live.com",
                "amazon.com", "amazonaws.com", "cloudfront.net",
                "apple.com", "icloud.com",
                "cloudflare.com", "cloudflarestream.com",
                "facebook.com", "fbcdn.net",
                "github.com", "githubusercontent.com",
            )
            for sni in snis:
                if not any(p in sni for p in KNOWN_PUBLIC):
                    return "PASS"  # 至少有一个 SNI 不在白名单
            return "FAIL"
        if "snis" in e_lower:
            return "PASS" if result.get("snis") else "FAIL"
        if "is_self_signed" in e_lower:
            certs = result.get("certificates", [])
            return "PASS" if any(c.get("is_self_signed") for c in certs) else "FAIL"
        
        # ===== 心跳 =====
        if "is_periodic" in e_lower:
            return "PASS" if result.get("is_periodic") else "FAIL"
        if "estimated_period_sec" in e_lower:
            period = result.get("estimated_period_sec", 0)
            # 支持 "in [25, 35]" 或 "≈ 10" 这种表达
            range_match = re.search(r"\[(\d+),\s*(\d+)\]", expect)
            if range_match:
                lo, hi = float(range_match.group(1)), float(range_match.group(2))
                return "PASS" if lo <= period <= hi else "FAIL"
            approx_match = re.search(r"[≈~]\s*(\d+)", expect)
            if approx_match:
                target = float(approx_match.group(1))
                return "PASS" if abs(period - target) <= 3 else "FAIL"
            return "PASS" if result.get("is_periodic") else "FAIL"
        
        # ===== 流量方向 =====
        if "ratio" in e_lower or "internal_penetration" in e_lower:
            return "PASS" if result.get("is_internal_penetration_suspect") else "FAIL"
        
        # ===== 包大小 =====
        if "mtu_pct" in e_lower:
            m = re.search(r"(\d+)", expect)
            threshold = float(m.group(1)) if m else 60
            return "PASS" if result.get("mtu_pct", 0) > threshold else "FAIL"
        
        # ===== 字节模式 =====
        if "match_count" in e_lower:
            return "PASS" if result.get("match_count", 0) > 0 else "FAIL"
        
        # ===== 主动探查 nginx =====
        if "is_nginx" in e_lower:
            return "PASS" if result.get("is_nginx") else "FAIL"
        
        # ===== UDP 会话 =====
        if "duration > 60" in e_lower or "udp 会话" in e_lower:
            convs = result.get("conversations", [])
            return "PASS" if any(c.get("duration_sec", 0) > 60 for c in convs) else "FAIL"
        
        # ===== QUIC 排除 =====
        if "initial_packets_count == 0" in e_lower:
            return "PASS" if result.get("initial_packets_count", 0) == 0 else "FAIL"
        
        # 兜底：未识别的 expect 默认 FAIL（更保守，不让弱证据冒充强证据）
        return "FAIL"
    
    prompt = JUDGE_PROMPT.format(
        skill_id=skill_id or "(unknown)",
        step_name=step_name or "(no name)",
        tool=tool,
        expect=expect,
        result=json.dumps(result, ensure_ascii=False, default=str)[:1500]
    )
    try:
        # caller 里带 skill+step 名，方便定位是哪个步骤
        _step_label = (step_name or "?")[:30].replace(" ", "_")
        resp = _try_real_llm(prompt, caller=f"judge_step({skill_id or '?'}.{_step_label})").upper()
        return "PASS" if "PASS" in resp else "FAIL"
    except Exception as e:
        logger.warning(f"judge_step LLM 调用失败: {e}，退化为 FAIL")
        return "FAIL"


# ─────────────────────────────── 最终判定 ───────────────────────────────

# ─────────────────────────────── 子 skill 排序（Feature 2） ───────────────────────────────

SKILL_RANK_PROMPT = """你正在对 FRP 内网穿透的子模式检测顺序做决策。

【当前流量特征摘要】
{flow_summary}

【候选 sub_skills】
{skill_list}

【你的任务】
根据流量特征，把这些 sub_skills 按"最有可能命中"的概率从高到低排序，输出每行一个 skill_id。
只输出 skill_id 列表，不要任何其他文字。

例如：
frp_quic
frp_kcp
frp_ws
...
"""


def llm_rank_skills(flow_summary: dict, sub_skills: list) -> list:
    """
    让 LLM 根据流量特征对 sub_skills 排序，返回 skill_id 列表（按优先级降序）。
    
    Args:
        flow_summary: 流量特征摘要（来自 _build_flow_summary）
        sub_skills: doc 的 sub_skills 列表 [{skill_id, priority, ...}, ...]
    
    Returns:
        ordered_skill_ids: 排好序的 skill_id 列表
    """
    candidate_ids = [s["skill_id"] for s in sub_skills]
    
    if LLM_MODE == "mock":
        return _heuristic_rank_skills(flow_summary, candidate_ids)
    
    # 用 sub_skills 自身的元信息构造 prompt
    skill_descs = []
    for s in sub_skills:
        sid = s["skill_id"]
        desc = s.get("title") or s.get("description", "")
        skill_descs.append(f"  - {sid}: {desc[:120]}")
    
    prompt = SKILL_RANK_PROMPT.format(
        flow_summary=json.dumps(flow_summary, ensure_ascii=False, indent=2),
        skill_list="\n".join(skill_descs)
    )
    
    try:
        resp = _try_real_llm(prompt, caller=f"rank_skills({len(candidate_ids)} candidates)")
        # 抽出 skill_id 行
        valid = set(candidate_ids)
        ordered = []
        for line in resp.splitlines():
            line = line.strip().strip("-*•").strip()
            for cid in valid:
                if cid in line and cid not in ordered:
                    ordered.append(cid)
                    break
        # 补齐遗漏的
        for cid in candidate_ids:
            if cid not in ordered:
                ordered.append(cid)
        logger.info(f"LLM 排序: {ordered}")
        return ordered
    except Exception as e:
        logger.warning(f"llm_rank_skills 失败 ({e})，回退到启发式")
        return _heuristic_rank_skills(flow_summary, candidate_ids)


def _heuristic_rank_skills(flow_summary: dict, candidate_ids: list) -> list:
    """
    启发式排序（mock 模式 / LLM 失败时使用）。
    
    思路：基于流量层最显著的特征做粗排：
        - 主要协议是 UDP → quic/kcp 优先
        - TCP 上 443 + TLS → wss 优先
        - TCP 80/8080 无 TLS → ws 优先
        - TCP + TLS + 非 443 → tcp_tls 优先
        - 其余 → tcp 兜底
    """
    proto = flow_summary.get("proto_dist", {})
    udp_cnt = proto.get("udp", 0)
    tcp_cnt = proto.get("tcp", 0)
    dst_ports = set(flow_summary.get("dst_ports", []))
    has_tls = flow_summary.get("has_tls", False)
    has_443 = 443 in dst_ports
    has_quic_port = bool(dst_ports & {443, 7003, 8443, 4433})  # QUIC 常用端口
    has_web_port = bool(dst_ports & {80, 8080, 7004})
    
    # 每个 skill 算分
    scores = {}
    for sid in candidate_ids:
        if sid == "frp_quic":
            # UDP + 典型 QUIC 端口
            scores[sid] = 100 if (udp_cnt > 0 and has_quic_port) else (50 if udp_cnt > 0 else 5)
        elif sid == "frp_kcp":
            # UDP 但不是 QUIC 端口（KCP 没固定端口）
            scores[sid] = 90 if (udp_cnt > 0 and not has_quic_port) else (40 if udp_cnt > 0 else 10)
        elif sid == "frp_wss":
            # TCP + 443 + TLS
            scores[sid] = 95 if (tcp_cnt > 0 and has_443 and has_tls) else 25
        elif sid == "frp_ws":
            # TCP + 80/8080 明文
            scores[sid] = 85 if (tcp_cnt > 0 and has_web_port and not has_tls) else 20
        elif sid == "frp_tcp_tls":
            # TCP + TLS + 非 443 端口
            scores[sid] = 75 if (tcp_cnt > 0 and has_tls and not has_443) else 30
        elif sid == "frp_tcp":
            # 通用 fallback
            scores[sid] = 60 if tcp_cnt > 0 else 5
        else:
            scores[sid] = 1
    
    ordered = sorted(candidate_ids, key=lambda x: scores.get(x, 0), reverse=True)
    logger.debug(f"启发式排序: {[(s, scores[s]) for s in ordered]}")
    return ordered


# ─────────────────────────────── 最终判定 ───────────────────────────────

FINAL_VERDICT_PROMPT = """你是网络流量攻击审计专家，专门识别 FRP 内网穿透工具的 6 种传输模式。基于下面的证据给出最终判定。

【流量上下文摘要】
{flow_context}

【FRP 子模式知识库（每种模式的核心识别特征）】
{skill_catalog}

【已审计 skill 列表】
{audited_docs}

【详细证据（按 skill 分组，★★★ 标记为决定性 hard fingerprint）】
{all_evidence}

【判定规则 - 严格遵守】
1. 如果任何证据被标注为 ★★★ HARD 且判定为 PASS，则该 skill 必为最终答案，confidence = 1.0
2. 没有 hard fingerprint 命中时，综合 PASS 数量 + 每个 skill 的"特异性证据"（独有的、其他模式不会 PASS 的步骤）做选择
3. ★ 资产交叉印证：流量上下文里的「资产记录」是独立于 skill 检测的第三方证据。当 skill 命中的特征与资产标签一致时（如 frp_wss 命中 SNI 可疑 + 资产显示目标 IP:443 是 nginx + 自签证书），confidence 应提升至 ≥ 0.85
4. frp_tcp 和 frp_tcp_tls 在网络层无法区分（v0.50+ 默认 TLS）；只看见 TCP/TLS 但无 SNI 异常时优先 frp_tcp
5. frp_wss 必须看到「SNI 可疑域名 + Go JA3 + 443 端口」三者组合；如果资产显示目标 443 是 nginx，则更应判 frp_wss 而非 frp_tcp
6. frp_ws 必须看到 HTTP URI = /~!frp 才能确认；只有心跳 PASS 不够

【输出格式 - 只输出 JSON，不要任何其他文字】
{{
  "is_attack": true/false,
  "attack_type": "intranet_penetration / unknown",
  "tool": "frp / nps / chisel / unknown",
  "matched_sub_skill": "frp_xxx (必须是已审计的 skill_id 之一)",
  "confidence": 0.0-1.0,
  "reasoning": "为什么选这个 skill（一句话）",
  "key_evidence": ["证据1", "证据2", "证据3"],
  "recommended_actions": ["..."]
}}
"""


def _build_flow_context(state_or_recalled: Dict[str, Any]) -> str:
    """格式化流量上下文给 LLM 看"""
    recalled = state_or_recalled.get("recalled_flows", state_or_recalled) or {}
    anchor = recalled.get("anchor_flow") or {}
    tls = recalled.get("tls_handshake")
    hb = recalled.get("heartbeat")
    
    lines = []
    if anchor:
        lines.append(f"- anchor 流: {anchor.get('src')} -> {anchor.get('dst')} ({anchor.get('proto')}), "
                     f"duration={anchor.get('duration')}s, orig_bytes={anchor.get('orig_bytes')}, "
                     f"resp_bytes={anchor.get('resp_bytes')}")
    if tls:
        snis = tls.get("snis", [])
        ja3s = tls.get("ja3_hashes", [])
        is_go_ja3 = tls.get("is_go_default_ja3", False)
        self_signed = tls.get("any_self_signed", False)
        cert_count = tls.get("certificate_count", 0)
        lines.append(f"- TLS Client Hello: {tls.get('count')} 次，SNI={snis[:3]}，"
                     f"JA3={ja3s[:2]}, Go默认JA3={is_go_ja3}, "
                     f"证书={cert_count} (自签={self_signed})")
    else:
        lines.append("- TLS Client Hello: 无（明文流量或 QUIC）")
    if hb:
        lines.append(f"- 周期性心跳: 周期={hb.get('period_sec')}s, "
                     f"bursts={hb.get('burst_count')}, packets_per_burst={hb.get('packets_per_burst')}")
    else:
        lines.append("- 周期性心跳: 未检出")
    data = recalled.get("data_streams", [])
    if data:
        lines.append(f"- 大字节数据流: {len(data)} 个候选")
    
    # ★ 资产信息（来自知识图谱 / SIM）
    asset = recalled.get("asset_info")
    if asset:
        try:
            from core.asset_lookup import summarize_asset_for_llm
            # 取目标端口（anchor 流的目的端口）
            anchor_dst = (anchor.get("dst") or "")
            port = None
            if ":" in anchor_dst:
                try:
                    port = int(anchor_dst.split(":")[-1])
                except ValueError:
                    pass
            asset_text = summarize_asset_for_llm(asset, port=port)
            lines.append(asset_text)
        except Exception:
            lines.append(f"- 资产信息: {asset.get('source', '?')}")
    
    return "\n".join(lines) if lines else "  (无上下文)"


# FRP 子模式的简要识别特征 catalog（写死，不动态读，避免 prompt 太长）
FRP_SKILL_CATALOG = {
    "frp_tcp":     "TCP 默认模式（v0.50+ 默认启用 TLS）。识别：Go 默认 JA3 + 30s 精确心跳 + 流量方向反转（resp/orig > 5）。无 hard fingerprint。资产线索：目标端口非标准（非 80/443/22 等常见），server_header 为 unknown，但有自签证书。",
    "frp_tcp_tls": "TCP + 强制 TLS 模式（transport.tls.force=true）。网络层与 frp_tcp 完全不可区分，识别特征同 frp_tcp。区别仅在服务端策略。资产线索同 frp_tcp，但端口标签可能含 forced_tls。",
    "frp_kcp":     "KCP over UDP 模式。识别：UDP 长会话 + 包大小分布中 MTU 饱和段（1280-2559 字节占比 > 60%）+ 30s 心跳。无 QUIC 握手（区别于 frp_quic）。资产线索：目标 UDP 端口标签 long_udp_session 或 non_dns_non_quic_udp。",
    "frp_quic":    "QUIC 模式。★★★ 决定性指纹：QUIC ALPN = 'frp'（合法 QUIC 是 h3/hq-XX）。辅助：10s 短心跳。资产线索：目标 UDP 端口 alpn 字段包含 'frp' 是决定性。",
    "frp_ws":      "WebSocket 明文模式。★★★ 决定性指纹：HTTP URI = /~!frp（硬编码在 frp 源码 issue #5278）。辅助：30s 心跳，明文 HTTP 1.1 Upgrade。资产线索：目标 TCP 端口有 HTTP 但 server_header 缺失或非主流，端口标签 frp_websocket_suspect。",
    "frp_wss":     "WebSocket over TLS 模式，必须前置 nginx/caddy 做 TLS 终结。识别：Go 默认 JA3 + 可疑 SNI（非主流域名/.test/动态 DNS）+ 自签证书 + 主动探查 nginx + 30s 心跳。无单一 hard fingerprint，需组合判定。★★★ 资产线索（决定性辅助）：目标 IP:443 的 server_header 含 nginx 或 caddy + 自签证书 + 端口标签 reverse_proxy_candidate 或 wss_terminator。",
}


def _build_skill_catalog(audited_skill_ids: list) -> str:
    """只列出审计过的 skill 的特征说明"""
    lines = []
    for sid in audited_skill_ids:
        desc = FRP_SKILL_CATALOG.get(sid, "(未知 skill)")
        lines.append(f"- {sid}: {desc}")
    return "\n".join(lines) if lines else "  (无)"


def _format_evidence_detailed(evidence: list) -> str:
    """详细版证据格式：按 skill 分组，包含真实工具输出片段 + hard fingerprint 标注"""
    if not evidence:
        return "  (暂无证据)"
    
    # 按 skill_id 分组
    from collections import defaultdict
    groups = defaultdict(list)
    for e in evidence:
        groups[e.get("skill_id", "?")].append(e)
    
    lines = []
    for sid, items in groups.items():
        lines.append(f"\n[{sid}]")
        for e in items:
            mark = "★★★ HARD" if e.get("hard_fingerprint") else "        "
            judgment = e.get("judgment", "?")
            j_mark = "✓ PASS" if judgment == "PASS" else "✗ FAIL"
            step = e.get("step_name", "?")[:40]
            tool = e.get("tool", "?")
            expect = (e.get("expected") or "")[:80]
            
            # 提取工具结果的关键字段（避免 prompt 太长）
            result = e.get("tool_result", {}) or {}
            key_fields = {}
            for k in ("has_frp_alpn", "has_frp_websocket_path", "is_go_default_ja3",
                      "is_periodic", "estimated_period_sec", "snis",
                      "is_internal_penetration_suspect", "mtu_pct", "match_count",
                      "is_nginx", "client_hellos", "ratio_b_to_a"):
                if k in result:
                    v = result[k]
                    if isinstance(v, list):
                        v = v[:3]
                    elif isinstance(v, (int, float)) and not isinstance(v, bool):
                        v = round(v, 3) if isinstance(v, float) else v
                    key_fields[k] = v
            
            lines.append(f"  {mark} {j_mark}  step{e.get('step_id')} {step}")
            lines.append(f"           tool={tool}, expect={expect}")
            if key_fields:
                lines.append(f"           工具关键输出: {json.dumps(key_fields, ensure_ascii=False, default=str)[:200]}")
    
    return "\n".join(lines)


def llm_final_verdict(evidence: list, audited_docs: list, state: Dict[str, Any] = None) -> Dict[str, Any]:
    """生成最终判定 JSON
    
    Args:
        evidence: 累积证据列表
        audited_docs: 已审计的 skill_id 列表
        state: 完整 state（可选）用于提取 flow_context
    """
    if LLM_MODE == "mock":
        return _mock_final_verdict(evidence, audited_docs)
    
    flow_ctx = _build_flow_context(state or {})
    skill_cat = _build_skill_catalog(audited_docs)
    ev_detail = _format_evidence_detailed(evidence)
    
    prompt = FINAL_VERDICT_PROMPT.format(
        flow_context=flow_ctx,
        skill_catalog=skill_cat,
        audited_docs=", ".join(audited_docs),
        all_evidence=ev_detail
    )
    
    logger.info(f"final_verdict prompt 长度: {len(prompt)} 字符")
    
    try:
        resp = _try_real_llm(prompt, caller=f"final_verdict(audited={','.join(audited_docs[:3])}{'...' if len(audited_docs)>3 else ''})")
        # 提取 JSON
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        if m:
            try:
                verdict = json.loads(m.group(0))
                # 校验 matched_sub_skill 必须在已审计列表中
                if verdict.get("matched_sub_skill") not in audited_docs:
                    logger.warning(f"LLM 选择了未审计的 skill: {verdict.get('matched_sub_skill')}, 回退到 mock")
                    return _mock_final_verdict(evidence, audited_docs)
                return verdict
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败: {e}, 原始响应: {resp[:200]}")
        return _mock_final_verdict(evidence, audited_docs)
    except Exception as e:
        logger.warning(f"final_verdict LLM 调用失败: {e}，退化为规则版")
        return _mock_final_verdict(evidence, audited_docs)


def _mock_final_verdict(evidence: list, audited_docs: list) -> Dict[str, Any]:
    """规则版判定：用证据里的 hard_fingerprint 推断"""
    # 1. 如果有 hard_fingerprint 命中，直接用命中的 skill（最强信号）
    hard_hits = {}
    for e in evidence:
        if e.get("hard_fingerprint") and e.get("judgment") == "PASS":
            sid = e.get("skill_id", "")
            hard_hits[sid] = hard_hits.get(sid, 0) + 1
    
    if hard_hits:
        best_skill = max(hard_hits.items(), key=lambda x: x[1])[0]
        tool_name = best_skill.split("_")[0] if "_" in best_skill else "unknown"
        # hard fingerprint 命中 → 高置信度
        confidence = 1.0
        return {
            "is_attack": True,
            "attack_type": "intranet_penetration",
            "tool": tool_name,
            "matched_sub_skill": best_skill,
            "confidence": confidence,
            "key_evidence": [e.get("summary") for e in evidence if e.get("hard_fingerprint")][:5],
            "recommended_actions": [
                "阻断目标 IP",
                "审查内网主机异常进程",
                f"用 {best_skill} 模式的 Suricata 规则部署到 IDS"
            ]
        }
    
    # 2. 没有 hard fingerprint，按 PASS 数+通过率+特异性综合打分
    # 关键改进：考虑 skill 的"独有强证据"——比如 frp_wss 命中 SNI 检测是它独有的
    skill_pass = {}
    skill_total = {}
    skill_unique_signals = {}  # 该 skill 命中的"独有/特异"证据数
    
    # 标记哪些 (skill_id, tool) 组合是该 skill 的"特异性证据"
    # 关键：这些证据 PASS 时必须确实指向该 skill，而非任何 pcap 都会 PASS
    # 比如 frp_kcp 的 pcap_quic_handshake 是"排除 QUIC"的步骤，不算特异（任何非 QUIC pcap 都 PASS）
    SPECIFIC_SIGNALS = {
        "frp_wss":     {"pcap_tls_sni_cert", "active_probe_http_server"},  # SNI 可疑 + nginx 反代
        "frp_tcp_tls": {"pcap_tls_sni_cert"},                              # SNI 命中
        "frp_kcp":     {"pcap_packet_size_distribution"},                  # MTU 饱和（KCP 独有）
        "frp_quic":    set(),  # 已被 hard fingerprint (ALPN=frp) 覆盖
        "frp_ws":      set(),  # 已被 hard fingerprint (/~!frp URI) 覆盖
        "frp_tcp":     set(),  # fallback，无特异性
    }
    
    for e in evidence:
        sid = e.get("skill_id", "")
        if not sid:
            continue
        skill_total[sid] = skill_total.get(sid, 0) + 1
        if e.get("judgment") == "PASS":
            skill_pass[sid] = skill_pass.get(sid, 0) + 1
            # 检查是否是该 skill 的特异性证据
            tool = e.get("tool", "")
            if tool in SPECIFIC_SIGNALS.get(sid, set()):
                skill_unique_signals[sid] = skill_unique_signals.get(sid, 0) + 1
    
    if not skill_total:
        return {
            "is_attack": False,
            "attack_type": "unknown",
            "tool": "unknown",
            "matched_sub_skill": "",
            "confidence": 0.0,
            "key_evidence": [],
            "recommended_actions": []
        }
    
    # 评分：基础 PASS 数 + 特异性证据 ×3（权重高，因为它们指向特定 skill）
    def _score(sid):
        base = skill_pass.get(sid, 0)
        unique = skill_unique_signals.get(sid, 0)
        rate = base / max(skill_total.get(sid, 1), 1)
        return base + unique * 3 + rate * 0.5
    
    best_skill = max(skill_total.keys(), key=_score)
    tool_name = best_skill.split("_")[0] if "_" in best_skill else "unknown"
    
    pass_rate = skill_pass.get(best_skill, 0) / max(skill_total.get(best_skill, 1), 1)
    confidence = pass_rate * 0.8  # 没 hard fingerprint 时上限 0.8
    
    return {
        "is_attack": confidence > 0.5,
        "attack_type": "intranet_penetration" if confidence > 0.5 else "unknown",
        "tool": tool_name,
        "matched_sub_skill": best_skill,
        "confidence": confidence,
        "key_evidence": [e.get("summary") for e in evidence if e.get("judgment") == "PASS" and e.get("skill_id") == best_skill][:5],
        "recommended_actions": [
            "阻断目标 IP",
            "审查内网主机异常进程",
            f"用 {best_skill} 模式的检测规则部署到 IDS"
        ]
    }


def _format_evidence(evidence: list) -> str:
    if not evidence:
        return "  (暂无证据)"
    lines = []
    for e in evidence:
        sid = e.get("skill_id", "?")
        step = e.get("step_name", "?")
        judgment = e.get("judgment", "?")
        lines.append(f"  - [{sid}] {step}: {judgment}")
    return "\n".join(lines)
