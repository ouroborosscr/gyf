#!/usr/bin/env python3
"""
evaluate.py — FRP 测试工作流批量评估

仿照 evaluate_recall_ip_port.py 的风格：
    1. 给定一组带 label 的 pcap
    2. 每个 pcap 跑完整 FRP 工作流
    3. 统计判定准确率，输出对比表

用法:
    # 在 gyf 项目根目录执行
    python -m frp_test.evaluate --pcap-dir /path/to/pcaps/
    
    # 用 mock LLM 模式（不需要 vLLM）
    FRP_TEST_LLM_MODE=mock python -m frp_test.evaluate --pcap-dir /path/to/pcaps/
"""

import os
import sys
import json
import argparse
import logging
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))


# ────── pcap 文件名 → 期望 skill_id 的映射规则 ──────

LABEL_RULES = [
    (r"frp_01_tcp[_.]",          "frp_tcp"),
    (r"frp_02_tcp_tls",          "frp_tcp_tls"),
    (r"frp_03_kcp",              "frp_kcp"),
    (r"frp_04_quic",             "frp_quic"),
    (r"frp_05_ws[_.]",           "frp_ws"),
    (r"frp_06_wss",              "frp_wss"),
]


def infer_expected_label(pcap_filename: str) -> str:
    """从 pcap 文件名推断期望的 skill_id"""
    for pattern, label in LABEL_RULES:
        if re.search(pattern, pcap_filename):
            return label
    return "unknown"


def evaluate_one_pcap(pcap_path: str, expected: str) -> dict:
    """跑一个 pcap，返回评估结果"""
    from core.frp_graph import get_frp_test_graph
    graph = get_frp_test_graph()
    
    init_state = {
        "pcap_path": str(pcap_path),
        "expected_label": expected,
        "evidence": [],
        "stage_summaries": [],
        "messages": []
    }
    
    final_state = None
    for event in graph.stream(init_state):
        for _, output in event.items():
            if output:
                final_state = output
    
    verdict = final_state.get("final_verdict", {}) if final_state else {}
    
    actual_skill = verdict.get("matched_sub_skill", "")
    actual_tool = verdict.get("tool", "")
    
    # 等价 skill 集合：网络层无法区分的模式互认
    # frp_tcp ≡ frp_tcp_tls（v0.50+ 默认 TLS，仅服务端 force_tls 策略差异，需主动 probe 验证）
    EQUIVALENT_SKILLS = [
        {"frp_tcp", "frp_tcp_tls"}
    ]
    
    def _is_equivalent(a: str, b: str) -> bool:
        if a == b:
            return True
        for eq_set in EQUIVALENT_SKILLS:
            if a in eq_set and b in eq_set:
                return True
        return False
    
    is_correct = _is_equivalent(actual_skill, expected)
    
    return {
        "pcap": Path(pcap_path).name,
        "expected_skill": expected,
        "actual_skill": actual_skill,
        "actual_tool": actual_tool,
        "is_attack": verdict.get("is_attack", False),
        "confidence": verdict.get("confidence", 0.0),
        "correct": is_correct,
        "key_evidence_count": len(verdict.get("key_evidence", [])),
        "verdict": verdict
    }


def main():
    parser = argparse.ArgumentParser(description="FRP 工作流批量评估")
    parser.add_argument("--pcap-dir", required=True, help="包含多个 pcap 的目录")
    parser.add_argument("--pattern", default="frp_*.pcap", help="pcap 文件匹配模式")
    parser.add_argument("--output", default="frp_evaluation_results.json", help="结果 JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - [%(levelname)s] - %(message)s"
    )
    
    pcap_dir = Path(args.pcap_dir).resolve()
    pcaps = sorted(pcap_dir.glob(args.pattern))
    
    if not pcaps:
        print(f"❌ 在 {pcap_dir} 下未找到匹配 {args.pattern} 的 pcap")
        sys.exit(1)
    
    print("=" * 80)
    print(f"FRP 工作流批量评估")
    print(f"  pcap 目录: {pcap_dir}")
    print(f"  匹配模式: {args.pattern}")
    print(f"  待评估 pcap 数: {len(pcaps)}")
    print(f"  LLM 模式: {os.environ.get('FRP_TEST_LLM_MODE', 'real')}")
    print("=" * 80)
    
    results = []
    for idx, pcap in enumerate(pcaps, 1):
        expected = infer_expected_label(pcap.name)
        print(f"\n[{idx}/{len(pcaps)}] {pcap.name} (期望: {expected})")
        try:
            r = evaluate_one_pcap(pcap, expected)
            results.append(r)
            
            mark = "✅" if r["correct"] else "❌"
            print(f"  {mark} 判定: {r['actual_skill']} (conf={r['confidence']:.2f})")
        except Exception as e:
            logging.error(f"评估失败: {e}", exc_info=args.verbose)
            results.append({
                "pcap": pcap.name,
                "expected_skill": expected,
                "actual_skill": "ERROR",
                "correct": False,
                "error": str(e)
            })
            print(f"  ❌ ERROR: {e}")
    
    # 输出汇总表
    print("\n" + "=" * 80)
    print("评估汇总")
    print("=" * 80)
    print(f"{'#':<4} | {'pcap':<35} | {'expected':<14} | {'actual':<14} | {'conf':<6} | {'结果'}")
    print("-" * 95)
    
    correct_count = 0
    for idx, r in enumerate(results, 1):
        mark = "✅" if r.get("correct") else "❌"
        conf = r.get("confidence", 0.0)
        print(
            f"#{idx:<3} | {r['pcap']:<35} | {r['expected_skill']:<14} | "
            f"{r.get('actual_skill', '?'):<14} | {conf:<6.2f} | {mark}"
        )
        if r.get("correct"):
            correct_count += 1
    
    print("=" * 80)
    accuracy = correct_count / len(results) if results else 0
    print(f"💡 准确率: {correct_count} / {len(results)} = {accuracy:.1%}")
    
    # 保存结果
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"💾 完整结果已保存到 {output_path.resolve()}")
    
    return 0 if correct_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
