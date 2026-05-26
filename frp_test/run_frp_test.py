#!/usr/bin/env python3
"""
run_frp_test.py — FRP 测试工作流入口

用法:
    # 在 gyf 项目根目录执行
    python -m frp_test.run_frp_test --pcap /path/to/frp_xxx.pcap
    
    # 用 mock LLM 模式（无需 vLLM 服务）：
    FRP_TEST_LLM_MODE=mock python -m frp_test.run_frp_test --pcap xxx.pcap
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# 确保能 import 到 gyf 项目根（用于复用 rag.py, llm.py）
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="FRP 测试工作流")
    parser.add_argument("--pcap", required=True, help="待测 pcap 文件路径")
    parser.add_argument("--expected", default=None, help="期望的 skill_id (用于评估)")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    parser.add_argument("--verbose-llm", action="store_true",
                        help="打印每次 LLM 调用的完整 prompt + 思考链 + 最终输出（debug 利器）")
    parser.add_argument("--output", default=None, help="结果 JSON 输出路径")
    args = parser.parse_args()
    
    if args.verbose_llm:
        os.environ["FRP_TEST_VERBOSE_LLM"] = "1"
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s"
    )
    
    pcap_path = str(Path(args.pcap).resolve())
    if not Path(pcap_path).exists():
        print(f"❌ pcap 不存在: {pcap_path}")
        sys.exit(1)
    
    # 构建图
    from core.frp_graph import get_frp_test_graph
    graph = get_frp_test_graph()
    
    # 初始 state
    init_state = {
        "pcap_path": pcap_path,
        "expected_label": args.expected,
        "evidence": [],
        "stage_summaries": [],
        "messages": []
    }
    
    print("=" * 70)
    print(f"FRP 测试工作流启动")
    print(f"  pcap: {pcap_path}")
    print(f"  LLM 模式: {os.environ.get('FRP_TEST_LLM_MODE', 'real')}")
    if args.expected:
        print(f"  期望 skill: {args.expected}")
    print("=" * 70)
    
    # 跑工作流
    final_state = None
    for event in graph.stream(init_state):
        for node_name, output in event.items():
            print(f"\n>>> 节点 [{node_name}] 完成")
            final_state = output
    
    # 拿最终判定
    # 由于 stream 模式只返回每步的增量，最后一次更新含 final_verdict
    print("\n" + "=" * 70)
    print("最终判定:")
    print("=" * 70)
    
    if final_state and "final_verdict" in final_state:
        verdict = final_state["final_verdict"]
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
        
        # 评估正确性
        if args.expected and verdict.get("matched_sub_skill") == args.expected:
            print(f"\n✅ 判定正确（期望 {args.expected}）")
        elif args.expected:
            print(f"\n❌ 判定错误：期望 {args.expected}, 实际 {verdict.get('matched_sub_skill')}")
        
        # 保存
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(verdict, f, ensure_ascii=False, indent=2)
            print(f"\n💾 结果已保存到 {args.output}")
    else:
        print("⚠️ 未生成最终判定")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
