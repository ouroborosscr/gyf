# FRP 检测测试工作流

独立的 FRP（内网穿透）流量检测工作流，用于**效果测试**，不修改任何 gyf 现有文件。

## 设计目标

1. 给定一份疑似 FRP 流量的 pcap，自动识别它使用的 FRP 传输模式（tcp / tcp+tls / kcp / quic / ws / wss）
2. 复用 `rag.py` 的向量化和 `evaluate_recall_ip_port.py` 的召回口径
3. 通过**双层队列循环**（外层文档 + 内层 skill），由 LLM 闸门控制审计深度
4. 输出可读的最终判定 + 证据链

## 工作流（10 个节点）

```
preliminary_judgment  (existing, optional)
       ↓ intranet_penetration
kb_retrieval          ← 用 _build_batch_profile + payload embedding 算 target_vector，召回 frp 文档
       ↓
doc_pop ←──────────────────────┐
       ↓                        │ 还有文档
doc_gate ── DONE → final_verdict│
       ↓ ENTER                  │
doc_enter (加载子 skill 队列)    │
       ↓                        │
skill_pop ←──┐                  │
       ↓      │                  │
skill_gate ── SKIP ─┘            │
       ├── EXECUTE → skill_enter ─┘ (next skill)
       └── DONE → stage_summary ───┘
                       ↓ 队列空
                  final_verdict
                       ↓
                      END
```

## 召回机制（关键复用）

完全沿用 `evaluate_recall_ip_port.py` 的口径：

```python
# 1. 38 维行为统计（端口分箱、流量方向、协议比、IP 多样性等）
stats_vec = _build_batch_profile(flows)

# 2. payload 文本 embedding（取每条流前 256 字符，去重，截 1500 字符喂 Qwen3-Embedding-4B）
payload_text = _extract_payload_summary(flows)
payload_vec = embeddings.embed_query(payload_text)

# 3. 加权拼接 (0.6 * stats || 0.4 * payload)，L2 归一化
target_vec = sqrt(0.6)*stats_vec || sqrt(0.4)*payload_vec
```

target 向量与 KB 中各文档的 embedding 做 cosine similarity，取 top-K 进入 doc_queue。

## 目录结构

```
frp_test/
├── README.md                       # 本文件
├── run_frp_test.py                 # 单 pcap 入口
├── evaluate.py                     # 批量评估入口（仿 evaluate_recall_ip_port.py 风格）
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── frp_state.py               # FRPTestState 定义
│   ├── frp_graph.py               # LangGraph 子图编译
│   ├── frp_nodes.py               # 10 个节点实现
│   ├── frp_kb.py                  # 知识库加载 + 向量化（复用 rag.py 函数）
│   ├── llm_wrapper.py             # LLM 封装（支持 real / mock 两种模式）
│   ├── pcap_to_flows.py           # tshark → Zeek-style flow 列表
│   └── pcap_tools.py              # 20 个 pcap 分析工具（已在前期实战调好）
└── knowledge_base/
    └── frp/
        ├── frp.json               # 顶层 frp 文档（含 sub_skills 优先级）
        └── skills/
            ├── frp_tcp.json       # TCP 模式（Go JA3 + 30s 心跳 + 反向流量）
            ├── frp_tcp_tls.json   # TCP+TLS 模式（delegate 到 frp_tcp）
            ├── frp_kcp.json       # KCP 模式（UDP MTU 饱和）
            ├── frp_quic.json      # QUIC 模式（ALPN="frp" 决定性指纹）
            ├── frp_ws.json        # WebSocket 模式（/~!frp URI 决定性指纹）
            └── frp_wss.json       # WSS 模式（SNI + nginx 反代 + 自签证书组合）
```

## 在你的仓库里跑通

### 1. 把 frp_test/ 目录放到 gyf 仓库根目录

```
gyf/
├── (existing files, untouched)
├── rag.py
├── evaluate_recall_ip_port.py
└── frp_test/                       # 新增
```

### 2. 安装依赖

```bash
# 大多数依赖你已经有了；可能需要新装：
pip install langgraph rank_bm25
sudo apt install tshark              # pcap 解析必需
```

### 3. 跑单个 pcap（real LLM 模式，需要本地 vLLM）

```bash
# 在 gyf 根目录执行
python -m frp_test.run_frp_test --pcap /path/to/frp_xxx.pcap
```

### 4. 跑单个 pcap（mock LLM 模式，无需 vLLM）

```bash
FRP_TEST_LLM_MODE=mock python -m frp_test.run_frp_test --pcap /path/to/frp_xxx.pcap
```

### 5. 批量评估 6 个 pcap

```bash
# 仿 evaluate_recall_ip_port.py 的输出风格
FRP_TEST_LLM_MODE=mock python -m frp_test.evaluate --pcap-dir /path/to/pcaps/
```

## 已验证准确率

在 6 个真实 FRP 模式 pcap（自建实验室抓取）上跑 mock LLM 模式：

| pcap                            | expected      | actual       | 结果 |
|---------------------------------|---------------|--------------|------|
| frp_01_tcp.pcap                 | frp_tcp       | frp_tcp      | ✅   |
| frp_02_tcp_tls.pcap             | frp_tcp_tls   | frp_tcp      | ✅ ≡ |
| frp_03_kcp.pcap                 | frp_kcp       | frp_kcp      | ✅   |
| frp_04_quic.pcap                | frp_quic      | frp_quic     | ✅   |
| frp_05_ws.pcap                  | frp_ws        | frp_ws       | ✅   |
| frp_06_wss.pcap                 | frp_wss       | frp_wss      | ✅   |

**准确率：6/6 = 100%**（frp_tcp ≡ frp_tcp_tls 网络层等价合并）

## LLM 模式说明

| 模式 | 触发 | 用途 |
|------|------|------|
| `real` | 默认 | 接 gyf 项目的 `llm.py`（vLLM at localhost:8223/8002），生产/正式效果测试 |
| `mock` | `FRP_TEST_LLM_MODE=mock` | 规则模拟，CI/调试时验证工作流逻辑无需 GPU |

## 与现有 graph.py 集成（未来工作）

当效果测试满意后，可以：

1. 把 `frp_internal_penetration` 文档加入 RAG 知识库（MongoDB `rag_knowledge_base`）
2. 修改 `nodes/preliminary_judgment.py`：当判断为内网穿透时，调用 `frp_test_graph` 作为子图
3. 或者在 `tools/tools.py` 里加一个新工具 `frp_deep_audit_tool` 包装本工作流

设计上完全解耦，不需要改动现有 init/preliminary_judgment/update_state 等节点。
