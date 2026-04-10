"""
诊断脚本 2：钩入 attention 层，检查传给 flash_attn 的 q/k/v 状态
用法：
  CUDA_LAUNCH_BLOCKING=1 LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
  CUDA_VISIBLE_DEVICES=1 python diagnose_flash_attn_2.py
"""

import torch
import sys
import functools

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"

# ============================================================
# 1. 加载模型
# ============================================================
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
)

if hasattr(model.config, "sliding_window"):
    model.config.sliding_window = None

print("模型加载完成")

# ============================================================
# 2. Monkey-patch flash_attn_varlen_func 来拦截输入
# ============================================================
import flash_attn.flash_attn_interface as fa_interface

_original_varlen_func = fa_interface.flash_attn_varlen_func
_call_count = 0
_crashed = False

def patched_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
    global _call_count, _crashed
    _call_count += 1
    layer_id = _call_count

    def check_tensor(name, t):
        issues = []
        if not t.is_contiguous():
            issues.append("NOT contiguous")
        if t.dtype != torch.bfloat16:
            issues.append(f"dtype={t.dtype}")
        if not t.is_cuda:
            issues.append("NOT on CUDA")

        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        if has_nan:
            nan_count = torch.isnan(t).sum().item()
            issues.append(f"has {nan_count} NaN")
        if has_inf:
            inf_count = torch.isinf(t).sum().item()
            issues.append(f"has {inf_count} Inf")

        abs_max = t.abs().max().item()
        abs_mean = t.abs().mean().item()

        status = "⚠️ " + ", ".join(issues) if issues else "✅"
        print(f"    {name}: shape={list(t.shape)}, stride={t.stride()}, "
              f"abs_max={abs_max:.4f}, abs_mean={abs_mean:.6f} {status}")
        return len(issues) == 0

    print(f"\n  [Layer {layer_id}] flash_attn_varlen_func 被调用")
    print(f"    cu_seqlens_q={cu_seqlens_q.tolist()}, max_seqlen_q={max_seqlen_q}")
    print(f"    cu_seqlens_k={cu_seqlens_k.tolist()}, max_seqlen_k={max_seqlen_k}")
    print(f"    kwargs keys: {list(kwargs.keys())}")

    q_ok = check_tensor("q", q)
    k_ok = check_tensor("k", k)
    v_ok = check_tensor("v", v)

    # 检查 cu_seqlens 合理性
    total_q = q.shape[0]
    total_k = k.shape[0]
    expected_q = cu_seqlens_q[-1].item()
    expected_k = cu_seqlens_k[-1].item()
    if total_q != expected_q:
        print(f"    ⚠️ q 总 token 数 {total_q} != cu_seqlens_q[-1] {expected_q}")
    if total_k != expected_k:
        print(f"    ⚠️ k 总 token 数 {total_k} != cu_seqlens_k[-1] {expected_k}")

    # 检查 max_seqlen 是否超过实际长度
    seqlens_q = []
    for i in range(len(cu_seqlens_q) - 1):
        seqlens_q.append(cu_seqlens_q[i+1].item() - cu_seqlens_q[i].item())
    actual_max_q = max(seqlens_q) if seqlens_q else 0
    if max_seqlen_q != actual_max_q:
        print(f"    ⚠️ max_seqlen_q={max_seqlen_q} 但实际最大 seqlen={actual_max_q}")

    return _original_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs)

fa_interface.flash_attn_varlen_func = patched_varlen_func

# 同时 patch transformers 里已经 import 的引用
import transformers.modeling_flash_attention_utils as tfa_utils
if hasattr(tfa_utils, 'flash_attn_varlen_func'):
    tfa_utils.flash_attn_varlen_func = patched_varlen_func
# 有些 transformers 版本用 _flash_attention_forward 里直接引用
# 需要找到它用的那个
import importlib
try:
    tfa_utils_mod = sys.modules.get('transformers.modeling_flash_attention_utils')
    if tfa_utils_mod and hasattr(tfa_utils_mod, 'flash_attn_varlen_func'):
        tfa_utils_mod.flash_attn_varlen_func = patched_varlen_func
except:
    pass

# patch flash_attn 模块本身，确保无论哪条路径调用都会被拦截
import flash_attn
flash_attn.flash_attn_varlen_func = patched_varlen_func

print("已 patch flash_attn_varlen_func\n")

# ============================================================
# 3. 用不同长度做 forward，观察输出
# ============================================================
print("=" * 60)
print("开始测试 forward (每次只测一个长度，崩溃时能看到最后一层的状态)")
print("=" * 60)

for seq_len in [256, 512, 1024, 2048, 4096]:
    print(f"\n{'='*60}")
    print(f"测试 seq_len={seq_len}")
    print(f"{'='*60}")

    _call_count = 0

    try:
        input_ids = torch.randint(0, 1000, (1, seq_len), device="cuda:0")
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()

        logits = out.logits
        logits_has_nan = torch.isnan(logits).any().item()
        logits_has_inf = torch.isinf(logits).any().item()
        print(f"\n  结果: ✅ 通过, 共调用 flash_attn {_call_count} 次")
        print(f"  logits: shape={list(logits.shape)}, has_nan={logits_has_nan}, has_inf={logits_has_inf}")

    except Exception as e:
        print(f"\n  结果: ❌ 在第 {_call_count} 次调用 flash_attn 时崩溃")
        print(f"  错误: {e}")
        break  # CUDA 坏了，后续测试无意义

    torch.cuda.empty_cache()