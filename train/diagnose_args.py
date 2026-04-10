"""
抓取 transformers 传给 flash_attn 的实际参数
用法: CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python diagnose_args.py
"""
import torch
import flash_attn.flash_attn_interface as fa_interface

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"

# ============================================================
# Monkey-patch flash_attn_varlen_func 来截获参数
# ============================================================
_original_varlen_func = fa_interface.flash_attn_varlen_func
call_count = [0]

def patched_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
    call_count[0] += 1
    if call_count[0] <= 2:  # 只打印前两次调用
        print(f"\n{'='*60}")
        print(f"flash_attn_varlen_func 第 {call_count[0]} 次调用")
        print(f"{'='*60}")
        print(f"  q:  shape={q.shape}, dtype={q.dtype}, device={q.device}, contiguous={q.is_contiguous()}")
        print(f"  k:  shape={k.shape}, dtype={k.dtype}, device={k.device}, contiguous={k.is_contiguous()}")
        print(f"  v:  shape={v.shape}, dtype={v.dtype}, device={v.device}, contiguous={v.is_contiguous()}")
        print(f"  cu_seqlens_q: {cu_seqlens_q} (dtype={cu_seqlens_q.dtype}, device={cu_seqlens_q.device})")
        print(f"  cu_seqlens_k: {cu_seqlens_k} (dtype={cu_seqlens_k.dtype}, device={cu_seqlens_k.device})")
        print(f"  max_seqlen_q: {max_seqlen_q} (type={type(max_seqlen_q).__name__})")
        print(f"  max_seqlen_k: {max_seqlen_k} (type={type(max_seqlen_k).__name__})")
        print(f"  kwargs: {list(kwargs.keys())}")
        for kk, vv in kwargs.items():
            if isinstance(vv, torch.Tensor):
                print(f"    {kk}: shape={vv.shape}, dtype={vv.dtype}")
            else:
                print(f"    {kk}: {vv}")

        # 检查数值
        print(f"\n  q  stats: min={q.min().item():.4f}, max={q.max().item():.4f}, "
              f"nan={torch.isnan(q).any().item()}, inf={torch.isinf(q).any().item()}")
        print(f"  k  stats: min={k.min().item():.4f}, max={k.max().item():.4f}, "
              f"nan={torch.isnan(k).any().item()}, inf={torch.isinf(k).any().item()}")
        print(f"  v  stats: min={v.min().item():.4f}, max={v.max().item():.4f}, "
              f"nan={torch.isnan(v).any().item()}, inf={torch.isinf(v).any().item()}")

        # 验证 cu_seqlens 合法性
        total_q = q.shape[0]
        total_k = k.shape[0]
        last_cu_q = cu_seqlens_q[-1].item()
        last_cu_k = cu_seqlens_k[-1].item()
        print(f"\n  q total tokens: {total_q}, cu_seqlens_q[-1]: {last_cu_q}, match: {total_q == last_cu_q}")
        print(f"  k total tokens: {total_k}, cu_seqlens_k[-1]: {last_cu_k}, match: {total_k == last_cu_k}")

        if total_q != last_cu_q:
            print(f"  ⚠️ cu_seqlens_q[-1] ({last_cu_q}) != q.shape[0] ({total_q})  ← 这会导致越界!")
        if total_k != last_cu_k:
            print(f"  ⚠️ cu_seqlens_k[-1] ({last_cu_k}) != k.shape[0] ({total_k})  ← 这会导致越界!")

        # 验证 head 数量关系
        num_heads_q = q.shape[1]
        num_heads_k = k.shape[1]
        print(f"\n  num_heads_q: {num_heads_q}, num_heads_k: {num_heads_k}")
        if num_heads_q != num_heads_k:
            if num_heads_q % num_heads_k == 0:
                print(f"  GQA 模式: {num_heads_q}/{num_heads_k} = {num_heads_q // num_heads_k} repeats")
            else:
                print(f"  ⚠️ num_heads_q ({num_heads_q}) 不能被 num_heads_k ({num_heads_k}) 整除!")

    return _original_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs)

fa_interface.flash_attn_varlen_func = patched_varlen_func

# 同时 patch _flash_attention_forward 里可能直接调用的路径
_original_varlen_fwd = None
try:
    import flash_attn.flash_attn_interface as _fai
    if hasattr(_fai, '_flash_attn_varlen_forward'):
        _original_varlen_fwd = _fai._flash_attn_varlen_forward
        def patched_low_level(*args, **kwargs):
            print(f"\n  [底层] _flash_attn_varlen_forward 被调用, args数量={len(args)}")
            return _original_varlen_fwd(*args, **kwargs)
        _fai._flash_attn_varlen_forward = patched_low_level
except:
    pass

# 也 patch transformers 的入口
try:
    import transformers.modeling_flash_attention_utils as tfau
    _orig_flash_fwd = tfau._flash_attention_forward
    def patched_tfau_flash_fwd(*args, **kwargs):
        print(f"\n[transformers._flash_attention_forward] 被调用")
        print(f"  args 数量: {len(args)}")
        print(f"  kwargs keys: {list(kwargs.keys())}")
        for kk, vv in kwargs.items():
            if isinstance(vv, torch.Tensor):
                print(f"    {kk}: shape={vv.shape}, dtype={vv.dtype}")
            elif vv is not None:
                print(f"    {kk}: {vv}")
        # 检查 query_states (通常是第一个 arg)
        if len(args) >= 3:
            qs, ks, vs = args[0], args[1], args[2]
            if isinstance(qs, torch.Tensor):
                print(f"  query_states:  shape={qs.shape}, dtype={qs.dtype}")
                print(f"  key_states:    shape={ks.shape}, dtype={ks.dtype}")
                print(f"  value_states:  shape={vs.shape}, dtype={vs.dtype}")
        return _orig_flash_fwd(*args, **kwargs)
    tfau._flash_attention_forward = patched_tfau_flash_fwd
except Exception as e:
    print(f"patch transformers 入口失败: {e}")

# ============================================================
# 加载模型并触发 forward
# ============================================================
from transformers import AutoTokenizer, AutoModelForCausalLM

print("加载模型 (bf16, flash_attention_2)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
)

# 打印模型的关键 config
cfg = model.config
print(f"\n模型 config:")
print(f"  num_attention_heads:   {getattr(cfg, 'num_attention_heads', '?')}")
print(f"  num_key_value_heads:   {getattr(cfg, 'num_key_value_heads', '?')}")
print(f"  hidden_size:           {getattr(cfg, 'hidden_size', '?')}")
print(f"  head_dim:              {getattr(cfg, 'head_dim', '?')}")
print(f"  sliding_window:        {getattr(cfg, 'sliding_window', '?')}")
print(f"  max_position_embeddings: {getattr(cfg, 'max_position_embeddings', '?')}")
print(f"  _attn_implementation:  {getattr(cfg, '_attn_implementation', '?')}")
print(f"  rope_type:             {getattr(cfg, 'rope_scaling', {}).get('rope_type', '?') if getattr(cfg, 'rope_scaling', None) else 'default'}")

if hasattr(cfg, "sliding_window") and cfg.sliding_window is not None:
    print(f"\n  ⚠️ sliding_window={cfg.sliding_window} — 这可能影响 flash_attn 调用!")
    print(f"  尝试设置为 None...")
    cfg.sliding_window = None

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
input_ids = torch.randint(0, 1000, (1, 256), device="cuda:0")

print(f"\n开始 forward (input shape={input_ids.shape})...")
try:
    with torch.no_grad():
        out = model(input_ids=input_ids)
    torch.cuda.synchronize()
    print(f"\n✅ forward 成功, logits shape={out.logits.shape}")
except Exception as e:
    print(f"\n❌ forward 崩溃: {e}")
    print(f"   崩溃发生在第 {call_count[0]} 次 flash_attn 调用处")