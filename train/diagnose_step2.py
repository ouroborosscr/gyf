"""
第二轮诊断：精确定位 4-bit + flash_attn 崩溃的根因
用法：
  CUDA_LAUNCH_BLOCKING=1 LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 CUDA_VISIBLE_DEVICES=1 python diagnose_step2.py
"""

import torch
import gc

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"
TEST_SEQ_LEN = 2048  # 上一轮 4096 崩了，用 2048 做安全边界测试

def clean():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ============================================================
print("=" * 60)
print("测试 A: 4-bit 量化 + SDPA (不用 flash_attn)")
print("   目的：排除 flash_attn，看量化模型本身是否正常")
print("=" * 60)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_sdpa = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",  # 用 PyTorch 原生 SDPA
        low_cpu_mem_usage=True,
    )
    if hasattr(model_sdpa.config, "sliding_window"):
        model_sdpa.config.sliding_window = None

    for slen in [512, 2048, 4096]:
        try:
            ids = torch.randint(0, 1000, (1, slen), device="cuda:0")
            with torch.no_grad():
                out = model_sdpa(input_ids=ids, attention_mask=torch.ones_like(ids))
            torch.cuda.synchronize()
            print(f"  4bit+SDPA seq_len={slen}: ✅ 通过")
        except Exception as e:
            print(f"  4bit+SDPA seq_len={slen}: ❌ 失败 - {e}")
        clean()

    del model_sdpa
    clean()

except Exception as e:
    print(f"  测试 A 整体失败: {e}")


# ============================================================
print("\n" + "=" * 60)
print("测试 B: bf16 全精度 + flash_attention_2 (不量化)")
print("   目的：排除量化，看 flash_attn 和模型本身是否正常")
print("=" * 60)

try:
    model_fa2 = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    if hasattr(model_fa2.config, "sliding_window"):
        model_fa2.config.sliding_window = None

    for slen in [512, 2048, 4096]:
        try:
            ids = torch.randint(0, 1000, (1, slen), device="cuda:0")
            with torch.no_grad():
                out = model_fa2(input_ids=ids, attention_mask=torch.ones_like(ids))
            torch.cuda.synchronize()
            print(f"  bf16+FA2 seq_len={slen}: ✅ 通过")
        except Exception as e:
            print(f"  bf16+FA2 seq_len={slen}: ❌ 失败 - {e}")
        clean()

    del model_fa2
    clean()

except Exception as e:
    print(f"  测试 B 整体失败: {e}")


# ============================================================
print("\n" + "=" * 60)
print("测试 C: 4-bit + flash_attention_2, 用 hook 检查 attention 输入")
print("   目的：捕获崩溃前传入 flash_attn 的 hidden_states 状态")
print("=" * 60)

try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_debug = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    if hasattr(model_debug.config, "sliding_window"):
        model_debug.config.sliding_window = None

    # 注册 hook 检查每一层 attention 输入
    hook_log = {}

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs):
            # self_attn 的输入是 hidden_states
            hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            if hs is not None:
                hook_log[layer_idx] = {
                    "shape": tuple(hs.shape),
                    "dtype": str(hs.dtype),
                    "device": str(hs.device),
                    "contiguous": hs.is_contiguous(),
                    "has_nan": bool(torch.isnan(hs).any()),
                    "has_inf": bool(torch.isinf(hs).any()),
                    "min": float(hs.min()),
                    "max": float(hs.max()),
                    "abs_mean": float(hs.abs().mean()),
                }
        return hook_fn

    handles = []
    for i, layer in enumerate(model_debug.model.layers):
        h = layer.self_attn.register_forward_pre_hook(make_hook(i), with_kwargs=True)
        handles.append(h)

    # 先测短序列收集正常 baseline
    print("\n  --- seq_len=128 (baseline) ---")
    try:
        hook_log.clear()
        ids = torch.randint(0, 1000, (1, 128), device="cuda:0")
        with torch.no_grad():
            out = model_debug(input_ids=ids, attention_mask=torch.ones_like(ids))
        torch.cuda.synchronize()
        
        nan_layers = [k for k, v in hook_log.items() if v["has_nan"]]
        inf_layers = [k for k, v in hook_log.items() if v["has_inf"]]
        print(f"  ✅ 通过, 总层数={len(hook_log)}")
        print(f"  含 NaN 的层: {nan_layers if nan_layers else '无'}")
        print(f"  含 Inf 的层: {inf_layers if inf_layers else '无'}")
        
        # 打印前3层和后3层的统计
        for idx in list(range(min(3, len(hook_log)))) + list(range(max(0, len(hook_log)-3), len(hook_log))):
            if idx in hook_log:
                v = hook_log[idx]
                print(f"    Layer {idx:>2}: shape={v['shape']}, contig={v['contiguous']}, "
                      f"nan={v['has_nan']}, inf={v['has_inf']}, "
                      f"range=[{v['min']:.4f}, {v['max']:.4f}], abs_mean={v['abs_mean']:.4f}")
    except Exception as e:
        print(f"  ❌ 失败 - {e}")
    clean()

    # 测可能崩溃的长序列
    for test_slen in [512, 1024, 2048, 4096]:
        print(f"\n  --- seq_len={test_slen} ---")
        try:
            hook_log.clear()
            ids = torch.randint(0, 1000, (1, test_slen), device="cuda:0")
            with torch.no_grad():
                out = model_debug(input_ids=ids, attention_mask=torch.ones_like(ids))
            torch.cuda.synchronize()
            
            nan_layers = [k for k, v in hook_log.items() if v["has_nan"]]
            inf_layers = [k for k, v in hook_log.items() if v["has_inf"]]
            print(f"  ✅ 通过, 总层数={len(hook_log)}")
            print(f"  含 NaN 的层: {nan_layers if nan_layers else '无'}")
            print(f"  含 Inf 的层: {inf_layers if inf_layers else '无'}")
        except Exception as e:
            # 崩溃了，看看 hook_log 里最后记录到哪一层
            print(f"  ❌ 崩溃在第 {len(hook_log)} 层之后")
            print(f"  最后成功记录的层:")
            last_keys = sorted(hook_log.keys())[-3:] if hook_log else []
            for idx in last_keys:
                v = hook_log[idx]
                print(f"    Layer {idx:>2}: shape={v['shape']}, contig={v['contiguous']}, "
                      f"nan={v['has_nan']}, inf={v['has_inf']}, "
                      f"range=[{v['min']:.4f}, {v['max']:.4f}], abs_mean={v['abs_mean']:.4f}")
            print(f"  错误: {e}")
            break  # CUDA 状态已损坏，不能继续
        clean()

    for h in handles:
        h.remove()
    del model_debug
    clean()

except Exception as e:
    import traceback
    print(f"  测试 C 整体失败: {e}")
    traceback.print_exc()


# ============================================================
print("\n" + "=" * 60)
print("测试 D: 4-bit + flash_attention_2, 用真实 tokenized 文本")
print("   目的：排除 randint token ID 命中异常嵌入行的可能")
print("=" * 60)

try:
    # 需要重新加载（CUDA 状态可能已损坏）
    # 先检查 CUDA 是否还能用
    torch.cuda.synchronize()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_real = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    if hasattr(model_real.config, "sliding_window"):
        model_real.config.sliding_window = None
    
    # 用真实文本填充到不同长度
    base_text = "The quick brown fox jumps over the lazy dog. " * 200
    
    for target_len in [512, 2048, 4096]:
        try:
            tokens = tokenizer(base_text, return_tensors="pt", truncation=True, max_length=target_len, padding="max_length")
            input_ids = tokens["input_ids"].to("cuda:0")
            attn_mask = tokens["attention_mask"].to("cuda:0")
            actual_len = input_ids.shape[1]
            
            with torch.no_grad():
                out = model_real(input_ids=input_ids, attention_mask=attn_mask)
            torch.cuda.synchronize()
            print(f"  真实文本 seq_len={actual_len}: ✅ 通过")
        except Exception as e:
            print(f"  真实文本 seq_len={target_len}: ❌ 失败 - {e}")
            break
        clean()

    del model_real
    clean()

except Exception as e:
    print(f"  测试 D 跳过（CUDA 状态已损坏或其他错误）: {e}")


print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)