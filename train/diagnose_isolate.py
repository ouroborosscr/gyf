"""
精准隔离诊断：逐步排查 flash_attn 崩溃点
用法: CUDA_VISIBLE_DEVICES=1 python diagnose_isolate.py
"""
import torch
import sys

print("=" * 60)
print("Step 0: 环境版本")
print("=" * 60)
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA (torch): {torch.version.cuda}")

try:
    import flash_attn; print(f"flash_attn:   {flash_attn.__version__}")
except Exception as e:
    print(f"flash_attn:   {e}"); sys.exit(1)

try:
    import transformers; print(f"transformers: {transformers.__version__}")
except: pass
try:
    import bitsandbytes; print(f"bitsandbytes: {bitsandbytes.__version__}")
except: pass
try:
    import peft; print(f"peft:         {peft.__version__}")
except: pass

props = torch.cuda.get_device_properties(0)
print(f"GPU:          {props.name}, {props.total_memory/1024**3:.1f}GB, SM {props.major}.{props.minor}")

# ============================================================
print("\n" + "=" * 60)
print("Step 1: 裸测 flash_attn_varlen_func (纯随机 tensor，不涉及任何模型)")
print("=" * 60)

from flash_attn import flash_attn_varlen_func

device = "cuda:0"
dtype = torch.bfloat16
# Qwen3.5-9B 参数: num_heads=48, num_kv_heads=8, head_dim=128
num_heads = 48
head_dim = 128
seq_len = 256

q = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)
k = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)
v = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)
cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

try:
    out = flash_attn_varlen_func(q, k, v, cu, cu, seq_len, seq_len, causal=True)
    torch.cuda.synchronize()
    print(f"  ✅ 裸调用通过 (shape={out.shape})")
    bare_ok = True
except Exception as e:
    print(f"  ❌ 裸调用就崩了: {e}")
    print("  → flash_attn 安装本身有问题，和模型无关")
    print("  → 请检查: flash_attn 是否匹配当前 CUDA 版本和 GPU 架构")
    print(f"  → 当前 CUDA: {torch.version.cuda}, GPU SM: {props.major}.{props.minor}")
    bare_ok = False

torch.cuda.empty_cache()

if not bare_ok:
    # 再试 flash_attn_func (非 varlen 版本)
    print("\n  额外测试: flash_attn_func (非 varlen)")
    try:
        from flash_attn import flash_attn_func
        q2 = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k2 = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v2 = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        out2 = flash_attn_func(q2, k2, v2, causal=True)
        torch.cuda.synchronize()
        print(f"  ✅ flash_attn_func 通过")
    except Exception as e:
        print(f"  ❌ flash_attn_func 也崩: {e}")
    sys.exit(1)

# ============================================================
print("\n" + "=" * 60)
print("Step 2: bf16 模型 + flash_attn (不量化)")
print("=" * 60)

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

try:
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    if hasattr(model_bf16.config, "sliding_window"):
        model_bf16.config.sliding_window = None

    input_ids = torch.randint(0, 1000, (1, 256), device=device)
    with torch.no_grad():
        out = model_bf16(input_ids=input_ids)
    torch.cuda.synchronize()
    print(f"  ✅ bf16 + flash_attn forward 通过")
    bf16_ok = True
except Exception as e:
    print(f"  ❌ bf16 + flash_attn forward 崩溃: {e}")
    bf16_ok = False

if 'model_bf16' in dir():
    del model_bf16
torch.cuda.empty_cache()
import gc; gc.collect()

# ============================================================
print("\n" + "=" * 60)
print("Step 3: 4-bit 模型 + SDPA (不用 flash_attn)")
print("=" * 60)

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

try:
    model_4bit_sdpa = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",  # 不用 flash_attn
        low_cpu_mem_usage=True,
    )
    if hasattr(model_4bit_sdpa.config, "sliding_window"):
        model_4bit_sdpa.config.sliding_window = None

    input_ids = torch.randint(0, 1000, (1, 256), device=device)
    with torch.no_grad():
        out = model_4bit_sdpa(input_ids=input_ids)
    torch.cuda.synchronize()
    print(f"  ✅ 4-bit + SDPA forward 通过")
except Exception as e:
    print(f"  ❌ 4-bit + SDPA forward 崩溃: {e}")

if 'model_4bit_sdpa' in dir():
    del model_4bit_sdpa
torch.cuda.empty_cache(); gc.collect()

# ============================================================
print("\n" + "=" * 60)
print("Step 4: 4-bit 模型 + flash_attn (你实际的配置)")
print("=" * 60)

try:
    model_4bit_fa = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    if hasattr(model_4bit_fa.config, "sliding_window"):
        model_4bit_fa.config.sliding_window = None

    input_ids = torch.randint(0, 1000, (1, 256), device=device)
    
    with torch.no_grad():
        out = model_4bit_fa(input_ids=input_ids)
    torch.cuda.synchronize()
    print(f"  ✅ 4-bit + flash_attn forward 通过")
except Exception as e:
    print(f"  ❌ 4-bit + flash_attn forward 崩溃: {e}")
    print(f"  → 前面 Step 1 裸调用通过但这里崩了，说明是 4-bit 量化后传给 flash_attn 的 tensor 有问题")

    # 深入检查：抓中间 tensor 的状态
    print("\n  --- 深入检查：hook 第一层 attention 的输入 ---")
    try:
        model_4bit_fa2 = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
        if hasattr(model_4bit_fa2.config, "sliding_window"):
            model_4bit_fa2.config.sliding_window = None

        # hook 第一层 self_attn 的输入
        attn_layer = model_4bit_fa2.model.layers[0].self_attn
        
        hook_data = {}
        def inspect_hook(module, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            if hs is not None:
                hook_data["hidden_states_dtype"] = hs.dtype
                hook_data["hidden_states_device"] = hs.device
                hook_data["hidden_states_shape"] = hs.shape
                hook_data["hidden_states_contiguous"] = hs.is_contiguous()
                hook_data["hidden_states_has_nan"] = torch.isnan(hs).any().item()
                hook_data["hidden_states_has_inf"] = torch.isinf(hs).any().item()
                hook_data["hidden_states_min"] = hs.min().item()
                hook_data["hidden_states_max"] = hs.max().item()
            
            # 检查 q_proj 权重状态
            if hasattr(module, "q_proj"):
                w = module.q_proj.weight
                hook_data["q_proj_dtype"] = str(w.dtype)
                hook_data["q_proj_device"] = str(w.device)
                if hasattr(w, "quant_state"):
                    hook_data["q_proj_quantized"] = True
                    hook_data["q_proj_quant_type"] = str(w.quant_state.quant_type) if hasattr(w.quant_state, "quant_type") else "unknown"
                else:
                    hook_data["q_proj_quantized"] = False

        attn_layer.register_forward_pre_hook(inspect_hook, with_kwargs=True)

        input_ids = torch.randint(0, 1000, (1, 64), device=device)
        try:
            with torch.no_grad():
                model_4bit_fa2(input_ids=input_ids)
        except:
            pass  # 预期会崩，但 hook 应该已经拿到数据了

        if hook_data:
            print(f"  hidden_states: dtype={hook_data.get('hidden_states_dtype')}, "
                  f"device={hook_data.get('hidden_states_device')}, "
                  f"shape={hook_data.get('hidden_states_shape')}")
            print(f"  contiguous={hook_data.get('hidden_states_contiguous')}, "
                  f"has_nan={hook_data.get('hidden_states_has_nan')}, "
                  f"has_inf={hook_data.get('hidden_states_has_inf')}")
            print(f"  value range: [{hook_data.get('hidden_states_min')}, {hook_data.get('hidden_states_max')}]")
            print(f"  q_proj: dtype={hook_data.get('q_proj_dtype')}, "
                  f"device={hook_data.get('q_proj_device')}, "
                  f"quantized={hook_data.get('q_proj_quantized')}")
            if hook_data.get("q_proj_quantized"):
                print(f"  q_proj quant_type: {hook_data.get('q_proj_quant_type')}")
        else:
            print("  hook 未触发（崩溃在 hook 之前）")
    except Exception as e2:
        print(f"  深入检查也失败: {e2}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)