"""
Flash Attention + QLoRA 崩溃诊断脚本
用法（和你训练时一样的环境变量）：
  CUDA_VISIBLE_DEVICES=1 python diagnose_flash_attn.py
"""

import torch
import sys
import os

print("=" * 60)
print("1. 环境信息")
print("=" * 60)
print(f"Python:       {sys.version}")
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA (torch): {torch.version.cuda}")

try:
    import flash_attn
    print(f"flash_attn:   {flash_attn.__version__}")
except Exception as e:
    print(f"flash_attn:   导入失败 - {e}")

try:
    import transformers
    print(f"transformers: {transformers.__version__}")
except:
    pass

try:
    import trl
    print(f"trl:          {trl.__version__}")
except:
    pass

try:
    import peft
    print(f"peft:         {peft.__version__}")
except:
    pass

try:
    import bitsandbytes
    print(f"bitsandbytes: {bitsandbytes.__version__}")
except:
    pass

try:
    import deepspeed
    print(f"deepspeed:    {deepspeed.__version__}")
except:
    pass

gpu_count = torch.cuda.device_count()
print(f"\n可见 GPU 数量: {gpu_count}")
for i in range(gpu_count):
    props = torch.cuda.get_device_properties(i)
    total_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1024**3
    print(f"  GPU {i}: {props.name}, {total_gb:.1f} GB, SM {props.major}.{props.minor}")


print("\n" + "=" * 60)
print("2. 单独测试 flash_attn_varlen_func (不涉及模型)")
print("=" * 60)

try:
    from flash_attn import flash_attn_varlen_func

    device = "cuda:0"
    dtype = torch.bfloat16

    # 模拟 Qwen3.5-9B 的注意力头参数
    # Qwen3.5-9B: num_heads=48, num_kv_heads=8, head_dim=128
    batch_size = 1
    num_heads = 48
    num_kv_heads = 8
    head_dim = 128

    for seq_len in [512, 2048, 4096, 8192]:
        try:
            q = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)
            k = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
            v = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

            cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            max_seqlen = seq_len

            out = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
            )
            torch.cuda.synchronize()
            print(f"  seq_len={seq_len:>5}: ✅ 通过, output shape={out.shape}")
        except Exception as e:
            print(f"  seq_len={seq_len:>5}: ❌ 失败 - {e}")

        torch.cuda.empty_cache()

except Exception as e:
    print(f"  flash_attn_varlen_func 测试整体失败: {e}")


print("\n" + "=" * 60)
print("3. 测试 flash_attn + GQA repeat (模拟实际推理路径)")
print("=" * 60)

try:
    from flash_attn import flash_attn_varlen_func

    device = "cuda:0"
    dtype = torch.bfloat16
    num_heads = 48
    num_kv_heads = 8
    head_dim = 128
    seq_len = 4096

    q = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)
    # 用 repeat_interleave 模拟 GQA 展开（transformers 的实际做法）
    k_base = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_base = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    
    n_rep = num_heads // num_kv_heads  # =6
    k = k_base[:, :, None, :].expand(-1, num_kv_heads, n_rep, head_dim).reshape(seq_len, num_heads, head_dim)
    v = v_base[:, :, None, :].expand(-1, num_kv_heads, n_rep, head_dim).reshape(seq_len, num_heads, head_dim)

    # 关键：展开后的 k, v 可能不是内存连续的
    print(f"  q.is_contiguous(): {q.is_contiguous()}")
    print(f"  k.is_contiguous(): {k.is_contiguous()} (GQA expanded)")
    print(f"  v.is_contiguous(): {v.is_contiguous()} (GQA expanded)")

    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    # 先试不 contiguous 的
    try:
        out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, seq_len, seq_len, causal=True)
        torch.cuda.synchronize()
        print(f"  非连续 tensor 测试: ✅ 通过")
    except Exception as e:
        print(f"  非连续 tensor 测试: ❌ 失败 - {e}")

    # 再试 contiguous 的
    try:
        out = flash_attn_varlen_func(q, k.contiguous(), v.contiguous(), cu_seqlens, cu_seqlens, seq_len, seq_len, causal=True)
        torch.cuda.synchronize()
        print(f"  连续 tensor 测试:   ✅ 通过")
    except Exception as e:
        print(f"  连续 tensor 测试:   ❌ 失败 - {e}")

    torch.cuda.empty_cache()

except Exception as e:
    print(f"  GQA 测试整体失败: {e}")


print("\n" + "=" * 60)
print("4. 加载 Qwen3.5-9B (4-bit) 并做一次真实 forward")
print("=" * 60)

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"

try:
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

    print(f"  模型加载成功, dtype={model.dtype}")
    print(f"  attn_implementation: {getattr(model.config, '_attn_implementation', 'unknown')}")

    # 测试不同长度的 forward
    for test_len in [128, 512, 2048, 4096]:
        try:
            input_ids = torch.randint(0, 1000, (1, test_len), device="cuda:0")
            attention_mask = torch.ones_like(input_ids)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()

            mem_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  forward seq_len={test_len:>5}: ✅ 通过, logits shape={out.logits.shape}, peak mem={mem_used:.1f}GB")
        except Exception as e:
            print(f"  forward seq_len={test_len:>5}: ❌ 失败 - {e}")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 测试带梯度的 forward（模拟训练）
    print("\n  --- 带梯度 forward（模拟训练 loss 计算） ---")
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, peft_config)
    peft_model.train()

    for test_len in [512, 2048, 4096]:
        try:
            input_ids = torch.randint(0, 1000, (1, test_len), device="cuda:0")
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()

            out = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.cuda.synchronize()

            mem_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  train  seq_len={test_len:>5}: ✅ 通过, loss={loss.item():.4f}, peak mem={mem_used:.1f}GB")
        except Exception as e:
            print(f"  train  seq_len={test_len:>5}: ❌ 失败 - {e}")

        # 清理梯度和缓存
        peft_model.zero_grad()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 测试带 gradient_checkpointing 的 forward（和你训练配置一致）
    print("\n  --- gradient_checkpointing + train forward ---")
    peft_model.gradient_checkpointing_enable()

    for test_len in [512, 2048, 4096]:
        try:
            input_ids = torch.randint(0, 1000, (1, test_len), device="cuda:0")
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()

            out = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.cuda.synchronize()

            mem_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  gc+train seq_len={test_len:>5}: ✅ 通过, loss={loss.item():.4f}, peak mem={mem_used:.1f}GB")
        except Exception as e:
            print(f"  gc+train seq_len={test_len:>5}: ❌ 失败 - {e}")

        peft_model.zero_grad()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

except Exception as e:
    import traceback
    print(f"  模型测试失败: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)