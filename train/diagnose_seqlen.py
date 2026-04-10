"""
重编译后：测试模型 forward 在不同序列长度下是否正常
用法: CUDA_LAUNCH_BLOCKING=1 LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=1 python diagnose_seqlen.py
"""
import torch
import gc

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"
device = "cuda:0"

# ============================================================
# Step 1: bf16 模型 + flash_attn，逐步增加序列长度
# ============================================================
print("=" * 60)
print("Step 1: bf16 + flash_attention_2, 逐步增加序列长度")
print("=" * 60)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
)

if hasattr(model.config, "sliding_window"):
    model.config.sliding_window = None

print(f"模型加载成功")
print(f"  head_dim={model.config.head_dim}")
print(f"  num_attention_heads={model.config.num_attention_heads}")
print(f"  num_key_value_heads={model.config.num_key_value_heads}")

for seq_len in [64, 128, 256, 512, 1024, 2048, 4096]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids)
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  seq_len={seq_len:>5}: ✅ 通过 (peak {mem:.1f} GB)")
    except Exception as e:
        print(f"  seq_len={seq_len:>5}: ❌ 崩溃 - {e}")
        # CUDA 出错后上下文已污染，后续测试不可靠
        print(f"  → 在 seq_len={seq_len} 处崩溃，停止后续测试")
        break

# 清理
del model
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# Step 2: 如果 Step 1 全部通过，测 4-bit + flash_attn + LoRA
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 4-bit + flash_attention_2 + LoRA (训练实际配置)")
print("=" * 60)

try:
    # 重新检查 CUDA 是否可用（如果 Step 1 崩了可能已经不行了）
    torch.cuda.synchronize()
except:
    print("  CUDA 上下文已损坏（Step 1 崩溃导致），请单独测试 Step 2")
    import sys; sys.exit(1)

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
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

from peft import LoraConfig, get_peft_model
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model.enable_input_require_grads()
peft_model = get_peft_model(model, peft_config)

# 推理测试
print("\n  --- 推理 (no_grad) ---")
for seq_len in [256, 1024, 2048, 4096]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
    try:
        with torch.no_grad():
            out = peft_model(input_ids=input_ids)
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  seq_len={seq_len:>5}: ✅ 通过 (peak {mem:.1f} GB)")
    except Exception as e:
        print(f"  seq_len={seq_len:>5}: ❌ 崩溃 - {e}")
        break

# 训练测试（带 backward）
print("\n  --- 训练 (forward + backward) ---")
peft_model.train()
for seq_len in [256, 1024, 2048, 4096]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    peft_model.zero_grad()
    
    input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
    labels = input_ids.clone()
    try:
        out = peft_model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  seq_len={seq_len:>5}: ✅ 通过 (loss={loss.item():.4f}, peak {mem:.1f} GB)")
    except Exception as e:
        print(f"  seq_len={seq_len:>5}: ❌ 崩溃 - {e}")
        break
    peft_model.zero_grad()

# gradient checkpointing 测试
print("\n  --- gradient_checkpointing + 训练 ---")
peft_model.gradient_checkpointing_enable()
for seq_len in [256, 1024, 2048, 4096]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    peft_model.zero_grad()
    
    input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
    labels = input_ids.clone()
    try:
        out = peft_model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  seq_len={seq_len:>5}: ✅ 通过 (loss={loss.item():.4f}, peak {mem:.1f} GB)")
    except Exception as e:
        print(f"  seq_len={seq_len:>5}: ❌ 崩溃 - {e}")
        break
    peft_model.zero_grad()

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)