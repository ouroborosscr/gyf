"""
Part 1: 定位 flash_attn 崩溃条件（哪些序列长度/batch 组合会崩）
Part 2: 逐层对比 flash_attn vs SDPA 的梯度差异，找到 10²⁷ 的来源层

用法: CUDA_LAUNCH_BLOCKING=1 CUDA_HOME=/usr/local/cuda-12.8 \
      LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=2 python diagnose_attn.py
"""
import torch
import gc, logging, json

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = "cuda:0"
MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)

peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)


def load_model(attn_impl):
    """加载模型，指定 attention 实现"""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation=attn_impl, low_cpu_mem_usage=True,
    )
    if hasattr(model.config, "sliding_window"):
        model.config.sliding_window = None
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    model.train()
    return model


def grpo_loss_and_grads(model, input_ids):
    """模拟 GRPO loss 的 forward + backward，返回每层 LoRA 的梯度最大值"""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-7
    )
    optimizer.zero_grad()
    
    out = model(input_ids=input_ids)
    logits = out.logits[:, :-1, :]
    target = input_ids[:, 1:]
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    per_token_logps = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    
    advantage = torch.tensor([1.5], device=device, dtype=torch.bfloat16)
    per_token_loss = -advantage * per_token_logps
    mask = torch.ones_like(per_token_logps)
    loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1)).mean()
    
    loss.backward()
    torch.cuda.synchronize()
    
    # 收集每层梯度
    layer_grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None and "lora" in name:
            gmax = p.grad.abs().max().item()
            gnan = torch.isnan(p.grad).any().item()
            
            # 提取 layer 编号和模块名
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i+1 < len(parts):
                    layer_num = int(parts[i+1])
                    # 模块类型
                    for mod in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
                        if mod in name:
                            lora_part = 'A' if 'lora_A' in name else 'B'
                            key = f"L{layer_num:02d}.{mod}.{lora_part}"
                            layer_grads[key] = {"max": gmax, "nan": gnan}
                            break
                    break
    
    optimizer.zero_grad()
    del out, logits, log_probs, per_token_logps, per_token_loss, loss
    gc.collect()
    torch.cuda.empty_cache()
    
    return layer_grads


def flash_attn_forward_test(model, seq_len):
    """测试 flash_attn forward 在不同序列长度下是否崩溃"""
    input_ids = torch.randint(100, 5000, (1, seq_len), device=device)
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids)
        result = "✅"
        del out
    except Exception as e:
        err = str(e).split('\n')[0][:60]
        result = f"❌ {err}"
    
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ============================================================
print("=" * 70)
print("Part 1: flash_attn 崩溃条件扫描")
print("  训练时 Step 1 成功 (generate + training forward+backward)")
print("  Step 2 崩在 ref_model forward (序列可能更长/内存碎片)")
print("=" * 70)

model_flash = load_model("flash_attention_2")

# 先测不同序列长度的纯 forward
print("\n[1a] flash_attn forward - 不同序列长度:")
for seq_len in [256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]:
    result = flash_attn_forward_test(model_flash, seq_len)
    print(f"  seq={seq_len:>5}: {result}")

# 测 forward+backward 后再 forward（模拟 Step1 train → Step2 ref forward）
print("\n[1b] flash_attn: forward+backward → 再次 forward (模拟 Step 1→2 转换):")
for seq_len in [512, 1024, 2048]:
    input_ids = torch.randint(100, 5000, (1, seq_len), device=device)
    try:
        # Step 1: forward + backward
        out = model_flash(input_ids=input_ids)
        logits = out.logits[:, :-1, :]
        target = input_ids[:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        per_token_logps = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        loss = (-1.5 * per_token_logps).mean()
        loss.backward()
        torch.cuda.synchronize()
        
        # 清理梯度但不清理模型状态
        for p in model_flash.parameters():
            if p.grad is not None:
                p.grad.zero_()
        
        del out, logits, log_probs, per_token_logps, loss
        gc.collect()
        torch.cuda.empty_cache()
        
        # Step 2: 再次 forward (模拟 ref model forward)
        input_ids2 = torch.randint(100, 5000, (1, seq_len), device=device)
        with torch.no_grad():
            out2 = model_flash(input_ids=input_ids2)
        print(f"  seq={seq_len}: ✅ train→ref forward 成功")
        del out2
    except Exception as e:
        err = str(e).split('\n')[0][:80]
        print(f"  seq={seq_len}: ❌ {err}")
    
    gc.collect()
    torch.cuda.empty_cache()

# 释放 flash model
del model_flash
gc.collect()
torch.cuda.empty_cache()

# ============================================================
print("\n" + "=" * 70)
print("Part 2: 逐层梯度对比 flash_attn vs SDPA")
print("  用相同输入，对比每层 LoRA 的梯度大小")
print("=" * 70)

seq_len = 512  # 用 flash_attn 能跑通的长度

# 固定输入
torch.manual_seed(42)
input_ids = torch.randint(100, 5000, (1, seq_len), device=device)

# flash_attn 的梯度
print(f"\n[2a] 加载 flash_attn 模型，计算梯度 (seq={seq_len})...")
model_flash = load_model("flash_attention_2")
flash_grads = grpo_loss_and_grads(model_flash, input_ids.clone())
del model_flash
gc.collect()
torch.cuda.empty_cache()

# SDPA 的梯度
print(f"[2b] 加载 SDPA 模型，计算梯度 (seq={seq_len})...")
model_sdpa = load_model("sdpa")
sdpa_grads = grpo_loss_and_grads(model_sdpa, input_ids.clone())
del model_sdpa
gc.collect()
torch.cuda.empty_cache()

# 对比
print(f"\n[2c] 逐层对比 (只显示前 8 层, attention + MLP):")
print(f"  {'Layer/Module':<25} {'flash_attn':>12} {'SDPA':>12} {'比值':>12}")
print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")

all_keys = sorted(set(list(flash_grads.keys()) + list(sdpa_grads.keys())))

# 只看前 8 层
attn_summary = {}
for key in all_keys:
    layer_num = int(key.split('.')[0][1:])
    if layer_num > 7:
        continue
    
    fg = flash_grads.get(key, {})
    sg = sdpa_grads.get(key, {})
    
    fmax = fg.get("max", 0)
    smax = sg.get("max", 0)
    
    ratio = smax / fmax if fmax > 1e-30 else float('inf')
    
    print(f"  {key:<25} {fmax:>12.4e} {smax:>12.4e} {ratio:>12.1f}x")
    
    # 按模块类型汇总
    mod = key.split('.')[1]
    if mod not in attn_summary:
        attn_summary[mod] = {"flash": [], "sdpa": []}
    attn_summary[mod]["flash"].append(fmax)
    attn_summary[mod]["sdpa"].append(smax)

print(f"\n[2d] 按模块类型汇总 (所有层的平均):")
print(f"  {'Module':<15} {'flash avg':>12} {'SDPA avg':>12} {'比值':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
for mod in sorted(attn_summary.keys()):
    favg = sum(attn_summary[mod]["flash"]) / len(attn_summary[mod]["flash"])
    savg = sum(attn_summary[mod]["sdpa"]) / len(attn_summary[mod]["sdpa"])
    ratio = savg / favg if favg > 1e-30 else float('inf')
    print(f"  {mod:<15} {favg:>12.4e} {savg:>12.4e} {ratio:>12.1f}x")

print(f"\n{'='*70}")
print("诊断完成")
print(f"{'='*70}")