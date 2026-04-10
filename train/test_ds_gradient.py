"""
直接用 torchrun 跑，测试 DeepSpeed 对梯度的影响
用法: CUDA_LAUNCH_BLOCKING=1 CUDA_HOME=/usr/local/cuda-12.8 \
      LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 test_ds_gradient.py
"""
import torch
import os, json, logging

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_P2P_DISABLE"] = "1"

import deepspeed
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO, format='%(message)s')

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"
device = "cuda:0"

# ============================================================
# 初始化分布式
# ============================================================
dist.init_process_group(backend="nccl")
torch.cuda.set_device(0)

# ============================================================
# 加载模型
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config,
    torch_dtype=torch.bfloat16, device_map={"": 0},
    attn_implementation="sdpa", low_cpu_mem_usage=True,
)
if hasattr(model.config, "sliding_window"):
    model.config.sliding_window = None

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.train()

# ============================================================
# GRPO loss 函数
# ============================================================
def grpo_loss(model_out, input_ids, loss_type="grpo"):
    logits = model_out.logits[:, :-1, :]
    target = input_ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    per_token_logps = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    
    advantage = torch.tensor([1.5], device=device, dtype=torch.bfloat16)
    per_token_loss = -advantage * per_token_logps
    mask = torch.ones_like(per_token_logps)
    
    if loss_type == "grpo":
        loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1)).mean()
    elif loss_type == "dapo":
        loss = (per_token_loss * mask).sum() / input_ids.size(0)
    return loss

def get_grad_stats(model):
    grad_max = 0.0
    grad_nan = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                grad_nan += 1
            else:
                gm = p.grad.abs().max().item()
                if gm > grad_max:
                    grad_max = gm
    return grad_max, grad_nan

# ============================================================
# 测试函数
# ============================================================
def test_without_deepspeed(loss_type, seq_len):
    """不用 DeepSpeed, 纯 PyTorch optimizer"""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-7
    )
    optimizer.zero_grad()
    
    input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
    out = model(input_ids=input_ids)
    loss = grpo_loss(out, input_ids, loss_type)
    loss.backward()
    torch.cuda.synchronize()
    
    gmax, gnan = get_grad_stats(model)
    
    # 清理
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    
    return loss.item(), gmax, gnan

def test_with_deepspeed(loss_type, seq_len, ds_config):
    """用 DeepSpeed engine"""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-7)
    
    ds_engine, ds_optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=trainable_params,
        config=ds_config,
    )
    
    input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
    out = ds_engine(input_ids=input_ids)
    loss = grpo_loss(out, input_ids, loss_type)
    
    ds_engine.backward(loss)
    torch.cuda.synchronize()
    
    gmax, gnan = get_grad_stats(model)
    loss_val = loss.item()
    
    # 不做 step，只看梯度
    # 清理 DeepSpeed engine
    ds_engine.zero_grad()
    torch.cuda.empty_cache()
    
    return loss_val, gmax, gnan

# ============================================================
# 跑测试
# ============================================================
print("\n" + "=" * 70)
print("对比 DeepSpeed vs 无 DeepSpeed 的梯度")
print("=" * 70)

seq_len = 512

# 1. 无 DeepSpeed
for lt in ["grpo", "dapo"]:
    loss_val, gmax, gnan = test_without_deepspeed(lt, seq_len)
    gmax_str = f"{gmax:.2e}" if gmax < float('inf') else "inf"
    print(f"\n  无 DeepSpeed | {lt} | seq={seq_len}: loss={loss_val:.2f}, grad_max={gmax_str}, nan={gnan}")

# 2. 有 DeepSpeed, 无 CPU offload
ds_config_no_offload = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    },
    "gradient_clipping": 0.3,
    "gradient_accumulation_steps": 1,
    "train_batch_size": 1,
    "train_micro_batch_size_per_gpu": 1,
}

for lt in ["grpo", "dapo"]:
    try:
        loss_val, gmax, gnan = test_with_deepspeed(lt, seq_len, ds_config_no_offload)
        gmax_str = f"{gmax:.2e}" if gmax < float('inf') else "inf"
        print(f"\n  DeepSpeed ZeRO-2 (无 offload) | {lt} | seq={seq_len}: loss={loss_val:.2f}, grad_max={gmax_str}, nan={gnan}")
    except Exception as e:
        print(f"\n  DeepSpeed ZeRO-2 (无 offload) | {lt}: ❌ {str(e)[:80]}")

# 3. 有 DeepSpeed, 有 CPU offload
ds_config_offload = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    },
    "gradient_clipping": 0.3,
    "gradient_accumulation_steps": 1,
    "train_batch_size": 1,
    "train_micro_batch_size_per_gpu": 1,
}

for lt in ["grpo", "dapo"]:
    try:
        loss_val, gmax, gnan = test_with_deepspeed(lt, seq_len, ds_config_offload)
        gmax_str = f"{gmax:.2e}" if gmax < float('inf') else "inf"
        print(f"\n  DeepSpeed ZeRO-2 + CPU offload | {lt} | seq={seq_len}: loss={loss_val:.2f}, grad_max={gmax_str}, nan={gnan}")
    except Exception as e:
        print(f"\n  DeepSpeed ZeRO-2 + CPU offload | {lt}: ❌ {str(e)[:80]}")

# 4. 有 DeepSpeed, 有 CPU offload, gradient_accumulation_steps=4 (你的实际配置)
ds_config_full = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    },
    "gradient_clipping": 0.3,
    "gradient_accumulation_steps": 4,
    "train_batch_size": 4,
    "train_micro_batch_size_per_gpu": 1,
}

for lt in ["grpo", "dapo"]:
    try:
        loss_val, gmax, gnan = test_with_deepspeed(lt, seq_len, ds_config_full)
        gmax_str = f"{gmax:.2e}" if gmax < float('inf') else "inf"
        print(f"\n  DeepSpeed 完整配置 (offload+accum=4) | {lt} | seq={seq_len}: loss={loss_val:.2f}, grad_max={gmax_str}, nan={gnan}")
    except Exception as e:
        print(f"\n  DeepSpeed 完整配置 | {lt}: ❌ {str(e)[:80]}")

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)

dist.destroy_process_group()