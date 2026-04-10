"""
测试 DeepSpeed CPU offload 是否导致 10^27 梯度
用 GRPO 实际的 loss 公式（不是普通 CE loss），对比有无 DeepSpeed
用法: CUDA_LAUNCH_BLOCKING=1 LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=2 python test_deepspeed_grad.py
"""
import subprocess, sys, os, json

# 不用 DeepSpeed 的 worker（用短序列避免 OOM）
WORKER = r'''
import torch, json, sys, gc

cfg = json.loads(sys.argv[1])
seq_len = cfg["seq_len"]
loss_type = cfg["loss_type"]

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"
device = "cuda:0"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=5e-7
)

# ============================================================
# 模拟 GRPO 的 loss 计算方式
# ============================================================
def compute_grpo_style_loss(model, input_ids, loss_type):
    """
    模拟 GRPOTrainer._compute_loss 的核心路径
    在 num_iterations=1 时：ratio=1, loss = -advantage * per_token_logps
    """
    labels = input_ids.clone()
    
    # 1. forward 得到 logits
    out = model(input_ids=input_ids)
    logits = out.logits[:, :-1, :]  # (B, L-1, V)
    target = labels[:, 1:]          # (B, L-1)
    
    # 2. 计算 per-token log probs (和 TRL 的 selective_log_softmax 一样)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    per_token_logps = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
    
    # 3. 模拟 advantage (归一化后的 reward, 典型值 ±2)
    advantage = torch.tensor([1.5], device=device, dtype=torch.bfloat16)
    
    # 4. ratio = 1.0 (第一步, old_logps == new_logps)
    # per_token_loss = -ratio * advantage * ...
    # 对于 grpo/dapo, per_token_loss = -min(ratio*adv, clamp(ratio)*adv)
    # 当 ratio=1 时, per_token_loss = -advantage (标量广播到每个 token)
    
    mask = torch.ones_like(per_token_logps)
    
    # 关键：用 per_token_logps 做 loss（这是 GRPO 的实际 backward 路径）
    # GRPO loss 是 -advantage * ratio * per_token_logps 的聚合
    # ratio=1 时就是 -advantage * per_token_logps
    per_token_loss = -advantage * per_token_logps
    
    if loss_type == "grpo":
        # GRPO: 每个序列内先 mean, 再 batch mean
        loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1)).mean()
    elif loss_type == "dapo":
        # DAPO: 所有 token 求总和, 只除以 batch_size
        batch_size = input_ids.size(0)
        loss = (per_token_loss * mask).sum() / batch_size
    elif loss_type == "ce":
        # 对照: 普通 cross entropy
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
        )
    
    return loss

# ============================================================
# 跑测试
# ============================================================
input_ids = torch.randint(0, 1000, (1, seq_len), device=device)

optimizer.zero_grad()
loss = compute_grpo_style_loss(model, input_ids, loss_type)
loss_val = loss.item()
loss.backward()
torch.cuda.synchronize()

# 收集梯度统计
grad_max = 0.0
grad_nan = 0
grad_total = 0
grad_by_layer = {}

for name, p in model.named_parameters():
    if p.grad is not None:
        grad_total += 1
        gmax = p.grad.abs().max().item()
        if torch.isnan(p.grad).any():
            grad_nan += 1
            gmax = float('inf')
        if gmax > grad_max:
            grad_max = gmax
        
        # 按 layer 记录最大梯度
        for layer_num in range(10):
            if f"layers.{layer_num}." in name:
                key = f"layer_{layer_num}"
                if key not in grad_by_layer or gmax > grad_by_layer[key]:
                    grad_by_layer[key] = gmax
                break

result = {
    "loss_type": loss_type,
    "seq_len": seq_len,
    "loss": round(loss_val, 4),
    "grad_max": f"{grad_max:.2e}" if grad_max < float('inf') else "NaN",
    "grad_nan": grad_nan,
    "grad_total": grad_total,
    "layers": {k: f"{v:.2e}" for k, v in sorted(grad_by_layer.items())},
}
print("RESULT:" + json.dumps(result))
'''

worker_path = "/tmp/_test_ds_grad.py"
with open(worker_path, "w") as f:
    f.write(WORKER)

TESTS = [
    # label, loss_type, seq_len
    ("A: CE loss (对照, seq=256)",         "ce",    256),
    ("B: GRPO loss, seq=256",              "grpo",  256),
    ("C: DAPO loss, seq=256",              "dapo",  256),
    ("D: GRPO loss, seq=512",              "grpo",  512),
    ("E: DAPO loss, seq=512",              "dapo",  512),
    ("F: GRPO loss, seq=1024",             "grpo",  1024),
    ("G: DAPO loss, seq=1024",             "dapo",  1024),
    ("H: CE loss, seq=1024",               "ce",    1024),
]

print("=" * 70)
print("无 DeepSpeed，模拟 GRPO/DAPO loss 的 backward 梯度")
print("如果不用 DeepSpeed 梯度也是 10^27，说明问题在 loss 计算路径")
print("如果不用 DeepSpeed 梯度正常，说明问题在 DeepSpeed CPU offload")
print("=" * 70)

for label, lt, sl in TESTS:
    cfg = json.dumps({"loss_type": lt, "seq_len": sl})
    
    try:
        result = subprocess.run(
            [sys.executable, worker_path, cfg],
            capture_output=True, text=True, timeout=300,
            env=os.environ.copy(),
        )
        
        output = result.stdout
        found = False
        for line in output.split("\n"):
            if line.startswith("RESULT:"):
                d = json.loads(line[7:])
                layers_str = ", ".join(f"{k}={v}" for k, v in list(d["layers"].items())[:4])
                print(f"\n  {label}")
                print(f"    loss={d['loss']}, grad_max={d['grad_max']}, grad_nan={d['grad_nan']}")
                print(f"    {layers_str}")
                found = True
                break
        
        if not found:
            if result.returncode != 0:
                for line in result.stderr.split("\n"):
                    if "Error" in line and "Warning" not in line:
                        print(f"\n  {label}")
                        print(f"    ❌ {line.strip()[:90]}")
                        break
    except subprocess.TimeoutExpired:
        print(f"\n  {label}")
        print(f"    ⏰ 超时")

print(f"\n{'='*70}")
print("测试完成")
print("=" * 70)