"""
测试 loss_type="grpo" vs "dapo" 对梯度的影响
在真实 GRPOTrainer + DeepSpeed 环境下跑 2 步，监控梯度
用法: CUDA_LAUNCH_BLOCKING=1 LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=2 python test_loss_type.py
"""
import subprocess, sys, os, json

WORKER = r'''
import torch, json, sys, os, gc

cfg = json.loads(sys.argv[1])
loss_type = cfg["loss_type"]
max_completion_length = cfg["max_completion_length"]

# 最小化环境
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_P2P_DISABLE"] = "1"

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from datetime import datetime

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"

# 梯度监控回调
class GradMonitorCallback(transformers.TrainerCallback):
    def __init__(self):
        self.step = 0
        self.hooks = []
        self.grad_max_per_step = {}
        self.grad_nan_per_step = {}
        self.installed = False
    
    def _install(self, model):
        if self.installed:
            return
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora" not in name:
                continue
            def make_hook(pname):
                def hook(grad):
                    s = self.step
                    gmax = grad.abs().max().item() if not (torch.isnan(grad).any() or torch.isinf(grad).any()) else float('inf')
                    gnan = torch.isnan(grad).any().item()
                    if s not in self.grad_max_per_step:
                        self.grad_max_per_step[s] = 0.0
                        self.grad_nan_per_step[s] = 0
                    if gmax > self.grad_max_per_step[s]:
                        self.grad_max_per_step[s] = gmax
                    if gnan:
                        self.grad_nan_per_step[s] += 1
                    return grad
                return hook
            p.register_hook(make_hook(name))
        self.installed = True
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model: self._install(model)
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.step += 1
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        s = self.step
        gmax = self.grad_max_per_step.get(s, 0)
        gnan = self.grad_nan_per_step.get(s, 0)
        
        # 检查参数
        param_nan = 0
        if model:
            for n, p in model.named_parameters():
                if p.requires_grad and torch.isnan(p.data).any():
                    param_nan += 1
        
        # 检查 logits
        logits_ok = True
        if model:
            try:
                device = next(model.parameters()).device
                with torch.no_grad():
                    out = model(input_ids=torch.randint(0,1000,(1,32),device=device))
                if torch.isnan(out.logits).any():
                    logits_ok = False
            except:
                logits_ok = False
        
        gmax_str = f"{gmax:.2e}" if gmax < 1e30 else "inf"
        status = "✅" if (gnan == 0 and param_nan == 0 and logits_ok) else "❌"
        print(f"STEP:{json.dumps({'step':s,'status':status,'grad_max':gmax_str,'grad_nan':gnan,'param_nan':param_nan,'logits_ok':logits_ok})}")

# 简单 reward
def simple_reward(completions, **kwargs):
    return [1.0 if len(c) > 100 else -1.0 for c in completions]

# 加载
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config,
    torch_dtype=torch.bfloat16, device_map={"": 0},
    low_cpu_mem_usage=True,
)
if hasattr(model.config, "sliding_window"):
    model.config.sliding_window = None
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
if hasattr(model, "generation_config"):
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)

# 构造简单数据集
prompts = []
for i in range(20):
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Write a short analysis of network traffic pattern {i}. Be detailed."}
    ]
    prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

dataset = Dataset.from_dict({"prompt": prompts})

monitor = GradMonitorCallback()

training_args = GRPOConfig(
    output_dir="/tmp/test_loss_type",
    learning_rate=5e-7,
    max_grad_norm=0.3,
    beta=0.04,
    loss_type=loss_type,
    lr_scheduler_type="cosine",
    logging_steps=1,
    max_steps=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=2,
    max_completion_length=max_completion_length,
    bf16=True,
    gradient_checkpointing=True,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    repetition_penalty=1,
    optim="adamw_torch",
    save_strategy="no",
    deepspeed="ds_config.json",
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[simple_reward],
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
    callbacks=[monitor],
)

print(f"CONFIG:{json.dumps({'loss_type':loss_type,'max_completion_length':max_completion_length})}")

try:
    trainer.train()
    print("DONE:OK")
except Exception as e:
    err = str(e).split('\n')[0][:120]
    print(f"DONE:FAIL:{err}")
'''

worker_path = "/tmp/_test_loss_type.py"
with open(worker_path, "w") as f:
    f.write(WORKER)

TESTS = [
    ("A: dapo + 512 (默认 loss_type)",  "dapo",  512),
    ("B: grpo + 512 (修复后)",           "grpo",  512),
]

for label, lt, mcl in TESTS:
    cfg = json.dumps({"loss_type": lt, "max_completion_length": mcl})
    
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    
    env = os.environ.copy()
    
    # 需要用 torchrun 因为有 DeepSpeed
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=1",
        worker_path, cfg
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800, env=env,
        )
        
        output = result.stdout + result.stderr
        
        for line in output.split("\n"):
            if line.startswith("CONFIG:"):
                d = json.loads(line[7:])
                print(f"  配置: loss_type={d['loss_type']}, max_completion_length={d['max_completion_length']}")
            elif line.startswith("STEP:"):
                d = json.loads(line[5:])
                print(f"  {d['status']} Step {d['step']}: grad_max={d['grad_max']} grad_nan={d['grad_nan']} "
                      f"param_nan={d['param_nan']} logits_ok={d['logits_ok']}")
            elif line.startswith("DONE:"):
                status = line[5:]
                if status == "OK":
                    print(f"  🎉 3 步训练完成")
                else:
                    print(f"  💀 {status}")
        
        if result.returncode != 0 and "DONE:" not in output:
            for line in output.split("\n"):
                if "CUDA error" in line or "probability tensor" in line or "OutOfMemory" in line:
                    print(f"  💀 {line.strip()[:100]}")
                    break
                    
    except subprocess.TimeoutExpired:
        print(f"  ⏰ 超时 (>30min)")

print(f"\n{'='*70}")
print("测试完成")
print(f"{'='*70}")