"""
Reproduce: Qwen3.5 position_ids bug → gradient explosion + CUDA crash
Uses GRPOTrainer + DeepSpeed with synthetic data (no private datasets needed)

Usage:
    # Test 1: WITHOUT fix → expect gradient explosion / crash
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 \
    torchrun --nproc_per_node=1 reproduce_grpo_bug.py --use_lora --use_4bit [--use_flash_attn] [--apply_fix]

    # Test 2: WITH fix → expect normal training
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 \
    torchrun --nproc_per_node=1 reproduce_grpo_bug.py --use_lora --use_4bit [--use_flash_attn] --apply_fix
"""
import os, sys, argparse, json, random, logging, gc, datetime
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from peft import LoraConfig

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s")

# ============================================================
# Config
# ============================================================
MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"  # Change to your local path or "Qwen/Qwen3.5-9B"
DS_CONFIG = "ds_config_repro.json"
NUM_SAMPLES = 20  # small dataset for quick repro

# ============================================================
# Monkey-patch: toggle the position_ids bug fix
# ============================================================
import transformers.modeling_flash_attention_utils as flash_utils

_original_fn = flash_utils._is_packed_sequence

def _fixed_is_packed_sequence(position_ids, batch_size):
    """Patched: reject >2D position_ids (PR #44911)"""
    if position_ids is None:
        return False
    if position_ids.dim() > 2:
        return False
    return _original_fn(position_ids, batch_size)

# ============================================================
# Synthetic dataset: fake network traffic prompts
# ============================================================
def generate_synthetic_dataset(tokenizer, num_samples=20):
    """Generate fake network traffic analysis prompts — long enough to match real training."""
    prompts = []
    ground_truths = []

    for i in range(num_samples):
        # Generate fake flow data (~25 flows per sample)
        flows = []
        for j in range(25):
            flow = {
                "ts": 1700000000 + random.randint(0, 86400),
                "uid": f"C{random.randint(100000,999999)}.{random.randint(1,999)}",
                "id.orig_h": f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
                "id.orig_p": random.randint(1024, 65535),
                "id.resp_h": f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
                "id.resp_p": random.choice([80, 443, 22, 53, 8080, 3389]),
                "proto": random.choice(["tcp", "udp"]),
                "duration": round(random.uniform(0.001, 120.0), 6),
                "orig_bytes": random.randint(0, 50000),
                "resp_bytes": random.randint(0, 500000),
                "conn_state": random.choice(["SF", "S0", "REJ", "RSTO", "SH"]),
                "history": random.choice(["ShADad", "Sh", "S", "ShADadfF", "D"]),
                "batch_index": j + 1,
                "stream_payload_decoded": "".join(
                    random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ./=-_:;", k=random.randint(50, 500))
                ),
            }
            flows.append(flow)

        flows_json = json.dumps(flows, ensure_ascii=False, indent=2)

        user_content = f"""我在进行流量检测,现在我将给你数条json格式的流量数据,这些流量数据中可能存在多个攻击,也可能不存在攻击。
攻击的类型为"内网穿透""渗透隐蔽信道""加密通信"。
我希望你能找到最靠前的攻击流量的编号和这个攻击包含连续的几条流量,并返回其攻击类型。

以下是本次需要分析的流量数据：
{flows_json}"""

        messages = [
            {"role": "system", "content": "You are a helpful network security assistant."},
            {"role": "user", "content": user_content},
        ]

        try:
            final_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            final_prompt = f"<|system|>You are a helpful assistant.<|user|>{user_content}<|assistant|>"

        prompts.append(final_prompt)
        ground_truths.append({"start": None, "end": None})

    logging.info(f"Generated {len(prompts)} synthetic samples")
    return Dataset.from_dict({"prompt": prompts, "ground_truth": ground_truths})

# ============================================================
# Simple reward functions
# ============================================================
def varied_reward_func(completions, **kwargs):
    """Assign deterministic but varied rewards to ensure non-zero advantage std."""
    rewards = []
    for i, comp in enumerate(completions):
        # Force different scores: use hash of content to get varied but deterministic rewards
        score = (hash(comp[:50]) % 100) / 100.0  # 0.0 ~ 0.99
        # Also add index-based variation to guarantee spread within a batch
        score = score * 0.5 + (i % 4) * 0.25  # ensures 4 generations get different offsets
        rewards.append(round(score, 2))
    return rewards

# ============================================================
# Gradient monitor callback
# ============================================================
class GradientMonitorCallback(transformers.TrainerCallback):
    """Uses register_hook to capture gradients during backward (works with DeepSpeed)."""

    def __init__(self):
        self.grad_max = 0.0
        self.grad_nan = 0
        self.worst_param = ""
        self._hooks = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        for name, p in model.named_parameters():
            if p.requires_grad:
                hook = p.register_hook(self._make_hook(name))
                self._hooks.append(hook)
        print(f"[GradMonitor] Installed hooks on {len(self._hooks)} parameters", flush=True)

    def _make_hook(self, name):
        def hook_fn(grad):
            if grad is None:
                return
            if torch.isnan(grad).any():
                self.grad_nan += 1
            else:
                gm = grad.abs().max().item()
                if gm > self.grad_max:
                    self.grad_max = gm
                    self.worst_param = name
        return hook_fn

    def on_step_end(self, args, state, control, **kwargs):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        gm = self.grad_max
        if gm > 1e10:
            print(f"\n[{ts}] ❌ Step {state.global_step} 梯度爆炸! max={gm:.2e} nan={self.grad_nan} @ {self.worst_param}", flush=True)
        elif gm > 0:
            print(f"\n[{ts}] ✅ Step {state.global_step} 梯度正常 max={gm:.2e} nan={self.grad_nan}", flush=True)
        else:
            print(f"\n[{ts}] ⚠️  Step {state.global_step} 梯度为零", flush=True)
        # Reset for next step
        self.grad_max = 0.0
        self.grad_nan = 0
        self.worst_param = ""

# ============================================================
# Write DeepSpeed config
# ============================================================
def write_ds_config():
    config = {
        "gradient_clipping": 0.1,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": 4,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    }
    with open(DS_CONFIG, "w") as f:
        json.dump(config, f, indent=2)
    logging.info(f"DeepSpeed config written to {DS_CONFIG}")

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--apply_fix", action="store_true", help="Apply the position_ids dim>2 fix")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # Apply or remove fix
    if args.apply_fix:
        flash_utils._is_packed_sequence = _fixed_is_packed_sequence
        print("🔧 Fix APPLIED: _is_packed_sequence rejects >2D position_ids")
    else:
        # Ensure we use the original (potentially buggy) version
        print("⚠️  Fix NOT applied: using original _is_packed_sequence")

    # Environment info
    print("=" * 60)
    print(f"Python:       {sys.version.split()[0]}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"flash_attn:   {'yes' if args.use_flash_attn else 'no (SDPA)'}")
    print(f"fix applied:  {args.apply_fix}")
    print("=" * 60)

    write_ds_config()

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": local_rank},
        "low_cpu_mem_usage": True,
    }
    if args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model_config = AutoConfig.from_pretrained(MODEL_PATH)
    if hasattr(model_config, "sliding_window"):
        model_config.sliding_window = None

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=model_config, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # Dataset
    dataset = generate_synthetic_dataset(tokenizer, num_samples=NUM_SAMPLES)

    # Training config
    training_args = GRPOConfig(
        output_dir="./repro_output",
        learning_rate=5e-9,
        max_grad_norm=0.1,
        beta=0.04,
        lr_scheduler_type="cosine",
        logging_steps=1,
        loss_type="dapo",
        epsilon=0.2,
        epsilon_high=0.28,
        mask_truncated_completions=False,
        delta=10.0,
        max_steps=3,  # Only need 2-3 steps to see the bug
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=4096,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        optim="adamw_torch",
        save_strategy="no",
        deepspeed=DS_CONFIG,
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[varied_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[GradientMonitorCallback()],
    )

    logging.info("Starting training...")
    try:
        trainer.train()
        logging.info("✅ Training completed successfully")
    except Exception as e:
        logging.error(f"❌ Training crashed: {e}")
        raise

if __name__ == "__main__":
    main()