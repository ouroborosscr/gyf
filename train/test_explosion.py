import os
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-0.8B"
    attn_mode = "eager" 
    
    print(f"🚀 开始真实的 QLoRA + DeepSpeed 梯度爆炸压测...")
    print(f"🔍 当前 Attention 模式: {attn_mode}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print("⏳ 正在加载 0.8B 模型 (4-bit NF4)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_mode,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))} 
    )
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    print("✅ 模型 4-bit 加载与 LoRA 注入完成！")
    
    # ==========================================
    # 🪝 核心修改：植入梯度间谍钩子 (Hook)
    # ==========================================
    # 用于在 DeepSpeed 清理梯度前，瞬间捕获极值
    grad_monitor = {"max": 0.0, "has_nan": False, "nan_layer": ""}

    def get_grad_hook(name):
        def hook(grad):
            if grad is not None:
                current_max = grad.abs().max().item()
                # 记录全局最大值
                if current_max > grad_monitor["max"]:
                    grad_monitor["max"] = current_max
                
                # 瞬间侦测 NaN 或 Inf
                if not torch.isfinite(grad).all():
                    grad_monitor["has_nan"] = True
                    if not grad_monitor["nan_layer"]:
                        grad_monitor["nan_layer"] = name
        return hook

    # 给所有需要求导的 LoRA 权重挂上钩子
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(get_grad_hook(name))
    
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 1e-4}
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            # 我们怀疑的爆炸元凶
            "overlap_comm": True,
            "contiguous_gradients": True
        }
    }
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    # 将序列缩小到 512 防止 Triton 算子再次 OOM
    seq_len = 512
    vocab_size = model.config.vocab_size if hasattr(model.config, "vocab_size") else 151936
    
    torch.autograd.set_detect_anomaly(True)
    
    try:
        for step in range(10):
            print("\n" + "="*40)
            print(f"🔄 Step {step} 开始前向传播...")
            dummy_input = torch.randint(0, vocab_size, (1, seq_len), device=model_engine.device)
            dummy_labels = dummy_input.clone()
            
            outputs = model_engine(input_ids=dummy_input, labels=dummy_labels)
            loss = outputs.loss
            
            print(f"📉 Step {step} Loss: {loss.item()}")
            if torch.isnan(loss) or torch.isinf(loss):
                print("💥 前向传播直接阵亡：发现 NaN/Inf！")
                break
                
            print(f"⏪ Step {step} 开始反向传播...")
            
            # 在反向传播前，清空上一轮的监控记录
            grad_monitor["max"] = 0.0
            grad_monitor["has_nan"] = False
            grad_monitor["nan_layer"] = ""
            
            # 这一步会触发所有挂在权重上的 Hook
            model_engine.backward(loss)
            
            print(f"📊 当前 Step 真实算出的最大梯度值: {grad_monitor['max']:.2e}")
            
            if grad_monitor["has_nan"]:
                print(f"🚨 致命爆炸点定位：在层 [{grad_monitor['nan_layer']}] 抓包到 NaN/Inf!")
                break
                
            model_engine.step()
            print("✅ Step 成功更新。")
            
    except Exception as e:
        print(f"\n❌ 捕获到异常崩溃：\n{e}")

if __name__ == "__main__":
    main()