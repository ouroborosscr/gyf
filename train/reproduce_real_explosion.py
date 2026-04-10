import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import gc

class ExplosionHunter:
    def __init__(self, threshold=5000.0):
        self.threshold = threshold
        self.exploded = False

    def hook_fn(self, module_name):
        def bwd_hook(module, grad_input, grad_output):
            if self.exploded: return
            for g in grad_output:
                if g is not None:
                    max_g = g.abs().max().item()
                    if max_g > self.threshold or torch.isnan(g).any():
                        print(f"\n💥 [爆点定位] 梯度在传入 【{module_name}】 之前就已经异常！")
                        print(f"   传入梯度最大值: {max_g:.2f}")
                        self.exploded = True
                        return
        return bwd_hook

    def attach(self, model):
        for name, module in reversed(list(model.named_modules())):
            if len(list(module.children())) == 0:
                module.register_full_backward_hook(self.hook_fn(name))

def run_real_test(backend_name):
    print(f"\n{'='*60}")
    print(f"🚀 真实权重 4-bit 极限施压 (防OOM版): {backend_name}")
    print(f"{'='*60}")
    
    MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B" 
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model_kwargs = {
        "device_map": "cuda:0",
        "quantization_config": bnb_config,
    }
    
    if backend_name == "flash_attention_2":
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    hunter = ExplosionHunter(threshold=5000.0)
    hunter.attach(model)

    # 🚀 修复点：严格对齐你的 per_device_train_batch_size=1
    seq_len = 4096
    batch_size = 1
    input_ids = torch.randint(10, 1000, (batch_size, seq_len)).cuda()
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bfloat16).cuda()
    
    input_ids[0, :2000] = 151643 
    attention_mask[0, :2000] = 0.0

    print(f"➡️  执行前向传播...")
    enable_math = (backend_name == "sdpa_math")
    
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=enable_math):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 🚀 修复点：计算完毕立即释放 out 对象的引用，并只保留需要的 loss 标量计算图
        logits = out.logits
        del out
        
        # 逐层计算以节省显存峰值
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        del logits
        
        token_loss = log_probs.max(dim=-1).values
        del log_probs
        
        fake_loss = -(token_loss * attention_mask).sum()

    print(f"⬅️  执行反向传播...")
    fake_loss.backward()
    
    if not hunter.exploded:
        max_grad = max([p.grad.abs().max().item() for p in model.parameters() if p.grad is not None])
        print(f"✅ 测试通过！未发生爆炸。到达底层的最终最大梯度: {max_grad:.2f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_real_test("sdpa_math")
    run_real_test("flash_attention_2")