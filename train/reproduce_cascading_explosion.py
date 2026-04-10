import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM
import math

class LayerCascadeHunter:
    def __init__(self):
        self.logs = []

    def hook_fn(self, name):
        def bwd_hook(module, grad_input, grad_output):
            g = grad_output[0]
            if g is not None:
                max_val = g.abs().max().item()
                # 只打印那些开始失控的层，模拟你的真实日志
                if max_val > 1000:
                    print(f"⚠️ [数值爆炸预警] 层: {name: <35} | 梯度 Max: {max_val:.2e}")
        return bwd_hook

    def attach(self, model):
        for name, module in model.named_modules():
            # 挂载在 V_proj 上，因为它是 Attention 反向传播最先冲刷的权重
            if "v_proj" in name or "q_proj" in name:
                module.register_full_backward_hook(self.hook_fn(name))

def reproduce_explosion():
    print("="*60)
    print("🚀 [终极复现] DAPO + SDPA Math 反向传播级联爆炸")
    print("="*60)

    device, dtype = "cuda", torch.bfloat16

    # 1. 构造一个足够深（32层）的模型，以提供“滚雪球”的跑道
    # 稍微调小 hidden_size 以防测试时 OOM，但不影响深度的乘数效应
    config = Qwen2Config(
        hidden_size=512, intermediate_size=2048,
        num_attention_heads=8, num_key_value_heads=2,
        num_hidden_layers=32, vocab_size=151936, pad_token_id=0,
    )
    # 强制使用最危险的 SDPA Math 引擎
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).to(dtype).cuda()

    # 模拟 LoRA：冻结底座，只开启投影层梯度
    for name, param in model.named_parameters():
        if "proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    hunter = LayerCascadeHunter()
    hunter.attach(model)

    # 2. 构造带 Left Padding 的长序列 (4096长度)
    seq_len = 4096
    input_ids = torch.randint(10, 1000, (1, seq_len), device=device)
    attention_mask = torch.ones((1, seq_len), dtype=dtype, device=device)
    
    # 模拟极端 Left Padding
    input_ids[0, :2000] = 0
    attention_mask[0, :2000] = 0.0

    print("➡️  执行前向传播 (SDPA Math)...")
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 3. 模拟 DAPO 的“汽油”：指数放大效应
        # 我们人为加入一个 exp() 函数，模拟 DAPO 中基于优势的梯度缩放
        log_probs = F.log_softmax(out.logits, dim=-1)
        token_loss = log_probs.max(dim=-1).values
        masked_loss = (token_loss * attention_mask).sum() / attention_mask.sum()
        
        # 【致命引爆点】：用 exp() 放大了顶层误差
        fake_dapo_loss = torch.exp(masked_loss.abs() * 0.5)

    print(f"⬅️  执行反向传播 (初始 Loss 规模: {fake_dapo_loss.item():.2e})...")
    print("-" * 60)
    
    # 释放显存
    del out, log_probs
    torch.cuda.empty_cache()

    fake_dapo_loss.backward()
    
    print("-" * 60)
    print("💡 结论: 如果你看到梯度随着层数降低（31 -> 0）呈指数级增长，这就完美复刻了你的真实崩溃！")

if __name__ == "__main__":
    reproduce_explosion()