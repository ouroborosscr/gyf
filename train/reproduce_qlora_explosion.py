import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

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
                    if max_g > self.threshold or torch.isnan(g).any() or torch.isinf(g).any():
                        print(f"\n💥 [爆点定位] 梯度在传入 【{module_name}】 之前就已经异常！")
                        print(f"   传入梯度最大值: {max_g:.2f}")
                        self.exploded = True
                        return
        return bwd_hook

    def attach(self, model):
        # 按照反向传播的顺序挂载，拦截第一案发现场
        for name, module in reversed(list(model.named_modules())):
            if len(list(module.children())) == 0:
                module.register_full_backward_hook(self.hook_fn(name))

def run_qlora_bug(attn_impl):
    print(f"\n{'='*60}")
    print(f"🚀 终极环境复现 (GradCheck + LoRA模拟 + {attn_impl})")
    print(f"{'='*60}")

    config = Qwen2Config(
        hidden_size=1024, intermediate_size=2816,
        num_attention_heads=8, num_key_value_heads=4,
        num_hidden_layers=16, vocab_size=151936, pad_token_id=0,
    )
    config._attn_implementation = attn_impl
    model = Qwen2ForCausalLM(config).to(torch.bfloat16).cuda()

    # 🚨 补齐致命条件 1：开启梯度检查点 (触发重算过程的数值不稳定)
    model.gradient_checkpointing_enable()

    # 🚨 补齐致命条件 2：冻结底座，只开放部分权重 (模拟 LoRA 的不对称反传流)
    for name, param in model.named_parameters():
        if "v_proj" in name or "q_proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    hunter = ExplosionHunter(threshold=5000.0)
    hunter.attach(model)

    # 构造带大量 Left Padding 的数据
    seq_len = 4096
    input_ids = torch.randint(10, 1000, (2, seq_len)).cuda()
    attention_mask = torch.ones((2, seq_len), dtype=torch.bfloat16).cuda()
    input_ids[0, :2000] = 0
    attention_mask[0, :2000] = 0.0

    print(f"➡️  前向传播...")
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    
    log_probs = F.log_softmax(out.logits, dim=-1)
    fake_loss = -(log_probs.max(dim=-1).values * attention_mask).sum()

    print(f"⬅️  反向传播 (触发重算)...")
    fake_loss.backward()
    
    if not hunter.exploded:
        max_grad = max([p.grad.abs().max().item() for p in model.parameters() if p.grad is not None])
        print(f"✅ 测试通过！最大梯度范数: {max_grad:.2f}")

if __name__ == "__main__":
    run_qlora_bug("sdpa")
    run_qlora_bug("flash_attention_2")