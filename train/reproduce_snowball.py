import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

class ExplosionHunter:
    def __init__(self, threshold=5000.0):
        self.threshold = threshold
        self.exploded = False

    def hook_fn(self, module_name):
        def bwd_hook(module, grad_input, grad_output):
            if self.exploded:
                return
            
            for i, g in enumerate(grad_input):
                if g is not None:
                    max_g = g.abs().max().item()
                    # 只要梯度超过阈值，或者出现 NaN/Inf，立即拦截并报告
                    if max_g > self.threshold or torch.isnan(g).any() or torch.isinf(g).any():
                        print(f"\n💥 [爆点定位] 梯度在经过 【{module_name}】 反向计算时雪球滚大了！")
                        print(f"   当前层最大梯度范数: {max_g:.2f}")
                        self.exploded = True
                        return
        return bwd_hook

    def attach(self, model):
        # 按照从后往前的反向传播顺序，方便观察
        for name, module in reversed(list(model.named_modules())):
            if len(list(module.children())) == 0: 
                module.register_full_backward_hook(self.hook_fn(name))

def run_snowball_test(attn_impl):
    print(f"\n{'='*60}")
    print(f"🚀 极限施压测试: {attn_impl}")
    print(f"{'='*60}")
    
    # 构造一个有足够深度（24层）的模型，让梯度有空间去“滚雪球”
    config = Qwen2Config(
        hidden_size=1024,
        intermediate_size=2816,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=24,  # 重点：必须足够深
        vocab_size=151936,
        pad_token_id=0,
    )
    config._attn_implementation = attn_impl
    
    try:
        model = Qwen2ForCausalLM(config).to(torch.bfloat16).cuda()
        
        hunter = ExplosionHunter(threshold=5000.0)
        hunter.attach(model)
        
        # 重点：超长序列 (8192) + 极端的左侧填充
        seq_len = 8192
        batch_size = 2
        input_ids = torch.randint(10, 1000, (batch_size, seq_len)).cuda()
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bfloat16).cuda()
        
        # 让第一条数据前 8000 个 token 都是 Padding
        pad_len = 8000
        input_ids[0, :pad_len] = 0
        attention_mask[0, :pad_len] = 0.0
        
        print(f"➡️  执行前向传播 (序列长度: {seq_len})...")
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 模拟 TRL 的 Mask Loss，触发底层梯度的连锁反应
        log_probs = F.log_softmax(out.logits, dim=-1)
        token_loss = log_probs.max(dim=-1).values
        fake_loss = -(token_loss * attention_mask).sum()
        
        print(f"⬅️  执行反向传播 (初始 Loss: {fake_loss.item():.2f})...")
        fake_loss.backward()
        
        if not hunter.exploded:
            # 拿到最终流入 Embedding 层的最大梯度
            max_grad = max([p.grad.abs().max().item() for p in model.parameters() if p.grad is not None])
            print(f"✅ 测试通过！未发生爆炸。到达底层的最终最大梯度: {max_grad:.2f}")

    except Exception as e:
        print(f"❌ 运行崩溃: {str(e)}")
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # 1. 测试 SDPA，你会看到梯度在深层网络中一步步被放大直到报警
    run_snowball_test("sdpa")
    
    # 2. 测试 FA2，观察它如何凭借 Unpadding 和 FP32 累加器稳如泰山
    run_snowball_test("flash_attention_2")