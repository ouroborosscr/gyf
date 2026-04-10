import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

def build_dummy_data(vocab_size, pad_token_id, seq_len=32, pad_len=16):
    """
    构造 Left Padding 的极端测试数据
    第一条：左侧填充了一半的 PAD，右侧是正常 Token
    第二条：完全没有填充的正常 Token
    """
    input_ids = torch.randint(10, vocab_size, (2, seq_len)).cuda()
    # 注意：这里 mask 用 bfloat16，为了后续与 token_loss 相乘时不发生类型报错
    attention_mask = torch.ones((2, seq_len), dtype=torch.bfloat16).cuda()
    
    # 对第一条序列执行左侧填充 (Left Padding)
    input_ids[0, :pad_len] = pad_token_id
    attention_mask[0, :pad_len] = 0.0
    
    return input_ids, attention_mask

def run_test(attn_impl):
    print(f"\n{'='*50}")
    print(f"🚀 测试注意力算子: {attn_impl}")
    print(f"{'='*50}")
    
    # 初始化迷你版的 Qwen 模型结构
    config = Qwen2Config(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=1000,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = attn_impl
    
    try:
        model = Qwen2ForCausalLM(config).to(torch.bfloat16).cuda()
        
        input_ids, attention_mask = build_dummy_data(
            vocab_size=config.vocab_size, 
            pad_token_id=config.pad_token_id
        )
        
        # 1. 前向传播 (不传 labels，拿到纯净的 logits)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        
        # ====================================================
        # 2. 模拟 TRL (GRPO/DAPO) 内部的手动 Loss 计算与 Mask 机制
        # ====================================================
        # 计算全词表的 log_softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 取出每个 token 预测概率最大值的对数，作为 dummy loss 信号
        token_loss = log_probs.max(dim=-1).values
        
        # 【致命点】：手动使用 attention_mask 屏蔽 padding 位置
        # 在 sdpa 算子下，padding 位置的 token_loss 会是 NaN
        # 数学上：NaN * 0.0 = NaN，NaN 污染了整个 fake_loss
        fake_loss = -(token_loss * attention_mask).sum()
        
        print(f"➡️  前向传播完成，Loss 值: {fake_loss.item()}")
        
        # 3. 反向传播
        fake_loss.backward()
        
        # 4. 检查梯度
        has_nan = False
        has_inf = False
        max_grad = 0.0
        nan_layer = ""
        
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_abs_max = p.grad.abs().max().item()
                if torch.isnan(p.grad).any():
                    has_nan = True
                    nan_layer = name
                    break
                if torch.isinf(p.grad).any():
                    has_inf = True
                    nan_layer = name
                    break
                max_grad = max(max_grad, grad_abs_max)
                
        if has_nan:
            print(f"🚨 警报: 捕获到 NaN 梯度！首个崩坏层 -> {nan_layer}")
        elif has_inf:
            print(f"🚨 警报: 捕获到 Inf 梯度！首个崩坏层 -> {nan_layer}")
        elif max_grad > 1e4:
            print(f"🚨 警报: 捕获到极端异常的巨大梯度 -> {max_grad}")
        else:
            print(f"✅ 梯度健康！最大梯度范数 -> {max_grad:.4f}")

    except Exception as e:
        print(f"❌ 运行崩溃: {str(e)}")
        
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("🔬 开始进行 Left Padding 梯度爆炸复现实验 (TRL Loss 模拟版)...")
    run_test("eager")
    run_test("sdpa")
    run_test("flash_attention_2")