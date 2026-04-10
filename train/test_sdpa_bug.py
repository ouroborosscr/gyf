import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

def build_dummy_data(vocab_size, pad_token_id, seq_len=32, pad_len=16):
    """
    构造 Left Padding 的极端测试数据
    第一条数据：左侧填充了一半的 PAD，右侧是正常 Token
    第二条数据：完全没有填充的正常 Token
    """
    input_ids = torch.randint(10, vocab_size, (2, seq_len)).cuda()
    attention_mask = torch.ones((2, seq_len), dtype=torch.long).cuda()
    
    # 对第一条序列执行左侧填充 (Left Padding)
    input_ids[0, :pad_len] = pad_token_id
    attention_mask[0, :pad_len] = 0
    
    # 构造 Labels，忽略 Padding 部分的 Loss
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    
    return input_ids, attention_mask, labels

def run_test(attn_impl):
    print(f"\n{'='*50}")
    print(f"🚀 测试注意力算子: {attn_impl}")
    print(f"{'='*50}")
    
    # 1. 初始化迷你版的 Qwen 模型结构 (Qwen3.5 也是基于此架构)
    config = Qwen2Config(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2, # 只要2层就能测出梯度爆炸
        vocab_size=1000,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = attn_impl
    
    try:
        # 强制使用 bfloat16 并放入 GPU，完美模拟你的训练环境
        model = Qwen2ForCausalLM(config).to(torch.bfloat16).cuda()
        
        input_ids, attention_mask, labels = build_dummy_data(
            vocab_size=config.vocab_size, 
            pad_token_id=config.pad_token_id
        )
        
        # 2. 前向传播
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        print(f"➡️  前向传播正常，Loss: {loss.item():.4f}")
        
        # 3. 反向传播
        loss.backward()
        
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
            print(f"🚨 警报: 捕获到极端异常的巨大梯度 -> {max_grad} (底层算子溢出前兆)")
        else:
            print(f"✅ 梯度健康！最大梯度范数 -> {max_grad:.4f}")

    except Exception as e:
        print(f"❌ 运行崩溃: {str(e)}")
        
    finally:
        # 清理显存，防止影响下一轮测试
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # 请确保在带有 GPU 的环境中运行此脚本
    print("🔬 开始进行 Left Padding 梯度爆炸复现实验...")
    
    # 跑这三组对照实验
    run_test("eager")             # 纯 Python 实现，带有 HF 官方的安全保护逻辑
    run_test("sdpa")              # PyTorch 原生 C++ 加速实现，会触发数学死锁
    run_test("flash_attention_2") # 外部高阶算子，物理剔除 Padding 避免死锁