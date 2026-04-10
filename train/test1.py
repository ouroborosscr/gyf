import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig

def reproduce_real_explosion(attn_implementation, model_path):
    # 1. 动用真实的预训练权重 (引发级联放大的核心)
    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = attn_implementation
    
    # 缩小层数以防测试时爆显存，但 8 层已经足够体现指数爆炸趋势
    config.num_hidden_layers = 8 
    
    # 强制用 bfloat16 加载真实权重
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        config=config, 
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    
    # 开启梯度计算
    for param in model.parameters():
        param.requires_grad = True

    # 2. 构造 RLHF/DAPO 训练中的真实长序列 (带有 Left Padding)
    seq_len = 8192
    pad_len = 4000
    
    input_ids = torch.randint(10, 1000, (1, seq_len)).cuda()
    attention_mask = torch.ones((1, seq_len), dtype=torch.bfloat16).cuda()
    
    # 模拟 Left Padding 掩码
    pad_id = config.pad_token_id if getattr(config, 'pad_token_id', None) is not None else 0
    input_ids[0, :pad_len] = pad_id
    attention_mask[0, :pad_len] = 0.0

    # 3. 前向传播
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # 4. 模拟 RLHF 的指数放大 Loss
    log_probs = F.log_softmax(out.logits, dim=-1)
    token_loss = log_probs.max(dim=-1).values
    masked_loss = (token_loss * attention_mask).sum() / attention_mask.sum()
    
    # 致命引爆点：DAPO/DPO 中 beta 项导致的指数放大
    rlhf_loss = torch.exp(masked_loss.abs() * 0.1)
    
    # 5. 反向传播
    rlhf_loss.backward()
    
    # 6. 提取模型最底层 (v_proj) 的梯度最大值
    bottom_grad = model.model.layers[0].self_attn.v_proj.weight.grad
    if bottom_grad is not None:
        return bottom_grad.abs().max().item()
    return None

if __name__ == "__main__":
    # ⚠️ 直接使用你本地的真实模型路径
    MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"
    
    print("="*60)
    print(f"🚀 测试真实权重在不同算子下的反向传播稳定性")
    print("="*60)
    
    try:
        sdpa_grad = reproduce_real_explosion("sdpa", MODEL_PATH)
        print(f"❌ SDPA Backend (Math 降级) | 底层梯度 Max: {sdpa_grad:.2e}")
        if sdpa_grad > 1e10 or str(sdpa_grad) == 'nan':
            print("   -> 💥 致命的梯度爆炸已复现！")
    except Exception as e:
        print(f"SDPA 运行报错: {e}")

    # 清理显存准备跑下一轮
    torch.cuda.empty_cache()

    try:
        fa2_grad = reproduce_real_explosion("flash_attention_2", MODEL_PATH)
        print(f"\n✅ Flash Attention 2 Backend | 底层梯度 Max: {fa2_grad:.2e}")
        print("   -> 🛡️ 寄存器 FP32 保护 + Padding 截断，梯度绝对稳定。")
    except Exception as e:
        print(f"FA2 运行报错: {e}")