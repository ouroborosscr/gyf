import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM
import sys

# 1. 劫持 PyTorch 底层的 SDPA 函数
original_sdpa = F.scaled_dot_product_attention

def hijacked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    print("\n🚨 [劫持成功] Qwen3.5 正在调用 PyTorch SDPA!")
    print(f"   Query 形状: {query.shape}")
    if attn_mask is not None:
        print(f"   ⚠️ Qwen 传来的 Mask 形状: {attn_mask.shape}")
        print(f"   ⚠️ Mask 的数据类型: {attn_mask.dtype}")
        
        # 算一笔恐怖的显存账
        elements = attn_mask.numel()
        gb_size = elements * 2 / (1024**3) # bfloat16 占 2 字节
        print(f"   💀 仅这个 Mask 矩阵就会占据显存: {gb_size:.6f} GB")
    else:
        print("   ✅ Mask 为 None")
    
    print(f"   is_causal 参数: {is_causal}")
    
    # 终止程序，我们只看参数
    sys.exit(0)

F.scaled_dot_product_attention = hijacked_sdpa

def audit_qwen_attention():
    print("="*60)
    print("🔍 [源码级审计] Qwen3.5 (SDPA 模式) 到底传了什么给底层？")
    print("="*60)

    # 初始化一个 Qwen3.5
    config = Qwen2Config(
        hidden_size=1024, intermediate_size=2816,
        num_attention_heads=8, num_key_value_heads=4,
        num_hidden_layers=2, vocab_size=151936, pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).to(torch.bfloat16).cuda()

    # 模拟 DAPO 的 Left Padding 长文本
    # 注意：为了不立刻 OOM，我们先用 8192 长度。你可以想象如果是 10万 长度会怎样！
    L = 8192
    input_ids = torch.randint(10, 1000, (1, L)).cuda()
    attention_mask = torch.ones((1, L), dtype=torch.long).cuda()
    attention_mask[0, :2000] = 0 # Left Padding

    print(f"➡️ 模拟 DAPO 训练输入，序列长度: {L}")
    model(input_ids=input_ids, attention_mask=attention_mask)

if __name__ == "__main__":
    audit_qwen_attention()