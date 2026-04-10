"""
诊断词表大小和 token ID 越界问题
用法: LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=1 python diagnose_vocab.py
"""
import torch
import json, os

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"

# ============================================================
print("=" * 60)
print("Step 1: 模型 config 中的词表大小")
print("=" * 60)

from transformers import AutoConfig
config = AutoConfig.from_pretrained(MODEL_PATH)
# 打印所有可能相关的属性
for attr in ["vocab_size", "hidden_size", "num_attention_heads", "num_key_value_heads", 
             "head_dim", "num_hidden_layers", "intermediate_size"]:
    print(f"  config.{attr} = {getattr(config, attr, 'N/A')}")

# 检查 text_config（多模态模型可能把词表放在子 config 里）
if hasattr(config, "text_config"):
    tc = config.text_config
    print(f"\n  config.text_config 存在:")
    for attr in ["vocab_size", "hidden_size"]:
        print(f"    text_config.{attr} = {getattr(tc, attr, 'N/A')}")
        
# 直接打印所有带 vocab 的属性
print(f"\n  所有含 'vocab' 的属性:")
for k, v in config.to_dict().items():
    if "vocab" in k.lower():
        print(f"    {k} = {v}")

# 检查 config.json 原始内容
config_path = os.path.join(MODEL_PATH, "config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        raw_config = json.load(f)
    print(f"\n  config.json 原始 vocab_size = {raw_config.get('vocab_size')}")

# ============================================================
print("\n" + "=" * 60)
print("Step 2: Tokenizer 的词表大小")
print("=" * 60)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print(f"  len(tokenizer) = {len(tokenizer)}")
print(f"  tokenizer.vocab_size = {tokenizer.vocab_size}")
print(f"  tokenizer.model_max_length = {tokenizer.model_max_length}")

# 特殊 token
print(f"\n  eos_token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
print(f"  bos_token: '{tokenizer.bos_token}' (id={tokenizer.bos_token_id})")
print(f"  pad_token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
print(f"  unk_token: '{tokenizer.unk_token}' (id={tokenizer.unk_token_id})")

# 额外的特殊 token
if hasattr(tokenizer, 'additional_special_tokens'):
    asts = tokenizer.additional_special_tokens
    if asts:
        print(f"\n  additional_special_tokens ({len(asts)} 个):")
        for t in asts[:10]:
            tid = tokenizer.convert_tokens_to_ids(t)
            print(f"    '{t}' → id={tid}")
        if len(asts) > 10:
            print(f"    ... 还有 {len(asts)-10} 个")
        # 最大的 special token ID
        all_special_ids = [tokenizer.convert_tokens_to_ids(t) for t in asts]
        print(f"    最大 special token id = {max(all_special_ids)}")

# 所有 special tokens
all_special = tokenizer.all_special_tokens
all_special_ids = tokenizer.all_special_ids
print(f"\n  all_special_tokens: {len(all_special)} 个")
print(f"  all_special_ids 范围: [{min(all_special_ids)}, {max(all_special_ids)}]")

# ============================================================
print("\n" + "=" * 60)
print("Step 3: lm_head 权重的实际维度")
print("=" * 60)

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
)

# lm_head 的维度
lm_head = model.lm_head
print(f"  lm_head.weight.shape = {lm_head.weight.shape}")
print(f"  → output vocab dim = {lm_head.weight.shape[0]}")

# embed_tokens 的维度
embed = model.model.embed_tokens
print(f"  embed_tokens.weight.shape = {embed.weight.shape}")
print(f"  → input vocab dim = {embed.weight.shape[0]}")

# 对比
lm_head_vocab = lm_head.weight.shape[0]
embed_vocab = embed.weight.shape[0]
config_vocab = getattr(config, 'vocab_size', None) or config.to_dict().get('vocab_size', 'N/A')
tok_vocab = len(tokenizer)

print(f"\n  对比:")
print(f"    config.vocab_size  = {config_vocab}")
print(f"    len(tokenizer)     = {tok_vocab}")
print(f"    embed_tokens dim   = {embed_vocab}")
print(f"    lm_head dim        = {lm_head_vocab}")

if not (config_vocab == tok_vocab == embed_vocab == lm_head_vocab):
    print(f"  ⚠️ 不一致! 这可能导致越界访问")

# ============================================================
print("\n" + "=" * 60)
print("Step 4: 检查 embedding 权重中的 NaN/Inf 和未初始化区域")
print("=" * 60)

with torch.no_grad():
    # embed_tokens 检查
    e_weight = embed.weight.float()
    print(f"  embed_tokens:")
    print(f"    nan: {torch.isnan(e_weight).any().item()}")
    print(f"    inf: {torch.isinf(e_weight).any().item()}")
    print(f"    全零行数: {(e_weight.abs().sum(dim=1) == 0).sum().item()}")
    print(f"    range: [{e_weight.min().item():.4f}, {e_weight.max().item():.4f}]")
    
    # 检查高 token ID 区域的 embedding 是否正常
    check_ranges = [
        (0, 100, "前 100"),
        (151600, 151643, "151600-151643"),
        (151643, 151700, "151643-151700 (你之前限制的边界附近)"),
        (151700, 151936, "151700-151936"),
    ]
    
    for start, end, label in check_ranges:
        if end > embed_vocab:
            end = embed_vocab
        if start >= embed_vocab:
            print(f"    {label}: 超出 embedding 范围 ({embed_vocab})")
            continue
        
        chunk = e_weight[start:end]
        zero_rows = (chunk.abs().sum(dim=1) == 0).sum().item()
        nan_rows = torch.isnan(chunk).any(dim=1).sum().item()
        norm_mean = chunk.norm(dim=1).mean().item()
        
        print(f"    {label}: 全零行={zero_rows}/{end-start}, nan行={nan_rows}, 平均norm={norm_mean:.4f}")

    # 如果词表大于 151936，检查更高范围
    if embed_vocab > 151936:
        chunk = e_weight[151936:]
        zero_rows = (chunk.abs().sum(dim=1) == 0).sum().item()
        nan_rows = torch.isnan(chunk).any(dim=1).sum().item()
        norm_mean = chunk.norm(dim=1).mean().item()
        print(f"    151936-{embed_vocab}: 全零行={zero_rows}/{embed_vocab-151936}, nan行={nan_rows}, 平均norm={norm_mean:.4f}")

# ============================================================
print("\n" + "=" * 60)
print("Step 5: 用真实 prompt 测试 logits 中是否有越界 token")
print("=" * 60)

# 用一段实际的流量分析 prompt
test_text = "You are a helpful network security assistant."
input_ids = tokenizer(test_text, return_tensors="pt").input_ids.to("cuda:0")
print(f"  输入 token ids range: [{input_ids.min().item()}, {input_ids.max().item()}]")
print(f"  输入长度: {input_ids.shape[1]}")

with torch.no_grad():
    out = model(input_ids=input_ids)
    logits = out.logits[0, -1, :]  # 最后一个 token 的 logits
    
    print(f"\n  logits shape: {logits.shape}")
    print(f"  logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    print(f"  logits nan: {torch.isnan(logits).any().item()}")
    print(f"  logits inf: {torch.isinf(logits).any().item()}")
    
    # softmax
    probs = torch.softmax(logits.float(), dim=-1)
    print(f"  probs nan: {torch.isnan(probs).any().item()}")
    print(f"  probs < 0: {(probs < 0).any().item()}")
    print(f"  probs sum: {probs.sum().item():.6f}")
    
    # top tokens
    top_k = 10
    top_vals, top_ids = logits.topk(top_k)
    print(f"\n  Top {top_k} token predictions:")
    for val, tid in zip(top_vals.tolist(), top_ids.tolist()):
        tok = tokenizer.decode([tid])
        in_range = "✅" if tid < lm_head_vocab else "❌ 越界"
        print(f"    id={tid:>6} logit={val:>8.2f} token='{tok}' {in_range}")

# ============================================================
print("\n" + "=" * 60)
print("Step 6: 模拟 generate 几步，检查生成的 token ID")
print("=" * 60)

tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

for num_new_tokens in [10, 50, 100]:
    try:
        gen_ids = model.generate(
            input_ids,
            max_new_tokens=num_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
        )
        torch.cuda.synchronize()
        
        new_ids = gen_ids[0, input_ids.shape[1]:]
        max_id = new_ids.max().item()
        min_id = new_ids.min().item()
        
        out_of_range = (new_ids >= lm_head_vocab).sum().item()
        out_of_151643 = (new_ids >= 151643).sum().item()
        out_of_151936 = (new_ids >= 151936).sum().item()
        
        print(f"  generate {num_new_tokens} tokens: id range=[{min_id}, {max_id}], "
              f">=151643: {out_of_151643}, >=151936: {out_of_151936}, >=vocab: {out_of_range}")
        
        if out_of_151643 > 0:
            high_ids = new_ids[new_ids >= 151643].tolist()
            print(f"    高 ID tokens: {high_ids[:20]}")
            for hid in high_ids[:5]:
                tok = tokenizer.decode([hid])
                print(f"      id={hid} → '{tok}'")
                
    except Exception as e:
        err = str(e).split('\n')[0]
        print(f"  generate {num_new_tokens} tokens: ❌ {err}")

# ============================================================
print("\n" + "=" * 60)
print("Step 7: 检查 generation_config")
print("=" * 60)

gen_config_path = os.path.join(MODEL_PATH, "generation_config.json")
if os.path.exists(gen_config_path):
    with open(gen_config_path) as f:
        gen_config = json.load(f)
    print(f"  generation_config.json:")
    for k, v in gen_config.items():
        print(f"    {k}: {v}")
else:
    print("  没有 generation_config.json")

if hasattr(model, "generation_config"):
    gc = model.generation_config
    print(f"\n  model.generation_config:")
    print(f"    eos_token_id: {gc.eos_token_id}")
    print(f"    pad_token_id: {gc.pad_token_id}")
    print(f"    bos_token_id: {gc.bos_token_id}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)