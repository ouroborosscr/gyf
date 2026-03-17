import os
import sys
import argparse

# ==========================================
# 启动方式（四种均支持，直接跑即可）：
#
#   单卡 bf16 全精度：  CUDA_VISIBLE_DEVICES=1   python    train_grpo.py
#   单卡 4-bit 量化：   CUDA_VISIBLE_DEVICES=1   python    train_grpo.py --quant4
#   双卡 bf16 全精度：  CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train_grpo.py
#   双卡 4-bit 量化：   CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train_grpo.py --quant4
# ==========================================
os.environ["PYTORCH_ALLOC_CONF"]            = "expandable_segments:True"
os.environ["TORCH_NCCL_ENABLE_MONITORING"]  = "0"
os.environ["TORCH_NCCL_BLOCKING_WAIT"]      = "0"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "60"   # 自检通过后再训练，60s 超时足够暴露真正的问题

import re
import json
import math
import random
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from pymongo import MongoClient
from peft import LoraConfig

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from utils import config
except ImportError:
    print("无法导入 utils.config，请确保路径正确")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [GRPO] - %(message)s')

# ==========================================
# 0. 路径配置
# ==========================================
MODEL_PATH     = "/date/sunchengrui/models/Qwen3.5-9B"
AGENT_RRM_PATH = "/date/sunchengrui/models/Agent-RRM"
OUTPUT_DIR     = "./qwen-agent-grpo-output"
TARGET_DATE    = "3_1"

ANALYSIS_PROMPT = (
    "我在进行流量检测,现在我将给你数条json格式的流量数据,这些流量数据中可能存在多个攻击,也可能不存在攻击。\n"
    "攻击的类型为\"内网穿透\"\"渗透隐蔽信道\"\"加密通信\"。\n"
    "我希望你能找到最靠前的攻击流量的编号和这个攻击包含连续的几条流量,并返回其攻击类型。\n\n"
    "【工具调用严格约束】：\n"
    "如果你决定调用 `report_suspicious_traffic_tool` 报警，请务必确保传入的 JSON 参数键名拼写绝对准确！\n"
    "必须严格使用 `is_suspicious` (布尔值)、`suspicious_flows_start` (整数) 和 `suspicious_flows_end` (整数)。"
    "绝对不能拼错或加入多余空格，否则会导致系统崩溃！\n\n"
    "以下是本次需要分析的流量数据：\n{flow_data_json}"
)

# 全局占位，在 main() 中根据 local_rank 初始化
reward_model_pipe = None


# ==========================================
# 1. 命令行参数解析
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="GRPO 训练脚本")
    parser.add_argument(
        "--quant4",
        action="store_true",
        default=False,
        help=(
            "启用 4-bit NF4 量化加载策略模型（显存约 5.5GB）。"
            "不传此参数则使用 bf16 全精度（显存约 18GB）。"
        ),
    )
    # torchrun 会向脚本注入若干内部参数，parse_known_args 忽略它们避免报错
    args, _ = parser.parse_known_args()
    return args


# ==========================================
# 2. 运行模式检测
# ==========================================
def detect_run_mode():
    """
    torchrun 启动时自动注入 LOCAL_RANK / WORLD_SIZE；
    python 直接启动时这些变量不存在，默认单卡模式。
    返回 (local_rank, is_ddp)。
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp     = world_size > 1
    mode_str   = (
        f"DDP 多卡（world_size={world_size}, local_rank={local_rank}）"
        if is_ddp else "单卡"
    )
    logging.info(f"运行模式：{mode_str}，设备 cuda:{local_rank}")
    return local_rank, is_ddp


# ==========================================
# 3. 策略模型加载配置
# ==========================================
def build_model_load_kwargs(use_quant4: bool) -> dict:
    """
    根据 --quant4 参数返回传给 from_pretrained 的关键字参数。

      --quant4 传入  ->  4-bit NF4 量化，显存约 5.5GB
                         适合多卡 DDP 或单卡显存不足的场景
      --quant4 不传  ->  bf16 全精度，显存约 18GB
                         单卡 80GB A100 绰绰有余，精度更高训练更稳定
    """
    if use_quant4:
        logging.info("量化策略：4-bit NF4（--quant4 已指定），显存约 5.5GB")
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        }
    else:
        logging.info("量化策略：bf16 全精度（未指定 --quant4），显存约 18GB")
        return {"torch_dtype": torch.bfloat16}


# ==========================================
# 4. 数据集构建
# ==========================================
def ensure_indexes(db, target_date):
    conn_col    = db[f"{target_date}_conn"]
    payload_col = db[f"{target_date}_payload"]
    label_col   = db[f"{target_date}_labels"]
    pred_col    = db[f"{target_date}_predictions"]

    if "ts_1" not in conn_col.index_information():
        logging.info(f"为 {target_date}_conn 创建 ts 索引...")
        conn_col.create_index([("ts", 1)])

    if "uid_1" not in payload_col.index_information():
        logging.info(f"为 {target_date}_payload 创建 uid 索引（可能需要几分钟）...")
        payload_col.create_index([("uid", 1)])
        payload_col.create_index([("ts", 1)])

    if "uid_1" not in label_col.index_information():
        label_col.create_index([("uid", 1)])

    if "uid_1" not in pred_col.index_information():
        pred_col.create_index([("uid", 1)])

    logging.info("所有索引检查完毕")


def fetch_flows_for_dataset(db, target_date, skip_idx, limit_val, max_payload_len=2000):
    conn_col    = db[f"{target_date}_conn"]
    payload_col = db[f"{target_date}_payload"]
    export_data = []
    count       = 0

    for conn_doc in conn_col.find({}).sort("ts", 1).skip(skip_idx).limit(limit_val):
        uid = conn_doc.get("uid")
        if not uid:
            continue

        full_hex, full_decoded = [], []
        packet_count = 0
        current_len  = 0

        for p_doc in payload_col.find({"uid": uid}).sort("ts", 1):
            packet_count += 1
            if max_payload_len > 0 and current_len >= max_payload_len:
                continue
            if "payload" in p_doc:
                full_hex.append(p_doc["payload"])
            if "payload_decoded" in p_doc:
                frag = p_doc["payload_decoded"]
                full_decoded.append(frag)
                current_len += len(frag)

        entry = conn_doc.copy()
        if "_id" in entry:
            entry["mongo_id"] = str(entry["_id"])
            del entry["_id"]
        if "ts_date" in entry and hasattr(entry["ts_date"], "isoformat"):
            entry["ts_date"] = entry["ts_date"].isoformat()

        entry["batch_index"]           = count + 1
        entry["packet_count_captured"] = packet_count

        hex_str     = "".join(full_hex)
        decoded_str = "".join(full_decoded)
        if max_payload_len > 0:
            if len(hex_str) > max_payload_len:
                hex_str = hex_str[:max_payload_len] + "...[TRUNCATED]"
            if len(decoded_str) > max_payload_len:
                decoded_str = decoded_str[:max_payload_len] + "...[TRUNCATED]"

        entry["stream_payload_hex"]     = hex_str
        entry["stream_payload_decoded"] = decoded_str
        export_data.append(entry)
        count += 1

    return export_data


def build_grpo_dataset(target_date=TARGET_DATE, range_start=1, range_end=3000, num_samples=200):
    logging.info(f"从 MongoDB 构建训练数据集（随机抽取 {num_samples} 组）...")
    client = MongoClient(config.DATABASE["mongo"]["uri"])
    db     = client[config.DATABASE["mongo"]["db_name"]]

    ensure_indexes(db, target_date)

    label_col       = db[f"{target_date}_labels"]
    pred_col        = db[f"{target_date}_predictions"]
    prompts         = []
    ground_truths   = []

    for _ in range(num_samples):
        L        = random.randint(20, 40)
        min_skip = max(0, range_start - 1)
        max_skip = max(min_skip, range_end - L)
        S        = random.randint(min_skip, max_skip)

        flows = fetch_flows_for_dataset(db, target_date, S, L)
        if not flows:
            continue

        prompts.append(ANALYSIS_PROMPT.replace(
            "{flow_data_json}", json.dumps(flows, ensure_ascii=False, indent=2)
        ))

        uids        = [f["uid"] for f in flows]
        true_labels = {
            d["uid"]: d["label"]
            for d in label_col.find({"uid": {"$in": uids}}, {"uid": 1, "label": 1})
        }
        missing = [u for u in uids if u not in true_labels]
        if missing:
            true_labels.update({
                d["uid"]: d["label"]
                for d in pred_col.find({"uid": {"$in": missing}}, {"uid": 1, "label": 1})
            })

        flow_labels           = []
        true_start            = None
        true_end              = None
        in_first_attack_block = False
        first_attack_ended    = False

        for i, uid in enumerate(uids):
            lbl       = true_labels.get(uid, "Unknown")
            is_attack = lbl != "Unknown" and str(lbl).lower() != "benign"
            flow_labels.append(1 if is_attack else 0)
            local_idx = i + 1

            if is_attack and not first_attack_ended:
                if not in_first_attack_block:
                    true_start            = local_idx
                    true_end              = local_idx
                    in_first_attack_block = True
                else:
                    true_end = local_idx
            elif not is_attack and in_first_attack_block:
                first_attack_ended = True

        ground_truths.append({
            "start":       true_start,
            "end":         true_end,
            "skip_val":    S,
            "limit_val":   L,
            "flow_labels": flow_labels,
        })

    logging.info(f"数据集构建完成，共 {len(prompts)} 条样本")
    return Dataset.from_dict({"prompt": prompts, "ground_truth": ground_truths})


# ==========================================
# 5. 奖励函数
# ==========================================
def format_reward_func(completions, **kwargs):
    return [
        1.0 if re.search(r"<think>.*?</think>", c, re.DOTALL) else -1.0
        for c in completions
    ]


def correctness_reward_func(completions, ground_truth, **kwargs):
    rewards = []
    for comp, truth in zip(completions, ground_truth):
        true_start  = truth.get("start")
        true_end    = truth.get("end")
        flow_labels = truth.get("flow_labels", [])

        start_match = re.search(r'(?:start|suspicious_flows_start)["\']?\s*[:=]\s*(\d+)', comp)
        end_match   = re.search(r'(?:end|suspicious_flows_end)["\']?\s*[:=]\s*(\d+)', comp)

        if not (start_match and end_match):
            rewards.append(1.0 if true_start is None else -1.0)
            continue

        pred_start = int(start_match.group(1))
        pred_end   = int(end_match.group(1))

        if pred_start > pred_end or pred_start < 1:
            rewards.append(-1.0)
            continue

        if true_start is None:
            rewards.append(-1.0)
            continue

        if pred_start == true_start and pred_end == true_end:
            rewards.append(1.0)
            continue

        if pred_end < true_start or pred_start > true_end:
            if pred_end < true_start:
                score = -0.5 + 0.4 * math.exp(-0.2 * (true_start - pred_end))
            else:
                score = -0.9 + 0.4 * math.exp(-0.2 * (pred_start - true_end))
            rewards.append(score)
            continue

        if pred_start <= true_start and pred_end >= true_end:
            penalty = sum(
                0.2 if (0 <= i - 1 < len(flow_labels) and flow_labels[i - 1] == 1) else 1.0
                for i in range(pred_start, pred_end + 1)
                if not (true_start <= i <= true_end)
            )
            rewards.append(0.5 + 0.4 * math.exp(-0.2 * penalty))
            continue

        if pred_start >= true_start and pred_end <= true_end:
            penalty = (pred_start - true_start) * 1.0 + (true_end - pred_end) * 0.2
            rewards.append(0.1 + 0.4 * math.exp(-0.3 * penalty))
            continue

        intersection = min(pred_end, true_end) - max(pred_start, true_start) + 1
        union        = max(pred_end, true_end) - min(pred_start, true_start) + 1
        rewards.append(-0.1 + 0.2 * (intersection / union if union > 0 else 0))

    return rewards


def logic_reward_func(completions, **kwargs):
    """
    DDP 安全版：只在 rank 0 上运行奖励模型推理，其他 rank 直接返回 0。
    原因：reward_model_pipe 推理会长时间持有 GIL，若所有 rank 同时跑
    会与 NCCL 集合通信产生死锁（watchdog 超时 SIGABRT）。
    rank 0 打分 → GRPOTrainer 内部会 allreduce 汇总 → 结果等价。
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        # 非主进程直接返回 0 分，不触碰奖励模型，让 NCCL 无障碍同步
        return [0.0] * len(completions)

    rewards = []
    for comp in completions:
        try:
            out = reward_model_pipe(
                f"Please evaluate the following agent reasoning and action:\n{comp}\n\n",
                max_new_tokens=150,
                max_length=None,
                return_full_text=False,
            )[0]["generated_text"]
            m = re.search(r"<score>\s*([0-9.]+)\s*</score>", out)
            rewards.append(float(m.group(1)) if m else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


# ==========================================
# 6. 主函数
# ==========================================
def ddp_sanity_check(model, is_ddp: bool):
    """
    DDP 快速自检：在 trainer.train() 之前主动做一次小型 allreduce，
    几秒内验证所有 rank 的参数数量和形状一致，发现问题立刻报错，
    不再等 30 分钟 NCCL watchdog 超时。
    """
    if not is_ddp:
        return

    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 1. 收集本 rank 的可训练参数数量
    trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
    param_tensor = torch.tensor([trainable_params], dtype=torch.long, device=f"cuda:{local_rank}")

    # 2. allgather 到所有 rank（小张量，毫秒级完成）
    gathered = [torch.zeros(1, dtype=torch.long, device=f"cuda:{local_rank}") for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, param_tensor)

    # 3. 检查所有 rank 的参数数量是否一致
    counts = [t.item() for t in gathered]
    if len(set(counts)) != 1:
        raise RuntimeError(
            f"\n\n❌ DDP 自检失败：各 rank 可训练参数数量不一致！\n"
            f"   各 rank 参数数：{dict(enumerate(counts))}\n"
            f"   这通常是 LoRA 注入不一致导致的，请检查 device_map 配置。\n"
        )

    if local_rank == 0:
        logging.info(f"✅ DDP 自检通过：所有 rank 可训练参数数量一致（{counts[0]} 个）")


def main():
    args               = parse_args()
    local_rank, is_ddp = detect_run_mode()
    device             = f"cuda:{local_rank}"

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 奖励模型（每个进程绑定到自己的卡）
    global reward_model_pipe
    logging.info(f"加载 Agent-RRM 至 {device} ...")
    reward_model_pipe = pipeline(
        "text-generation",
        model=AGENT_RRM_PATH,
        dtype=torch.bfloat16,
        device=device,
    )

    # 策略模型（量化由 --quant4 控制）
    # ── 兼容性检查 ──────────────────────────────────────────────────────────
    # 1. DDP + quant4：bitsandbytes 量化模型无法被 DDP 正确包装，禁止组合。
    # 2. DDP 模式下不能传 device_map=device（字符串形式）：
    #    accelerate 的 dispatch_model 会与 DDP 包装冲突，
    #    导致 LoRA 只注入 rank 0，rank 1 参数为 0，ALLGATHER 永远挂起。
    #    正确做法：DDP 时不传 device_map，由 GRPOTrainer 内部统一处理设备分配。
    # ────────────────────────────────────────────────────────────────────────
    if is_ddp and args.quant4:
        raise ValueError(
            "\n\n❌ 不支持的组合：DDP 多卡 + --quant4\n"
            "   bitsandbytes 4-bit 量化与 DDP 不兼容。\n"
            "   多卡请直接用 bf16（两卡各 ~18GB，80GB A100 完全够用）：\n"
            "     CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train_grpo.py\n"
        )

    load_kwargs = build_model_load_kwargs(args.quant4)

    if is_ddp:
        # DDP 模式：不传 device_map，从 CPU 加载后由 GRPOTrainer/DDP 统一放卡
        # 这样 LoRA 能在所有 rank 上一致注入，DDP param 校验才能通过
        logging.info(f"加载 Qwen3.5-9B（DDP 模式，CPU 加载交由 Trainer 分配设备）...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            **load_kwargs,
        )
    else:
        # 单卡模式：直接指定设备，快速加载
        logging.info(f"加载 Qwen3.5-9B 至 {device} ...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map=device,
            **load_kwargs,
        )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # 数据集
    dataset = build_grpo_dataset(
        target_date=TARGET_DATE,
        range_start=1,
        range_end=3000,
        num_samples=200,
    )

    # 训练参数
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        logging_steps=10,
        max_steps=500,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=2048,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        temperature=0.7,
        top_p=0.9,
    )

    # LoRA（全量微调会 OOM，始终保留）
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["o_proj", "out_proj"],
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, correctness_reward_func, logic_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 训练前快速自检，几秒内验证 DDP 配置正确，避免等 30 分钟超时
    ddp_sanity_check(trainer.model, is_ddp)

    logging.info("开始 GRPO 强化学习训练...")
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_agent_model"))
    logging.info("训练完成，模型已保存！")


if __name__ == "__main__":
    main()