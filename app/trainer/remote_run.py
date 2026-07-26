#!/usr/bin/env python3
"""
firefly-client remote_run.py - Real QLoRA training

Key fix: accelerate big_modeling.py dispatch_model patch required for
bitsandbytes 0.50.0 + transformers 4.44.0 + accelerate 1.14.0.
See: accelerate_patch.py
"""
import os, sys, json, time, logging, torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("train")

MODEL_PATH  = os.environ.get("FIREFLY_MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct")
LORA_R      = int(os.environ.get("FIREFLY_LORA_RANK", "8"))
LORA_ALPHA  = int(os.environ.get("FIREFLY_LORA_ALPHA", "16"))
MAX_STEPS   = int(os.environ.get("FIREFLY_MAX_STEPS", "60"))
BS          = int(os.environ.get("FIREFLY_BATCH_SIZE", "1"))
GRAD_ACCUM  = int(os.environ.get("FIREFLY_GRAD_ACCUM", "4"))
LR          = float(os.environ.get("FIREFLY_LR", "2e-4"))
DATA_PATH   = os.environ.get("FIREFLY_DATA_PATH", "")
OUTPUT_DIR  = os.environ.get("FIREFLY_OUTPUT_DIR", "/root/.firefly/train_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info("[train] MODEL_PATH=" + MODEL_PATH)
logger.info("[train] LORA_R=%d LORA_ALPHA=%d MAX_STEPS=%d" % (LORA_R, LORA_ALPHA, MAX_STEPS))
logger.info("[train] BS=%d GRAD_ACCUM=%d LR=%s" % (BS, GRAD_ACCUM, LR))

bf16_supported = torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if bf16_supported else torch.float16
logger.info("[train] dtype=%s bf16=%s" % (dtype, bf16_supported))

logger.info("[train] loading tokenizer: " + MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

logger.info("[train] loading model (4-bit QLoRA)...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
)
logger.info("[train] model loaded, device=" + str(next(model.parameters()).device))
logger.info("[train] GPU mem=" + str(round(torch.cuda.memory_allocated()/1024**3, 2)) + " GB")

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

logger.info("[train] applying LoRA r=%d alpha=%d" % (LORA_R, LORA_ALPHA))
lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

logger.info("[train] preparing dataset...")
import json as _json
if DATA_PATH and os.path.exists(DATA_PATH):
    samples = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                item = _json.loads(line)
                text = "### Instruction:\n" + item.get("instruction","") + "\n### Response:\n" + item.get("output","")
                samples.append({"text": text})
            except: pass
else:
    samples = [
        {"text": "### Instruction:\n什么是联邦学习？\n### Response:\n联邦学习是一种分布式机器学习方法，多个参与方在不出让原始数据的前提下协作训练模型。"},
        {"text": "### Instruction:\n解释QLoRA的原理。\n### Response:\nQLoRA结合了4位量化与LoRA微调，可在单GPU上高效微调大模型。"},
        {"text": "### Instruction:\nLoRA和全量微调有什么区别？\n### Response:\nLoRA只训练少量适配器参数，大幅降低显存和计算开销。"},
        {"text": "### Instruction:\n什么是FedAvg算法？\n### Response:\nFedAvg是联邦学习的核心聚合算法，各节点本地训练后将权重上传服务器做加权平均。"},
        {"text": "### Instruction:\n为什么QLoRA适合分布式训练？\n### Response:\n因为它显存需求低，单卡即可运行，适合志愿算力参与方。"},
    ]

ds = Dataset.from_list(samples)
logger.info("[train] dataset: %d samples" % len(ds))

def formatting_func(example):
    return example["text"]

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BS,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="no",
    report_to=["none"],
    bf16=bf16_supported,
    fp16=not bf16_supported,
    dataloader_num_workers=0,
)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    formatting_func=formatting_func,
    max_seq_length=512,
    tokenizer=tokenizer,
)

t0 = time.time()
logger.info("[train] starting %d steps..." % MAX_STEPS)
trainer.train()
elapsed = time.time() - t0
logger.info("[train] done in %.1fs" % elapsed)

adapter_dir = os.path.join(OUTPUT_DIR, "adapter")
os.makedirs(adapter_dir, exist_ok=True)
trainer.save_model(adapter_dir)
logger.info("[train] saved to " + adapter_dir)

# 清理不需要的文件
for f in ["training_args.bin"]:
    p = os.path.join(adapter_dir, f)
    if os.path.exists(p):
        os.remove(p)
        logger.info("[train] removed " + f)

log_hist = trainer.state.log_history
train_loss = trainer.state.train_loss if hasattr(trainer.state, "train_loss") and trainer.state.train_loss else -1.0
if train_loss == -1.0 and log_hist:
    step_losses = [(e.get("step",-1), e.get("loss",None)) for e in log_hist if "loss" in e and "step" in e]
    if step_losses:
        train_loss = step_losses[-1][1]

meta = {
    "model_path": MODEL_PATH,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "max_steps": MAX_STEPS,
    "learning_rate": LR,
    "final_train_loss": train_loss,
    "training_time_s": round(elapsed, 1),
    "adapter_dir": adapter_dir,
}
meta_path = os.path.join(adapter_dir, "firefly_trainer_meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
logger.info("[train] meta saved: " + meta_path)
print(json.dumps(meta, indent=2, ensure_ascii=False))
