import argparse
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from fastNLP import logger


from data_loader import GSM8KLoader_all_2,GSM8KLoader, StrategyQALoader, AugASDivLoader, AQuALoader
from llm_model2_loss9 import EfficientSoftCoTFromSmallModel
from utils2_loss import pre_process_gsm8k, CustomDataCollator
import warnings
import numpy as np

# 在训练开始前添加
warnings.filterwarnings("ignore", message="Some tensors share memory")

args = argparse.ArgumentParser( )
args.add_argument('--large_model_id', type=str, default='/media/wupengli/premodel/premodel/Llama-3.1-8B-Instruct/')
args.add_argument('--small_model_id', type=str, default='/media/wupengli/premodel/premodel/Llama-3.2-1B-Instruct/')
args.add_argument('--output_name', type=str, default='output_gsm8k_sample9_03')
args.add_argument('--batch_size', type=int, default=1)
args.add_argument('--task_name', type=str,default='gsm8k' )#choices=['gsm8k', 'strategyqa', 'asdiv-aug', 'aqua',]
args.add_argument('--num_thought_tokens', type=int, default=32)
args.add_argument('--n_epochs', type=float, default=10)
args.add_argument('--k_shot', type=int, default=0)
args.add_argument('--tune_base_model', action='store_true', default=False)
args.add_argument('--tune_assistant_model', action='store_true', default=False)


arg = args.parse_args()

logger.info(f'args: {arg.__dict__}')

large_model_id = arg.large_model_id
small_model_id = arg.small_model_id
output_name = arg.output_name
batch_size = arg.batch_size
task_name = arg.task_name
n_epochs = arg.n_epochs
num_thought_tokens = arg.num_thought_tokens
k_shot = arg.k_shot
tune_base_model = arg.tune_base_model
tune_assistant_model = arg.tune_assistant_model


large_model_name = large_model_id.split('/')[-1]
small_model_name = small_model_id.split('/')[-1]
post_fix = f'{task_name}-{n_epochs}-{num_thought_tokens}-{large_model_name}-{small_model_name}'
output_dir = f'./results/{output_name}-{post_fix}'
log_dir = f'./logs/{output_name}-{post_fix}'
save_model_dir = f'./ckpt/{output_name}-{post_fix}'

logger.info(f'Output Dir: {output_dir}')
logger.info(f'Log Dir: {log_dir}')
logger.info(f'Save Model Dir: {save_model_dir}')

model_dtype = torch.bfloat16
param_dtype = str(model_dtype)

base_tokenizer = AutoTokenizer.from_pretrained(large_model_id, token='your-huggingface-token')
assistant_tokenizer = AutoTokenizer.from_pretrained(small_model_id, token='your-huggingface-token')

if 'Llama' in large_model_id:
    base_special_token = ['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>']
    base_backbone = 'llama'
elif 'Qwen' in large_model_id:
    base_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
    base_backbone = 'qwen'
else:
    raise NotImplementedError
if 'Llama' in small_model_id:
    assistant_special_token = ['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>']
    assistant_backbone = 'llama'
elif 'Qwen' in small_model_id:
    assistant_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
    assistant_backbone = 'qwen'
else:
    raise NotImplementedError

model = EfficientSoftCoTFromSmallModel(
    small_model_id,
    large_model_id,
    num_thought_tokens,
    tune_base_model=tune_base_model,
    tune_assistant_model=tune_assistant_model,
)

logger.info(f'Successfully Init Model `{model.__class__.__name__}`')

trainable_param = 0
total_param = 0
for n, p in model.named_parameters():
    if p.requires_grad:
        trainable_param += p.view(-1).size(0)
    total_param += p.view(-1).size(0)
logger.info(f'Trainable Parameters: {trainable_param}; Total Parameters: {total_param}')

if task_name in ['gsm8k']:
    db = GSM8KLoader_all_2().load()
    preprocess_method = pre_process_gsm8k
elif task_name in ['strategyqa']:
    db = StrategyQALoader().load()
    # preprocess_method = pre_process_strategy_qa
elif task_name in ['asdiv-aug']:
    db = AugASDivLoader().load()
    preprocess_method = pre_process_gsm8k
elif task_name in ['aqua']:
    db = AQuALoader().load()
    # preprocess_method = pre_process_aqua
else:
    raise NotImplementedError

train_dataset = db.get_dataset('train')
eval_dataset = db.get_dataset('dev')

if k_shot > 0:
    train_dataset = train_dataset[: k_shot]
# 核心判断逻辑：文件存在 → 读取，不存在 → 初始化空列表
if os.path.exists('./llama_token_loss_list.npz'):
    # 文件存在，读取numpy数组列表
    loaded_data = np.load('./llama_token_loss_list.npz', allow_pickle=True)
    outputs_from_llm_base = loaded_data["loss_list"].tolist()
    print(f"✅ 成功读取历史Loss文件！列表长度：{len(outputs_from_llm_base)}")
else:
    # 文件不存在，初始化空列表，不影响后续逻辑
    outputs_from_llm_base = []
    print(f"⚠️ 未找到文件 LOSS_SAVE_PATH ，已初始化空的Loss列表")
train_rows = []
def base_loss(inputs):
    inputs = {
        k: torch.tensor(v).unsqueeze(0).to('cuda') for k, v in inputs.items() if isinstance(v, List)
    }
    inputs_embeds = model.base_model.get_input_embeddings()(inputs['input_ids'])

    outputs_from_llm_base = model.base_model(
        attention_mask=inputs['attention_mask'],
        inputs_embeds=inputs_embeds,
        labels=inputs['labels'],
        return_dict=True,
    )
    logits = outputs_from_llm_base.logits  # [batch_size, seq_len, vocab_size]
    labels = inputs['labels']  # [batch_size, seq_len]
    batch_size, seq_len = labels.shape

    # 2. 核心：LLaMA自回归对齐 - logits右移一位，和labels匹配 (因果预测逻辑)
    shift_logits = logits[..., :-1, :].contiguous()  # 去掉最后一个token的预测 [bs, seq_len-1, vocab]
    shift_labels = labels[..., 1:].contiguous()  # 去掉第一个token的标签 [bs, seq_len-1]

    # 3. 计算【逐token的原始损失值】- 核心API，和模型内部loss计算逻辑完全一致
    # 结果 shape: [batch_size, sequence_length - 1] → 每个位置对应一个token的loss
    token_loss = F.cross_entropy(
        input=shift_logits.reshape(-1, shift_logits.size(-1)),  # 展平计算交叉熵
        target=shift_labels.reshape(-1),
        reduction='none'  # 关键参数：none → 不做任何聚合，返回每个token的独立loss
    ).view(batch_size, -1)

    # 4. 【可选但推荐】获取每个token的有效状态：过滤掉padding的token(-100)
    # mask=1 → 有效token，mask=0 → padding token(不计损失)
    valid_token_mask = (shift_labels != -100).float()

    # 5. 整合：有效token的loss + 对应token的位置 (解决你的可视化需求)
    # 取第0个样本(99%场景都是单样本推理，batch_size=1)，多样本可循环
    # print(token_loss)
    single_sample_token_loss = token_loss[0].cpu().detach().float().numpy()  # [seq_len-1] 一维数组，每个元素=对应token的loss
    single_sample_valid_mask = valid_token_mask[0].cpu().detach().float().numpy()

    # 6. 最终结果：筛选出【仅有效token】的loss值 + 有效token的索引
    valid_token_loss = single_sample_token_loss[single_sample_valid_mask == 1]
    # valid_token_indices = np.where(single_sample_valid_mask == 1)[0]
    # print(valid_token_loss)
    valid_token_loss_mean = valid_token_loss.mean()
    return valid_token_loss,valid_token_loss_mean
len_ins = 0
for ins in tqdm(train_dataset, desc='Preprocess Training Set'):
    inputs = preprocess_method(
        ins, base_tokenizer, assistant_tokenizer, num_thought_tokens,
        add_bot_eot=True, split='train',
        base_special_token=base_special_token,
        assistant_special_token=assistant_special_token,
        base_backbone=base_backbone,
        assistant_backbone=assistant_backbone,
    )
    if not os.path.exists('./llama_token_loss_list.npz'):
        valid_token_loss, valid_token_loss_mean = base_loss(inputs)
        outputs_from_llm_base.append([valid_token_loss, valid_token_loss_mean])
    else:
        valid_token_loss, valid_token_loss_mean = outputs_from_llm_base[len_ins//2]
    inputs['valid_token_loss_mean'] = valid_token_loss_mean
    inputs['valid_token_loss'] = valid_token_loss
    train_rows.append(inputs)
    len_ins += 1
if not os.path.exists('./llama_token_loss_list.npz'):
    np.savez("./llama_token_loss_list.npz", loss_list=outputs_from_llm_base)


eval_rows = []
# 核心判断逻辑：文件存在 → 读取，不存在 → 初始化空列表
if os.path.exists('./llama_token_loss_list_eval.npz'):
    # 文件存在，读取numpy数组列表
    loaded_data = np.load('./llama_token_loss_list_eval.npz', allow_pickle=True)
    outputs_from_llm_base_eval = loaded_data["loss_list"].tolist()
    print(f"✅ 成功读取历史Loss文件！列表长度：{len(outputs_from_llm_base_eval)}")
else:
    # 文件不存在，初始化空列表，不影响后续逻辑
    outputs_from_llm_base_eval = []
    print(f"⚠️ 未找到文件 LOSS_SAVE_PATH ，已初始化空的Loss列表")
len_ins = 0
for ins in tqdm(eval_dataset, desc='Preprocess Testing Set'):
    inputs = preprocess_method(
        ins, base_tokenizer, assistant_tokenizer, num_thought_tokens,
        add_bot_eot=True, split='dev',
        base_special_token=base_special_token,
        assistant_special_token=assistant_special_token,
        base_backbone=base_backbone,
        assistant_backbone=assistant_backbone,
    )
    if not os.path.exists('./llama_token_loss_list_eval.npz'):
        valid_token_loss, valid_token_loss_mean = base_loss(inputs)
        outputs_from_llm_base_eval.append([valid_token_loss, valid_token_loss_mean])
    else:
        valid_token_loss, valid_token_loss_mean = outputs_from_llm_base_eval[len_ins//2]
    # print('valid_token_loss',valid_token_loss)
    # print('valid_token_loss_mean',[valid_token_loss_mean])

    inputs['valid_token_loss_mean'] = valid_token_loss_mean
    inputs['valid_token_loss'] = valid_token_loss
    eval_rows.append(inputs)
    len_ins += 1
if not os.path.exists('./llama_token_loss_list_eval.npz'):
    np.savez("./llama_token_loss_list_eval.npz", loss_list=outputs_from_llm_base_eval)

train_data = Dataset.from_pandas(pd.DataFrame(train_rows))
eval_data = Dataset.from_pandas(pd.DataFrame(eval_rows))

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=n_epochs,
    save_total_limit=10 if task_name in ['gsm8k', 'aqua'] else 2,
    bf16=True,
    logging_dir=log_dir,
    logging_steps=500,
    remove_unused_columns=True,
save_safetensors=False,
    weight_decay=0.01,
gradient_accumulation_steps=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=CustomDataCollator(),
)
trainer.train()
model.save_pretrained(save_model_dir)
logger.info(f'Finish training, save model to dir `{save_model_dir}`')


