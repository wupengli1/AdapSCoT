
import re
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from tqdm import tqdm
import torch


from transformers import AutoTokenizer, GenerationConfig
from fastNLP import logger

from llm_model2_loss9 import EfficientSoftCoTFromSmallModel
from data_loader import GSM8KLoader_all_2, StrategyQALoader, AugASDivLoader, AQuALoader, DULoader
from utils2_loss import pre_process_gsm8k


args = argparse.ArgumentParser()
args.add_argument('--base_model_id', type=str, default='/media/wupengli/premodel/premodel/Llama-3.1-8B-Instruct/')
args.add_argument('--assistant_model_id', type=str, default='/media/wupengli/premodel/premodel/Llama-3.2-1B-Instruct/')
args.add_argument('--params_file_name', type=str, default='/media/wupengli/its0/SoftCoT-main/SoftCoT-main/ckpt/output_gsm8k_sample9_01-gsm8k-10-32--/projection.bin')
# args.add_argument('--params_file_name', type=str, default=None)

args.add_argument('--base_model_ckpt', type=str, default=None)
args.add_argument('--assistant_model_ckpt', type=str, default=None)
args.add_argument('--num_thought_tokens', type=int, default=32)
args.add_argument('--num_thought_tokens_top', type=int, default=4)

args.add_argument('--num_return_sequences', type=int, default=1)
args.add_argument('--task_name', type=str, default='gsm8k')
args.add_argument('--print_input', action='store_true', default=False)
args.add_argument('--print_response', action='store_true', default=True)
args.add_argument('--test_k', type=int, default=0)
args.add_argument('--seed', type=int, default=44)
args.add_argument('--tune_base_model', action='store_true', default=False)
args.add_argument('--tune_assistant_model', action='store_true', default=False)
arg = args.parse_args()
logger.info(f'Args: {arg.__dict__}')
num_thought_tokens_top = arg.num_thought_tokens_top
base_model_id = arg.base_model_id
assistant_model_id = arg.assistant_model_id
params_file_name = arg.params_file_name
base_model_ckpt = arg.base_model_ckpt
assistant_model_ckpt = arg.assistant_model_ckpt
num_thought_tokens = arg.num_thought_tokens
num_return_sequences = arg.num_return_sequences
task_name = arg.task_name
print_input = arg.print_input
print_response = arg.print_response
test_k = arg.test_k
seed = arg.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
tune_base_model = arg.tune_base_model
tune_assistant_model = arg.tune_assistant_model

large_model_name = base_model_id.split('/')[-1]
small_model_name = assistant_model_id.split('/')[-1]

if base_model_ckpt in ['None']:
    base_model_ckpt = None
if assistant_model_ckpt in ['None']:
    assistant_model_ckpt = None

model_dtype = torch.bfloat16
param_dtype = str(model_dtype)

base_tokenizer = AutoTokenizer.from_pretrained(base_model_id, token='your-huggingface-token')
assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_model_id, token='your-huggingface-token')

if 'Llama' in base_model_id:
    base_special_token = ['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>']
    base_backbone = 'llama'
elif 'Qwen' in base_model_id:
    base_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
    # generation_config.pad_token_id = 151643
    base_backbone = 'qwen'
else:
    raise NotImplementedError
if 'Llama' in assistant_model_id:
    assistant_special_token = ['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>']
    assistant_backbone = 'llama'
elif 'Qwen' in assistant_model_id:
    assistant_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
    assistant_backbone = 'qwen'
else:
    raise NotImplementedError

model = EfficientSoftCoTFromSmallModel(
    assistant_model_id,
    base_model_id,
    num_thought_tokens,
    tune_base_model=tune_base_model,
    tune_assistant_model=tune_assistant_model,
    path_to_projection_module=params_file_name,
    path_to_small_language_model=assistant_model_ckpt,
)
#2808
# model_path = "/media/wupengli/its0/SoftCoT-main/SoftCoT-main/results/output_gsm8k_sample9_01-gsm8k-10-32--/checkpoint-8415/pytorch_model.bin"
# model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
# model.to("cuda")
logger.info(f'Successfully Init Model `{model.__class__.__name__}`')
model.eval()
model.assistant_model.eval()
model.base_model.eval()

if task_name in ['gsm8k']:
    db = GSM8KLoader_all_2().load()
    preprocess_method = pre_process_gsm8k
elif task_name in ['strategyqa']:
    db = StrategyQALoader().load()
    preprocess_method = pre_process_strategy_qa
elif task_name in ['asdiv-aug']:
    db = AugASDivLoader().load()
    preprocess_method = pre_process_gsm8k
elif task_name in ['aqua']:
    db = AQuALoader().load()
    preprocess_method = pre_process_aqua
elif task_name in ['du']:
    db = DULoader().load()
    preprocess_method = pre_process_du
else:
    raise NotImplementedError

ds = db.get_dataset('test')

if test_k > 0:
    ds = ds[: test_k]

generation_config = GenerationConfig.from_pretrained(base_model_id)
if base_backbone in ['llama']:
    generation_config.pad_token_id = 128009
elif base_backbone in ['qwen']:
    generation_config.pad_token_id = 151643
else:
    raise NotImplementedError
generation_config.top_p = 1.0
generation_config.temperature = 1.0

correct_count = 0
for idx, ins in enumerate(tqdm(ds)):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if task_name in ['gsm8k', 'asdiv-aug', 'aqua']:
        answer = ins['answer'].split('\n')[-1]
        assert answer.startswith('####')
        answer = answer.replace(',', '')
        if task_name in ['gsm8k', 'asdiv-aug']:
            if '.' in answer:
                answer = float(answer[4:])
            else:
                answer = int(answer[4:])
        else:
            answer = answer[4:].strip()
    elif task_name in ['strategyqa', 'du']:
        answer = ins['answer']
    else:
        raise NotImplementedError

    logger.info(f'Ground Truth Answer: {answer}')

    inputs = preprocess_method(
        ins, base_tokenizer, assistant_tokenizer, num_thought_tokens,
        add_bot_eot=(num_thought_tokens > 0), split='test',
        base_special_token=base_special_token,
        assistant_special_token=assistant_special_token,
        base_backbone=base_backbone,
        assistant_backbone=assistant_backbone,
        device=model.device,
    )
    if print_input:
        logger.info(f'Raw Inputs for Base Model: {base_tokenizer.decode(inputs["input_ids"][0])}')
        # logger.info(f'Raw Inputs for Assistant Model: {assistant_tokenizer.decode(inputs["assistant_input_ids"][0])}')

    terminators = [
        base_tokenizer.eos_token_id,
    ]
    if base_backbone in ['llama']:
        terminators.append(base_tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    model_answer_list = []
    model_answer_count = {}

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    inputs_embeds = model.base_model.get_input_embeddings()(inputs['input_ids'])

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    inputs_embeds,_ = model.get_inputs_embeds_for_base_model(
        inputs['assistant_input_ids'],
        inputs['assistant_attention_mask'],
        inputs['input_ids'],
        inputs_embeds,
        inputs['thought_index'],
        print_input,
    )

    thought_index = inputs['thought_index']
    input_thought_end_idx = thought_index[0,0, 1]
    # print(input_thought_end_idx)
    # print(inputs_embeds.shape)
    inputs_embeds_1 = inputs_embeds[0, :input_thought_end_idx + 2].unsqueeze(0).to(model.device)
    # labels = torch.cat((torch.tensor([-100] * (input_thought_end_idx + 2)).to(model.device), outputs[0]), 0)
    # attention_mask = [1] * len(labels)
    outputs_from_llm = model.base_model(
        inputs_embeds=inputs_embeds_1,
        output_attentions=True,
    )
    input_thought_start_idx = thought_index[0,0, 0].item()
    input_thought_end_idx = thought_index[0,0, 1].item()
    soft_token_indices = torch.arange(input_thought_start_idx, input_thought_end_idx, dtype=torch.int64).to(
        model.device)
    item = 0
    for layer_att1 in outputs_from_llm.attentions:
        # 步骤1：detach并移到cpu，避免梯度泄漏
        # print('layer_att.shape',layer_att1.shape,layer_att2.shape)
        layer_att1 = layer_att1.detach()  # .cpu()  # [1, num_heads, seq_len, seq_len]

        # 步骤2：取第一个样本（batch_size=1），并去掉最后一列（因logits右移，最后一个token无预测）
        single_layer_att1 = layer_att1[0, :, :, :]  # [num_heads, seq_len-1, seq_len]

        # 步骤3：仅保留目标token的行（注意力的行=当前预测的token，列=上下文token）
        # print(single_layer_att1.shape,single_layer_att1[:, target_token_indices[: -1],
        #               :].shape)
        target_att1 = single_layer_att1[:, :,
                      :][:, :, soft_token_indices]  # [num_heads, num_target_tokens, seq_len]
        # print(soft_grad)
        # print(torch.mean(target_att1.view(-1,in put_thought_end_i dx-input_thought_start_idx)).shape)
        if item == 0:
            soft_score = torch.mean(target_att1, (0, 1))
        else:
            soft_score += torch.mean(target_att1, (0, 1))
        item += 1
    values, indices = torch.topk(soft_score, num_thought_tokens_top)
    keep_cols = torch.ones(soft_token_indices.shape[0], dtype=torch.bool, device=soft_token_indices.device)
    keep_cols[indices] = False
    soft_token_indices_del = soft_token_indices[keep_cols]
    # print('indices', indices)

    # print('keep_cols0', keep_cols)
    # print('keep_cols0.shape', keep_cols.shape)

    keep_cols = torch.ones(inputs_embeds.shape[1], dtype=torch.bool, device=soft_token_indices.device)
    keep_cols[soft_token_indices_del] = False
    # print('0', inputs_embeds.shape)
    # print('keep_cols', keep_cols)
    # print('keep_cols.shape', keep_cols.shape)

    inputs_embeds = inputs_embeds[:, keep_cols]
    # print('1', inputs_embeds.shape)
    attention_mask = inputs['attention_mask'][:, keep_cols]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    outputs = model.base_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs['attention_mask'],
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        generation_config=generation_config,
        num_return_sequences=num_return_sequences,
    )

    for i in range(outputs.shape[0]):
        # response = outputs[i][inputs['input_ids'].shape[-1]:]
        response = outputs[i]
        raw_model_answer = base_tokenizer.decode(response, skip_special_tokens=True)

        if print_response:
            logger.info(f'Answer ({idx + 1}-{i + 1}/{len(ds)}): {base_tokenizer.decode(response)}<|end-of-response|>')

        if task_name in ['gsm8k', 'asdiv-aug']:
            cleaned_model_answer = raw_model_answer.replace(',', '')
            cleaned_model_answer = cleaned_model_answer.replace('%', '')
            cleaned_model_answer = cleaned_model_answer.replace('$', '')
        else:
            cleaned_model_answer = raw_model_answer

        match = re.findall(r'\s*([\d,]+(?:\.\d+)?)\s*', cleaned_model_answer)

        if task_name in ['gsm8k', 'asdiv-aug']:
            try:
                if match:
                    last_match = match[-1]
                    cleaned_match = last_match.replace(',', '')
                    cleaned_match = cleaned_match.replace('%', '')
                    cleaned_match = cleaned_match.replace('$', '')
                    if '.' in cleaned_match:
                        model_answer = round(float(cleaned_match), 2)
                    else:
                        model_answer = int(cleaned_match)
                else:
                    model_answer = None
                if model_answer is None and not print_response:
                    logger.info(f'None Model Answer ({idx + 1}-{i + 1}/{len(ds)}): {base_tokenizer.decode(response)}')
            except Exception as e:
                model_answer = None
                logger.error(f'Error: {e}')
        elif task_name in ['strategyqa']:
            last_yes = re.search(r'\bsey\b', raw_model_answer.lower()[::-1])
            if last_yes is not None:
                last_yes = last_yes.start()
            else:
                last_yes = len(raw_model_answer)
            last_no = re.search(r'\bon\b', raw_model_answer.lower()[::-1])
            if last_no is not None:
                last_no = last_no.start()
            else:
                last_no = len(raw_model_answer)
            if last_yes == last_no == len(raw_model_answer):
                model_answer = None
            else:
                model_answer = last_yes < last_no
        elif task_name in ['aqua', 'du']:
            m_answer = re.search(r'\b[a-f]\b', raw_model_answer.lower()[::-1])
            if m_answer is not None:
                model_answer = m_answer.group(0).upper()
            else:
                model_answer = None
        else:
            raise NotImplementedError

        model_answer_list.append(model_answer)
        if model_answer in model_answer_count and model_answer is not None:
            model_answer_count[model_answer] += 1
        else:
            model_answer_count[model_answer] = 1

    max_model_count = 0
    final_model_answer = None

    for k, v in model_answer_count.items():
        if v > max_model_count:
            final_model_answer = k
            max_model_count = v

    logger.info(f'Ground Truth Answer: {answer}')
    logger.info(f'Model Answer: {final_model_answer}')
    is_correct = (final_model_answer == answer)
    logger.info(f'Is Correct: {is_correct}')
    if is_correct:
        correct_count += 1
    logger.info(f'Correct Count: {correct_count}/{idx + 1}')
    logger.info(f'{"-" * 20}')
