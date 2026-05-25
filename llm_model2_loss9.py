import os
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import Cache
from peft import PeftModel, LoraConfig
from fastNLP import logger
import numpy as np

class SoftCoTAbstractClass(nn.Module):

    def __init__(self,
                 small_language_model_id,
                 large_language_model_id,
                 num_thought_tokens=2,
                 tune_assistant_model=False,
                 tune_base_model=False,
                 **kwargs,
                 ):
        super().__init__()
        self.assistant_model = AutoModelForCausalLM.from_pretrained(
            small_language_model_id,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            _fast_init=False,
            token='your-huggingface-token',
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            large_language_model_id,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            _fast_init=False,
            token='your-huggingface-token',
        )
        self.config = AutoConfig.from_pretrained(
            large_language_model_id,
            token='your-huggingface-token',
        )

        self.base_tokenizer = AutoTokenizer.from_pretrained(
            large_language_model_id,
            token='your-huggingface-token',
        )
        self.assistant_tokenizer = AutoTokenizer.from_pretrained(
            small_language_model_id,
            token='your-huggingface-token',
        )

        self.num_thought_tokens = num_thought_tokens
        self.tune_assistant_model = tune_assistant_model
        self.tune_base_model = tune_base_model

        self.projection = nn.Linear(self.assistant_model.config.hidden_size, self.base_model.config.hidden_size,
                                    dtype=torch.bfloat16)

        # 新增：一致性损失权重，可外部传入kwargs调节，默认0.05，数学推理任务最优区间0.03-0.1
        self.consistency_loss_weight = kwargs.get('consistency_loss_weight', 0.05)

        for n, p in self.assistant_model.named_parameters():
            p.requires_grad = tune_assistant_model
        for n, p in self.base_model.named_parameters():
            p.requires_grad = tune_base_model

        # LoRA configuration
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,  # Scaling factor
            target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA (depends on your model)
            lora_dropout=0.1,  # Dropout probability
            bias="none",  # Type of bias ("none", "all", or "lora_only")
            task_type="CAUSAL_LM"  # Task type (e.g., "SEQ2SEQ_LM", "CAUSAL_LM", etc.)
        )
        if tune_assistant_model:
            self.assistant_model = PeftModel(self.assistant_model, lora_config)
            logger.info(f'LoRA assistant model.')
        if tune_base_model:
            self.base_model = PeftModel(self.base_model, lora_config)
            logger.info(f'LoRA base model.')

    @property
    def device(self):
        return self.base_model.device

    def save_pretrained(self, save_model_dir_root: str, **kwargs):
        save_detail = []
        os.makedirs(save_model_dir_root, exist_ok=True)
        if self.tune_base_model:
            base_model_file = os.path.join(save_model_dir_root, 'base_model.bin')
            logger.info(f'Saving base model to `{base_model_file}`')
            torch.save(self.base_model.state_dict(), base_model_file)
            save_detail.append('Base Model')

        if self.tune_assistant_model:
            assistant_model_file = os.path.join(save_model_dir_root, 'assistant_model.bin')
            logger.info(f'Saving assistant model to `{assistant_model_file}`')
            torch.save(self.assistant_model.state_dict(), assistant_model_file)
            save_detail.append('Assistant Model')

        torch.save(self.projection.state_dict(), os.path.join(save_model_dir_root, 'projection.bin'))
        save_detail.append('Projection Module')
        logger.info(
            f'Saving parameters of projection module, includes: {[k for k, v in self.projection.state_dict().items()]}'
        )

        logger.info(f'Successfully saved [{", ".join(save_detail)}] to dir `{save_model_dir_root}`.')


class EfficientSoftCoTFromSmallModel(SoftCoTAbstractClass):

    def __init__(
            self,
            small_language_model_id,
            large_language_model_id,
            num_thought_tokens=2,
            tune_assistant_model=False,
            tune_base_model=False,
            path_to_projection_module=None,
            path_to_small_language_model=None,
            path_to_large_language_model=None,
            **kwargs,
    ):
        super().__init__(
            small_language_model_id=small_language_model_id,
            large_language_model_id=large_language_model_id,
            num_thought_tokens=num_thought_tokens,
            tune_assistant_model=tune_assistant_model,
            tune_base_model=tune_base_model,
            **kwargs,
        )

        if path_to_projection_module is not None and path_to_projection_module not in ['None']:
            self.projection.load_state_dict(
                torch.load(path_to_projection_module, map_location='cpu', weights_only=True))
            logger.info(f'Load weights from file `{path_to_projection_module}` for projection module.')
        self.projection.to(self.base_model.device)

        device = self.device
        if path_to_small_language_model is not None and path_to_small_language_model not in ['None']:
            self.assistant_model.load_state_dict(torch.load(path_to_small_language_model, weights_only=True))
            logger.info(f'Load weights from file `{path_to_small_language_model}` for assistant model.')
            self.assistant_model.to(device)
        if path_to_large_language_model is not None and path_to_large_language_model not in ['None']:
            self.base_model.load_state_dict(torch.load(path_to_large_language_model, weights_only=True))
            logger.info(f'Load weights from file `{path_to_large_language_model}` for base model.')
            self.base_model.to(device)

    # 新增：私有函数-计算一致性损失 核心逻辑
    def _compute_consistency_loss(self, main_proj_split, branch_proj_list):
        """
        main_proj_split: 切分后的主投影向量列表 [k段]
        branch_proj_list: 分支投影向量列表 [projected_inputs_embeds_list[1:]]
        return: 归一化的一致性损失
        """
        total_loss = 0.0
        # 余弦相似度损失：归一化避免尺度影响，loss=1-sim 越小代表越一致
        for main_seg, branch_proj in zip(main_proj_split, branch_proj_list):
            main_seg_norm = F.normalize(main_seg, dim=-1)
            branch_proj_norm = F.normalize(branch_proj, dim=-1)
            cos_sim = F.cosine_similarity(main_seg_norm, branch_proj_norm, dim=-1).mean()
            total_loss += (1 - cos_sim)
        # 归一化损失，避免分支数影响
        if len(branch_proj_list) > 0:
            total_loss = total_loss / len(branch_proj_list)
        return total_loss * self.consistency_loss_weight

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            thought_index: Optional[torch.LongTensor] = None,
            assistant_input_ids: Optional[torch.LongTensor] = None,
            assistant_attention_mask: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            valid_token_loss=None,
            valid_token_loss_mean=None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            print_index=False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_len = input_ids.size()
        consistency_loss = None

        if seq_len > 1:
            if input_ids is not None and inputs_embeds is None:
                inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

            # 核心修改：调用重构后的方法，返回 替换后的embeds + 投影向量列表
            inputs_embeds, consistency_loss = self.get_inputs_embeds_for_base_model(
                assistant_input_ids,
                assistant_attention_mask,
                input_ids,
                inputs_embeds,
                thought_index,
                print_index,
            )

            outputs_from_llm = self.base_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                labels=labels,
                output_attentions=True,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,

            )


            # 1. 取出核心变量
            if self.training:

                logits = outputs_from_llm.logits  # [batch_size, seq_len, vocab_size]
                labels = labels  # [batch_size, seq_len]
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
                single_sample_token_loss = token_loss[0]  # [seq_len-1] 一维数组，每个元素=对应token的loss
                single_sample_valid_mask = valid_token_mask[0]

                # 6. 最终结果：筛选出【仅有效token】的loss值 + 有效token的索引
                valid_token_loss_current = single_sample_token_loss[single_sample_valid_mask == 1]
                # align_len = len(valid_token_loss_current)  # min(len(base_loss), len(current_loss))
                # print('align_len',align_len)
                # 5. 获取有效token掩码（过滤padding token -100）
                valid_token_mask = (labels != -100).float()  # [bs, seq_len-1]
                # 提取单样本的有效token掩码（batch_size=1）
                single_sample_valid_mask = valid_token_mask[0].cpu().detach().bool().numpy()  # [seq_len-1]
                # 获取目标token的索引（仅有效token的位置）
                target_token_indices = np.where(single_sample_valid_mask)[0]-1 # 一维数组，如 [0,1,2,3]
                input_thought_start_idx = thought_index[0, 0,0].item()
                input_thought_end_idx = thought_index[0, 0,1].item()
                soft_token_indices = torch.arange(input_thought_start_idx, input_thought_end_idx, dtype=torch.int64).to(
                    self.device)
                # print('target_token_indices',target_token_indices,target_token_indices.shape)
                # print('target_token_indices',target_token_indices[:-1],target_token_indices[:-1].shape)

                # print('target_token_indices1',np.where(single_sample_valid_mask)[0],np.where(single_sample_valid_mask)[0].shape)
                # print('target_token_indices2',single_sample_valid_mask,single_sample_valid_mask.shape)
                # print('valid_token_loss_current',valid_token_loss_current,valid_token_loss_current.shape)

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
                    # print('att',single_layer_att1.shape,target_token_indices.shape,target_token_indices)
                    target_att1 = single_layer_att1[:, target_token_indices[:],
                                  :][:, :, soft_token_indices]  # [num_heads, n   um_target_tokens, seq_ len]
                    # print(soft_grad)
                    # print('target_att1', target_att1.shape)

                    # print(torch.mean(target_att1.view(-1,in put_thought_end_i dx-input_thought_start_idx)).shape)
                    if item == 0:
                        soft_score = torch.mean(target_att1, (0, 2))
                    else:
                        soft_score += torch.mean(target_att1, (0, 2))
                    # soft_score = torch.mean(target_att1, (0, 2))

                    item += 1
                soft_score /= len(outputs_from_llm.attentions)
                # outputs_from_llm.loss = 0.0
                # print('valid_token_loss',valid_token_loss,torch.max(valid_token_loss),torch.min(valid_token_loss))
                # print('soft_score',soft_score,torch.max(soft_score),torch.min(soft_score))
                # print('valid_token_loss',valid_token_loss,torch.max(valid_token_loss),torch.min(valid_token_loss))
                #
                a = (soft_score - torch.min(soft_score)) / (torch.max(soft_score) - torch.min(soft_score))
                # print('a',a,torch.max(a),torch.min(a))
                # print('valid_token_loss',valid_token_loss,valid_token_loss.shape)
                b = a * valid_token_loss
                # print('b',b,torch.max(b),torch.min(b))

                # a = []
                # for token_idx in range(align_len):
                #     outputs_from_llm.loss += (valid_token_loss_current[token_idx] * (
                #                 1 + b[token_idx]))
                # print('b',torch.mean(b),valid_token_loss_mean)
                outputs_from_llm.loss = outputs_from_llm.loss * (1 + torch.mean(b))
            # 融合损失：训练阶段才叠加一致性损失

            if self.training and consistency_loss is not None and outputs_from_llm.loss is not None:
                # print('consistency_loss',consistency_loss)

                outputs_from_llm.loss = outputs_from_llm.loss + consistency_loss

        else:
            outputs_from_llm = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        return outputs_from_llm

    def get_inputs_embeds_for_base_model(
            self,
            assistant_input_ids,
            assistant_attention_mask,
            input_ids,
            inputs_embeds,
            thought_index,
            print_index=False,
    ):
        if self.num_thought_tokens == 0:
            if print_index:
                logger.info(f'Number of thought tokens is zero, does not change the inputs embeds.')
            return inputs_embeds, None

        batch_size, seq_len, hidden_size = inputs_embeds.size()
        # 适配三维assistant_input_ids: [batch, n_assist, assist_seq_len]
        batch_size, n_assist, assist_seq_len = assistant_input_ids.shape
        projected_inputs_embeds_list = []
        consistency_loss = None

        # ========== 第一步：遍历所有分支，生成投影向量列表 ==========
        for assist_idx in range(n_assist):
            cur_assist_input = assistant_input_ids[:, assist_idx, :]
            cur_assist_mask = assistant_attention_mask[:, assist_idx, :]

            assistant_outputs = self.assistant_model(
                input_ids=cur_assist_input,
                attention_mask=cur_assist_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            assistant_hidden_states = assistant_outputs['hidden_states'][-1]
            projected_embeds = self.projection(assistant_hidden_states)
            projected_inputs_embeds_list.append(projected_embeds)

        # ========== 第二步：核心需求实现 - 切分+一致性损失 ==========
        if self.training and len(projected_inputs_embeds_list) > 1:
            # 取主分支投影向量 [0] + 其他分支投影向量 [1:]
            main_proj_embeds = projected_inputs_embeds_list[0]
            branch_proj_list = projected_inputs_embeds_list[1:]
            k = len(branch_proj_list)  # 其他分支的数量

            # 遍历每个样本，按thought_token区间切分+计算损失
            loss_list = []
            for b in range(batch_size):
                # 获取当前样本的thought_token起止索引
                a_s = thought_index[b, 0,2].item()
                a_e = thought_index[b,0, 3].item()
                main_proj_segments = []
                # print('main_proj_embeds',k, k, start, end)
                for i in range(k):
                    main_proj_segments.append(main_proj_embeds[b, a_s:a_e, :])
                # 遍历切分后的每一段，与对应分支计算损失
                for i in range(k):
                    branch_a_s = thought_index[b, 1+i, 2].item()
                    branch_a_e = thought_index[b, 1+i, 3].item()
                    branch_proj_seg = branch_proj_list[i][b, branch_a_s:branch_a_e, :]
                    loss_list.append(
                        self._compute_consistency_loss([main_proj_segments[i]], [branch_proj_seg.detach()]))

            # 计算批次平均一致性损失
            if loss_list:
                consistency_loss = torch.stack(loss_list).mean()

        # ========== 第三步：原始逻辑 - 用主分支投影向量替换inputs_embeds ==========
        main_proj_embeds = projected_inputs_embeds_list[0]
        for b in range(batch_size):
            input_thought_start_idx = thought_index[b, 0,0].item()
            input_thought_end_idx = thought_index[b, 0,1].item()
            assistant_thought_start_idx = thought_index[b, 0,2].item()
            assistant_thought_end_idx = thought_index[b, 0,3].item()
            inputs_embeds[b, input_thought_start_idx: input_thought_end_idx] = \
                main_proj_embeds[b, assistant_thought_start_idx: assistant_thought_end_idx]

            if print_index:
                raw_assistant_inputs = self.assistant_tokenizer.decode(
                    assistant_input_ids[b, 0, assistant_thought_start_idx: assistant_thought_end_idx])
                if input_ids is not None:
                    raw_base_inputs = self.base_tokenizer.decode(
                        input_ids[b, input_thought_start_idx: input_thought_end_idx])
                else:
                    raw_base_inputs = f'Input IDs is None, embeddings from index {input_thought_start_idx} to {input_thought_end_idx}'
                logger.info(f'Instance {b + 1}/{batch_size} - Embeddings from: <|start|>{raw_assistant_inputs}<|end|>')
                logger.info(f'Instance {b + 1}/{batch_size} - Embeddings to: <|start|>{raw_base_inputs}<|end|>')

        return inputs_embeds, consistency_loss