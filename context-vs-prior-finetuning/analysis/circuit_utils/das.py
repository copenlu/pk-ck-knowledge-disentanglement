from pyvene import (
    IntervenableModel,
    LowRankRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from collections import OrderedDict
import pandas as pd
import re
import os
from torch.nn import CrossEntropyLoss
import torch
from tqdm import trange
from datasets import Dataset
from nnsight import NNsight
from main import load_model_and_tokenizer
from model_utils.utils import MODEL_ID_TO_TEMPLATES_DICT, construct_query_with_demonstrations 
import random
from nnpatch.api.gemma import Gemma2
from analysis.circuit_utils.utils import encode_answer
from analysis.circuit_utils.decoding import get_patched_residuals, get_probs, patch_scope, config_to_site


PROMPT_MODE_TO_INSTRUCTION = {
    "prior_only": "Ignore the context in answering the query.",
    "context_only": "Only consider the context in answering the query.",
    "both_w_instruction": "Consider the context in answering the query."
}


def apply_chat_template(PATHS, tokenizer, query, context, prompt_mode, dataset):
            model_name = PATHS["MODEL_NAME"]
            if dataset == 'strategyqa':
                system_prompt = "Answer the following query considering the provided context. Answer in either of these two: True or False. If you are unable to answer the query, respond Unknown."
                if prompt_mode == "prior_wo_context":
                    user_prompt = """Query: %s\n""" % (query.strip())
                    system_prompt = "Answer the following query. Answer in either of these two: True or False. If you are unable to answer the query, respond Unknown."
                elif prompt_mode == "prior_only" or prompt_mode == "context_only" or prompt_mode == "both_w_instruction":
                    user_prompt = """Context: %s\nInstruction: %s\nQuery: %s\n""" % (context.strip(), PROMPT_MODE_TO_INSTRUCTION[prompt_mode].strip(), query.strip())
                elif prompt_mode == 'both_wo_instruction':
                    user_prompt = """Context: %s\nQuery: %s\n""" % (context.strip(), query.strip())

            else:
                system_prompt = "Answer the following query considering the provided context. Answer with only one word."
                if prompt_mode == "prior_wo_context":
                    user_prompt = """Query: %s\n""" % (query.strip())
                    system_prompt = "Answer the following query. Answer with only one word."
                elif prompt_mode == "prior_only" or prompt_mode == "context_only" or prompt_mode == "both_w_instruction":
                    user_prompt = """Context: %s\nInstruction: %s\nQuery: %s\n""" % (context.strip(), PROMPT_MODE_TO_INSTRUCTION[prompt_mode].strip(), query.strip())
                elif prompt_mode == 'both_wo_instruction':
                    user_prompt = """Context: %s\nQuery: %s\n""" % (context.strip(), query.strip())

            if 'gemma' in model_name:
                chat = [
                    {
                "role": "user",
                "content": f'{system_prompt}\n{user_prompt}'
                }]
            else:
                chat = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }]

            tokens = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
            return tokenizer.decode(tokens)

def load_map_training_data(tokenizer, args, PATHS):
    train_data = pd.read_csv(PATHS["TRAIN_DATA"])
    train_data.head()

    prior_only_texts, context_only_texts, both_texts = [], [], []
    for row in train_data.itertuples(index=False):
        text_prior_only = apply_chat_template(PATHS, tokenizer, getattr(row, "query"), getattr(row, "context"), prompt_mode='prior_only', dataset=args.dataset)
        text_context_only = apply_chat_template(PATHS, tokenizer, getattr(row, "query"), getattr(row, "context"), prompt_mode='context_only', dataset=args.dataset)
        text_both = apply_chat_template(PATHS, tokenizer, getattr(row, "query"), getattr(row, "context"), prompt_mode='both_w_instruction', dataset=args.dataset)
        prior_only_texts.append(text_prior_only)
        context_only_texts.append(text_context_only)
        both_texts.append(text_both)
    train_data["prior_only"] = prior_only_texts
    train_data["context_only"] = context_only_texts
    train_data["text"] = both_texts   

    target_texts, source_texts_1, source_texts_2, target_answer, source_answer_1, source_answer_2, texts_inverted = [], [], [], [], [], [], []
    for row in train_data.itertuples(index=False):
        target_texts.append(getattr(row, "text"))
        target_answer.append(getattr(row, "answer"))
        source_texts_1.append(getattr(row, "prior_only"))
        source_answer_1.append(getattr(row, "prior_answer"))
        source_texts_2.append(getattr(row, "context_only"))
        source_answer_2.append(getattr(row, "ctx_answer"))
        texts_inverted.append(getattr(row, "prior_only"))

        # target_texts.append(getattr(row, "text"))
        # target_answer.append(getattr(row, "answer"))
        # source_texts.append(getattr(row, "context_only"))
        # source_answer.append(getattr(row, "ctx_answer"))
        # texts_inverted.append(getattr(row, "context_only"))

    return target_texts, source_texts_1, source_texts_2, target_answer, source_answer_1, source_answer_2, texts_inverted


def load_map_test_data(tokenizer, args, PATHS):
    train_data = pd.read_csv(PATHS["TEST_DATA"])
    train_data.head()

    prior_only_texts, context_only_texts, both_texts = [], [], []
    for row in train_data.itertuples(index=False):
        text_prior_only = apply_chat_template(PATHS, tokenizer, getattr(row, "query"), getattr(row, "context"), prompt_mode='prior_only', dataset=args.dataset)
        text_context_only = apply_chat_template(PATHS, tokenizer, getattr(row, "query"), getattr(row, "context"), prompt_mode='context_only', dataset=args.dataset)
        text_both = apply_chat_template(PATHS, tokenizer, getattr(row, "query"), getattr(row, "context"), prompt_mode='both_w_instruction', dataset=args.dataset)
        prior_only_texts.append(text_prior_only)
        context_only_texts.append(text_context_only)
        both_texts.append(text_both)
    train_data["prior_only"] = prior_only_texts
    train_data["context_only"] = context_only_texts
    train_data["text"] = both_texts  

    target_texts, source_texts_1, source_texts_2, target_answer, source_answer_1, source_answer_2, texts_inverted = [], [], [], [], [], [], []
    for row in train_data.itertuples(index=False):
        target_texts.append(getattr(row, "text"))
        target_answer.append(getattr(row, "answer"))
        source_texts_1.append(getattr(row, "prior_only"))
        source_answer_1.append(getattr(row, "prior_answer"))
        source_texts_2.append(getattr(row, "context_only"))
        source_answer_2.append(getattr(row, "ctx_answer"))
        texts_inverted.append(getattr(row, "prior_only"))

        # target_texts.append(getattr(row, "text"))
        # target_answer.append(getattr(row, "answer"))
        # source_texts.append(getattr(row, "context_only"))
        # source_answer.append(getattr(row, "ctx_answer"))
        # texts_inverted.append(getattr(row, "context_only"))

    return target_texts, source_texts_1, source_texts_2, target_answer, source_answer_1, source_answer_2, texts_inverted


def remove_instruction(text, name_of_instruction="Instruction", replace_with="Instruction: Answer with one word only.\n"):
    pattern = f'{name_of_instruction}:.*?(?=Query:)'
    return re.sub(pattern, replace_with, text, flags=re.DOTALL)


def create_dataset(source_tokens_1, source_tokens_2, target_tokens, source_label_index_1, source_label_index_2, target_label_index, source_attn_mask_1, source_attn_mask_2, target_attn_mask, *args):
    # Create a dataset with the same structure as the original data
    dataset = Dataset.from_dict({
        'input_ids': target_tokens.tolist(),
        'attention_mask': target_attn_mask.tolist(),
        'source_input_ids_1': source_tokens_1.tolist(),
        'source_input_ids_2': source_tokens_2.tolist(),
        'source_attention_mask_1': source_attn_mask_1.tolist(),
        'source_attention_mask_2': source_attn_mask_2.tolist(),
        'labels_prior': torch.stack([source_label_index_1, target_label_index], dim=1).tolist(),
        'labels_context': torch.stack([source_label_index_2, target_label_index], dim=1).tolist(),
        'sources->base': [[target_tokens.shape[1] - 1]] * len(target_tokens),
        # 'source_0->base.0.pos': [[target_tokens.shape[1] - 1]] * len(target_tokens),
        # 'source_0->base.1.pos': [[target_tokens.shape[1] - 1]] * len(target_tokens),
        'subspaces_prior': [0] * len(target_tokens),
        'subspaces_context': [1] * len(target_tokens),
    })
    return dataset

def prepare_train_data(args, PATHS, tokenizer, device, same_query=False, remove_weight=False, name_of_instruction="Instruction", seed=42):
    target_texts, source_texts_1, source_texts_2, target_answer, source_answer_1, source_answer_2, texts_inverted = load_map_training_data(tokenizer, args, PATHS)
    random.seed(seed)
    
    source_index_1, target_index  = encode_answer(source_answer_1, target_answer, tokenizer, device, args)
    source_index_2, target_index  = encode_answer(source_answer_2, target_answer, tokenizer, device, args)
    # same_answer_indices = (source_index == target_index).detach().cpu()
    target_texts = [target_texts[i] for i in range(len(target_texts))]# if not same_answer_indices[i]]
    source_texts_1 = [source_texts_1[i] for i in range(len(source_texts_1))]# if not same_answer_indices[i]]
    source_texts_2 = [source_texts_2[i] for i in range(len(source_texts_2))]# if not same_answer_indices[i]]
    # source_index = source_index[~same_answer_indices]
    # target_index = target_index[~same_answer_indices]
    
    source_tokens_1 = tokenizer(source_texts_1, return_tensors="pt", padding=True)
    source_tokens_2 = tokenizer(source_texts_2, return_tensors="pt", padding=True)
    attention_mask_source_1 = source_tokens_1["attention_mask"].to(device)
    attention_mask_source_2 = source_tokens_2["attention_mask"].to(device)
    target_tokens = tokenizer(target_texts, return_tensors="pt", padding=True)
    attention_mask_target = target_tokens["attention_mask"].to(device)
    source_tokens_1 = source_tokens_1["input_ids"].to(device)
    source_tokens_2 = source_tokens_2["input_ids"].to(device)
    target_tokens = target_tokens["input_ids"].to(device)
    
    max_len = max(source_tokens_1.shape[1], source_tokens_2.shape[1], target_tokens.shape[1])
    source_tokens_1 = torch.nn.functional.pad(source_tokens_1, (max_len - source_tokens_1.shape[1], 0), value=tokenizer.pad_token_id)
    source_tokens_2 = torch.nn.functional.pad(source_tokens_2, (max_len - source_tokens_2.shape[1], 0), value=tokenizer.pad_token_id)
    target_tokens = torch.nn.functional.pad(target_tokens, (max_len - target_tokens.shape[1], 0), value=tokenizer.pad_token_id)
    attention_mask_source_1 = torch.nn.functional.pad(attention_mask_source_1, (max_len - attention_mask_source_1.shape[1], 0), value=0)
    attention_mask_source_2 = torch.nn.functional.pad(attention_mask_source_2, (max_len - attention_mask_source_2.shape[1], 0), value=0)
    attention_mask_target = torch.nn.functional.pad(attention_mask_target, (max_len - attention_mask_target.shape[1], 0), value=0)
    
    # texts_inverted = train_data["text_inverted"].tolist()
    texts_inverted = [texts_inverted[i] for i in range(len(texts_inverted))]# if not same_answer_indices[i]]
    target_inverted_tokens = tokenizer(texts_inverted, return_tensors="pt", padding=True)
    attention_mask_target_inverted = target_inverted_tokens["attention_mask"].to(device)
    target_inverted_tokens = target_inverted_tokens["input_ids"].to(device)
    return source_tokens_1, source_tokens_2, target_tokens, source_index_1, source_index_2, target_index, attention_mask_source_1, attention_mask_source_2, attention_mask_target, target_inverted_tokens, attention_mask_target_inverted


def prepare_test_data(args, PATHS, tokenizer, device, same_query=False, remove_weight=False, name_of_instruction="Instruction", seed=42):
    target_texts, source_texts_1, source_texts_2, target_answer, source_answer_1, source_answer_2, texts_inverted = load_map_test_data(tokenizer, args, PATHS)
    random.seed(seed)
    
    source_index_1, target_index  = encode_answer(source_answer_1, target_answer, tokenizer, device, args)
    source_index_2, target_index  = encode_answer(source_answer_2, target_answer, tokenizer, device, args)
    # same_answer_indices = (source_index == target_index).detach().cpu()
    target_texts = [target_texts[i] for i in range(len(target_texts))]# if not same_answer_indices[i]]
    source_texts_1 = [source_texts_1[i] for i in range(len(source_texts_1))]# if not same_answer_indices[i]]
    source_texts_2 = [source_texts_2[i] for i in range(len(source_texts_2))]# if not same_answer_indices[i]]
    # source_index = source_index[~same_answer_indices]
    # target_index = target_index[~same_answer_indices]
    
    source_tokens_1 = tokenizer(source_texts_1, return_tensors="pt", padding=True)
    source_tokens_2 = tokenizer(source_texts_2, return_tensors="pt", padding=True)
    attention_mask_source_1 = source_tokens_1["attention_mask"].to(device)
    attention_mask_source_2 = source_tokens_2["attention_mask"].to(device)
    target_tokens = tokenizer(target_texts, return_tensors="pt", padding=True)
    attention_mask_target = target_tokens["attention_mask"].to(device)
    source_tokens_1 = source_tokens_1["input_ids"].to(device)
    source_tokens_2 = source_tokens_2["input_ids"].to(device)
    target_tokens = target_tokens["input_ids"].to(device)
    
    max_len = max(source_tokens_1.shape[1], source_tokens_2.shape[1], target_tokens.shape[1])
    source_tokens_1 = torch.nn.functional.pad(source_tokens_1, (max_len - source_tokens_1.shape[1], 0), value=tokenizer.pad_token_id)
    source_tokens_2 = torch.nn.functional.pad(source_tokens_2, (max_len - source_tokens_2.shape[1], 0), value=tokenizer.pad_token_id)
    target_tokens = torch.nn.functional.pad(target_tokens, (max_len - target_tokens.shape[1], 0), value=tokenizer.pad_token_id)
    attention_mask_source_1 = torch.nn.functional.pad(attention_mask_source_1, (max_len - attention_mask_source_1.shape[1], 0), value=0)
    attention_mask_source_2 = torch.nn.functional.pad(attention_mask_source_2, (max_len - attention_mask_source_2.shape[1], 0), value=0)
    attention_mask_target = torch.nn.functional.pad(attention_mask_target, (max_len - attention_mask_target.shape[1], 0), value=0)
    
    # texts_inverted = train_data["text_inverted"].tolist()
    texts_inverted = [texts_inverted[i] for i in range(len(texts_inverted))]# if not same_answer_indices[i]]
    target_inverted_tokens = tokenizer(texts_inverted, return_tensors="pt", padding=True)
    attention_mask_target_inverted = target_inverted_tokens["attention_mask"].to(device)
    target_inverted_tokens = target_inverted_tokens["input_ids"].to(device)
    return source_tokens_1, source_tokens_2, target_tokens, source_index_1, source_index_2, target_index, attention_mask_source_1, attention_mask_source_2, attention_mask_target, target_inverted_tokens, attention_mask_target_inverted

def filter_confident_samples(args, model, tt, st_1, st_2, ti, si_1, si_2, amt, ams_1, ams_2, batch_size=48):
    os.makedirs("cache", exist_ok=True)
    # print(f"Checking for cache/confident_indices_{args.model_id}.pt")
    # if os.path.exists(f"cache/confident_indices_{args.model_id}.pt"):
    #     return torch.load(f"cache/confident_indices_{args.model_id}.pt")

    # print(f"Checking for analysis/cache/confident_indices_{args.model_id}.pt")   
    # if os.path.exists(f"analysis/cache/confident_indices_{args.model_id}.pt"):
    #     return torch.load(f"analysis/cache/confident_indices_{args.model_id}.pt")
        
    confident_indices = []
    device = model.device
    for i in trange(0, tt.shape[0], batch_size):
        batch_end = min(i + batch_size, tt.shape[0])
        
        # Get batch data
        batch_tt = tt[i:batch_end].to(device)
        batch_st_1 = st_1[i:batch_end].to(device)
        batch_st_2 = st_2[i:batch_end].to(device)
        batch_ti = ti[i:batch_end].to(device)
        batch_si_1 = si_1[i:batch_end].to(device)
        batch_si_2 = si_2[i:batch_end].to(device)
        batch_amt = amt[i:batch_end].to(device)
        batch_ams_1 = ams_1[i:batch_end].to(device)
        batch_ams_2 = ams_2[i:batch_end].to(device)
        
        # Get model outputs
        with torch.no_grad():
            outputs_tt = model(batch_tt, attention_mask=batch_amt).logits
            outputs_st_1 = model(batch_st_1, attention_mask=batch_ams_1).logits
            outputs_st_2 = model(batch_st_2, attention_mask=batch_ams_2).logits
        
        # Get the indices of the highest logits
        max_logit_indices_tt = outputs_tt[:, -1, :].argmax(dim=-1)
        max_logit_indices_st_1 = outputs_st_1[:, -1, :].argmax(dim=-1)
        max_logit_indices_st_2 = outputs_st_2[:, -1, :].argmax(dim=-1)
        # Check if the highest logit indices match the target indices
        match_tt = (max_logit_indices_tt == batch_ti)
        match_st_1 = (max_logit_indices_st_1 == batch_si_1)
        match_st_2 = (max_logit_indices_st_2 == batch_si_2)
        
        # not_same_answer_in_gt = (batch_ti != batch_si)
        # Combine conditions and add to confident_indices
        confident_batch = torch.where(match_tt & match_st_1 & match_st_2)[0] + i
        confident_indices.extend(confident_batch.tolist())

    confident_indices = torch.tensor(confident_indices)
    print(f"Number of confident samples: {len(confident_indices)}/{tt.shape[0]}")
    print(f"Saving to cache/confident_indices_{args.model_id}.pt")
    torch.save(confident_indices, f"cache/confident_indices_{args.model_id}.pt")
    return confident_indices


def inputs_collator(inputs, device):
    for k, v in inputs.items():
        if "->" in k:
            inputs[k] = torch.tensor(v).to(device)
        elif "subspace" in k:
            inputs[k] = v
        elif v is not None:
            inputs[k] = torch.tensor(v).to(device)
    return inputs


def compute_loss(logits, labels):
   
    last_token_logits = logits[:, -1, :]
    loss_fct = CrossEntropyLoss()

    # Enable model parallelism
    labels = labels.to(logits.device)
    loss = loss_fct(last_token_logits, labels[:, 0])
    
    # print("Batch Peek:", tokenizer.decode(last_token_logits.argmax(dim=-1)[0]))
    # print("Batch Label:", tokenizer.decode(labels[0]))
    return loss


def compute_metrics(eval_preds, eval_labels, tokenizer=None, return_target_accuracy=True, verbose=False,):
    total_count = 0
    correct_count = 0
    alter_correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        len_before = len(eval_label)
        eval_pred = eval_pred[eval_label[:,0] != eval_label[:,1]]
        eval_label = eval_label[eval_label[:,0] != eval_label[:,1]]
        len_after = len(eval_label)
        if len_before != len_after:
            print(f"Filtered {len_before - len_after} samples due to equal first token")
        pred_test_labels = torch.argmax(eval_pred[:, -1, :], dim=-1)
        correct_labels = eval_label[:, 0] == pred_test_labels
        alter_correct_count += (eval_label[:, 1] == pred_test_labels).sum().tolist()
        if verbose:
            for b_idx in range(len(eval_pred)):
                print("Pred:", tokenizer.decode(pred_test_labels[b_idx].item()), " Labels (source):", tokenizer.decode(eval_label[b_idx][0].item()), " Labels (target):", tokenizer.decode(eval_label[b_idx][1].item()))
                print("Correct:", correct_labels[b_idx].item())
                print("Alter Correct:", (eval_label[b_idx][1] == pred_test_labels[b_idx]).item())
        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()
    accuracy = round(correct_count / total_count, 2)
    alter_accuracy = round(alter_correct_count / total_count, 2)
    if return_target_accuracy:
        if alter_accuracy + accuracy > 1:
            print(alter_correct_count, correct_count, total_count, accuracy, alter_accuracy)
            raise ValueError("Accuracy is greater than 1")
        return {"accuracy": accuracy, "target_accuracy": alter_accuracy}
    return {"accuracy": accuracy}


def print_random_sample(dataset, tokenizer, prefix=""):
    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]
    
    source_tokens = sample['source_input_ids']
    target_tokens = sample['input_ids']
    
    print(f"{prefix} Sample {idx}:")
    print("Source tokens:", tokenizer.decode(source_tokens))
    print("Target tokens:", tokenizer.decode(target_tokens))
    print("Target answer:", tokenizer.decode([sample['labels'][1]]))
    print("Source answer:", tokenizer.decode([sample['labels'][0]]))
    print()


def get_patch_scope_probs(nnmodel, tokenizer, source_tokens, target_tokens, source_attention_mask, target_attention_mask, source_answer, target_answer, site, batch_size=24, N_LAYERS=42, api=Gemma2, max_index=None):    
    if max_index is not None:
        target_tokens = target_tokens[:max_index]
        source_tokens = source_tokens[:max_index]
        source_attention_mask = source_attention_mask[:max_index]
        target_attention_mask = target_attention_mask[:max_index]
        source_answer = source_answer[:max_index]
        target_answer = target_answer[:max_index]
    residuals = []
            # Step 1: Find upper bound
    for i in range(0, target_tokens.shape[0], batch_size):
        site.reset()
        residuals_batch, _ = get_patched_residuals(nnmodel, site, source_tokens[i:i+batch_size], target_tokens[i:i+batch_size], source_attention_mask[i:i+batch_size], target_attention_mask[i:i+batch_size], scan=False, validate=False)
        residuals.append(residuals_batch)

    residuals = torch.cat(residuals, dim=1)

    logits = patch_scope(nnmodel, tokenizer, residuals, verbose=False)
    p_mean_source, p_std_source, p_median_source = get_probs(logits, source_answer)
    p_mean_target, p_std_target, p_median_target = get_probs(logits, target_answer)
    return p_mean_source, p_mean_target
    
def auto_search(model, tokenizer, patching_arguments, n_layers=42, eps=0.3, thres=0.85, phi=0.05, batch_size=20, api=Gemma2, max_index=None, lower_bound=None, upper_bound=None):
    _, _, source_tokens, target_tokens, source_attention_mask, target_attention_mask, source_answer, target_answer = patching_arguments

    nnmodel = NNsight(model)
    
    if upper_bound is None:
        print("Step 1: Searching for upper bound...")
        ## Adding layers can only increase the max probability -> binary search
        left, right = 0, n_layers - 1
        while left <= right:
            upper_bound = (left + right) // 2
            print(f"Trying upper bound: {upper_bound}")
            site_config = {
                "o":
                {
                    "layers": list(range(0, upper_bound)),
                },
            }
            site = config_to_site(site_config, api=api, model=model)
            p_mean_source, p_mean_target = get_patch_scope_probs(nnmodel, tokenizer, source_tokens, target_tokens, source_attention_mask, target_attention_mask, source_answer, target_answer, site, batch_size=batch_size, max_index=max_index)
            print(f"Upper bound: {upper_bound} - Max probability: {p_mean_source.max().item():.4f}, Probability at layer {upper_bound}: {p_mean_source[upper_bound, 0].item():.4f}")
            if p_mean_source.max().item() > thres:
                right = upper_bound - 1
            else:
                left = upper_bound + 1

        if p_mean_source.max().item() <= thres:
            upper_bound = upper_bound + 1
    else:
        site_config = {
            "o":
            {
                "layers": list(range(0, upper_bound)),
        },
        }
        site = config_to_site(site_config, api=api, model=model)
        p_mean_source, p_mean_target = get_patch_scope_probs(nnmodel, tokenizer, source_tokens, target_tokens, source_attention_mask, target_attention_mask, source_answer, target_answer, site, batch_size=batch_size, max_index=max_index)
        print(f"Upper bound: {upper_bound} - Max probability: {p_mean_source.max().item():.4f}, Probability at layer {upper_bound-1}: {p_mean_source[upper_bound-1, 0].item():.4f}")

    print(f"Upper bound: {upper_bound}")

    if lower_bound is None:
        print("Step 2: Searching for lower bound...")
        ## Removing layers from left can only decrease the max probability -> binary search
        left, right = 0, upper_bound
        while left <= right:
            lower_bound = (left + right) // 2
            site_config = {
                "o":
                {
                    "layers": list(range(lower_bound, upper_bound)),
                },
            }
            site = config_to_site(site_config, api=api, model=model)
            p_mean_source, p_mean_target = get_patch_scope_probs(nnmodel, tokenizer, source_tokens, target_tokens, source_attention_mask, target_attention_mask, source_answer, target_answer, site, batch_size=batch_size, max_index=max_index)
            print(f"Lower bound: {lower_bound} - Max probability: {p_mean_source.max().item():.4f}, Probability at layer {lower_bound}: {p_mean_source[lower_bound, 0].item():.4f}, Probability at last layer: {p_mean_source[-1, 0].item():.4f}")
            if p_mean_source.max().item() <= thres:
                right = lower_bound - 1
            else:
                left = lower_bound + 1
        if p_mean_source.max().item() <= thres:
            lower_bound = lower_bound - 1
    else:
        site_config = {
            "o":
            {
                "layers": list(range(lower_bound, upper_bound)),
            },
        }
        site = config_to_site(site_config, api=api, model=model)
        p_mean_source, p_mean_target = get_patch_scope_probs(nnmodel, tokenizer, source_tokens, target_tokens, source_attention_mask, target_attention_mask, source_answer, target_answer, site, batch_size=batch_size, max_index=max_index)
        print(f"Lower bound: {lower_bound} - Max probability: {p_mean_source.max().item():.4f}, Probability at layer {lower_bound}: {p_mean_source[lower_bound, 0].item():.4f}, Probability at last layer: {p_mean_source[-1, 0].item():.4f}")
    print(f"Lower bound: {lower_bound}")

    base_range = list(range(lower_bound, upper_bound))
    # Refine step
    # Test whether refinement is needed

    print("Step 3: Refining...")
    while p_mean_source[-1, 0].item() <= p_mean_target[-1, 0].item() + eps:
        print(f"Refining (Probability source: {p_mean_source[-1, 0].item():.4f}, Probability target: {p_mean_target[-1, 0].item():.4f}, Difference: {p_mean_source[-1, 0].item() - p_mean_target[-1, 0].item():.4f} (eps={eps}))...")
        print("Current base range: ", base_range)
        for candidate in range(max(base_range), n_layers):
            if p_mean_source[candidate-1, 0].item() - p_mean_source[candidate, 0].item() > phi and candidate not in base_range:
                break
        print(f"Refined base range: {base_range + [candidate]}")
        base_range.append(candidate)
        site_config = {
            "o":
            {
                 "layers": base_range,
            },
        }
        site = config_to_site(site_config, api=api, model=model)
        p_mean_source, p_mean_target = get_patch_scope_probs(nnmodel, tokenizer, source_tokens, target_tokens, source_attention_mask, target_attention_mask, source_answer, target_answer, site, batch_size=batch_size, max_index=max_index)
        print(f"Out probability source: {p_mean_source[-1, 0].item():.4f}, Out probability target: {p_mean_target[-1, 0].item():.4f}, difference: {p_mean_source[-1, 0].item() - p_mean_target[-1, 0].item():.4f} (eps={eps})")

    print("No more refinement needed")
    return base_range


def convert_pyvene_to_nnpatch(state_dict):
    out = OrderedDict()
    out["rank"] = torch.tensor(1)
    for k, v in state_dict.items():
        if k.startswith("rotate"):
            out[k.replace("rotate_layer.", "")] = v
        elif "interchange_dim" in k:
            continue
        else:
            out[k] = v
    return out
