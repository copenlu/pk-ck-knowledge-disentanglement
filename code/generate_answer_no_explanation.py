import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import itertools
import torch
#import hf_olmo
import random
import numpy as np
import time
import pickle
import pandas as pd
import ast
import re
from difflib import SequenceMatcher
import unicodedata


PROMPT_MODE_TO_INSTRUCTION = {
    "prior_only": "Ignore the context in answering the query.",
    "context_only": "Only consider the context in answering the query.",
    "both_w_instruction": "Consider the context in answering the query."
}

# Numbered choices like "1. foo", "2) bar"
NUM_CHOICE_LINE_RE = re.compile(r'^\s*(\d+)\s*[\.\):\-]\s*(.+?)\s*$', re.UNICODE)
CHOICE_LINE_RE = re.compile(r'^\s*([A-Ea-e])\s*[\.\):\-]\s*(.+?)\s*$', re.UNICODE)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']*")
LETTER_FULL = re.compile(r'(?i)^[abcde](?:[.)])?$')  # A, A., A) etc.

def get_answer_hidden(outputs, step_idx, *, layer=-1, encoder_decoder=False):
    """
    Returns (last_layer_vec, all_layers_vecs) for the generated token at `step_idx`.
    Shapes: last_layer_vec [hidden_size], all_layers_vecs: list of [hidden_size] per layer.
    """
    # In HF, generate() collects decoder_hidden_states for enc-dec, hidden_states for decoder-only.
    hs_time = getattr(outputs, "decoder_hidden_states", None) if encoder_decoder \
              else getattr(outputs, "hidden_states", None)
    if hs_time is None or step_idx is None:
        return None, None

    hs_layers = hs_time[step_idx]  # tuple over layers (emb + L layers) or a Tensor
    if isinstance(hs_layers, tuple):
        last_layer_tensor = hs_layers[-1]                 # [B, seq_len(=1), H]
        last_vec = last_layer_tensor[0, -1, :].detach().cpu().numpy()
        all_layers = [h[0, -1, :].detach().cpu().numpy() for h in hs_layers]
    else:
        # Some models may return only last hidden state tensor per step
        last_vec = hs_layers[0, -1, :].detach().cpu().numpy()
        all_layers = None
    return last_vec, all_layers

def parse_choices_from_query(query_text: str):
    """
    Extract choices dict like {'A': 'answer_1', 'B': 'answer_2', 'C': 'Unknown'}
    from lines like: 'A. answer_1'
    """
    choices = {}
    for line in query_text.splitlines():
        m = CHOICE_LINE_RE.match(line)
        if m:
            letter = m.group(1).upper()
            text = m.group(2).strip()
            choices[letter] = text
    return choices

def parse_numbered_choices_from_query(query_text: str):
    """Extract {'1': '...', '2': '...'} from lines like '1. ...'."""
    choices = {}
    for line in query_text.splitlines():
        m = NUM_CHOICE_LINE_RE.match(line)
        if m:
            key = m.group(1)           # '1', '2', ...
            text = m.group(2).strip()
            choices[key] = text
    return choices

def make_choice_full_regex(keys):
    """
    Build a regex that matches exactly one of the keys (e.g., '1' or '2'),
    optionally followed by '.' or ')'.
    """
    alt = "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))
    return re.compile(rf'(?i)^(?P<key>(?:{alt}))(?:[.)])?$', re.UNICODE)

def first_visible_token_text(tokenizer, tok):
    """
    Convert a single token to display text (removes BPE markers).
    """
    return tokenizer.convert_tokens_to_string([tok])  # 1-token string

def first_content_step(new_token_ids, outputs_scores, tokenizer):
    toks = [tokenizer.convert_ids_to_tokens(int(t)) for t in new_token_ids[0]]

    # If a role header leaked into the generation, skip it
    start = 0
    if '<|end_header_id|>' in toks:
        start = toks.index('<|end_header_id|>') + 1

    # Skip special tokens and pure whitespace/newlines (e.g. 'Ċ', 'ĊĊ')
    specials = set(getattr(tokenizer, "all_special_tokens", []))
    i = start
    while i < len(toks):
        t = toks[i]
        if (t in specials) or (tokenizer.convert_tokens_to_string([t]).strip() == ""):
            i += 1
        else:
            break

    # Stop before end-of-turn if present
    end = toks.index('<|eot_id|>') if '<|eot_id|>' in toks else len(toks)
    return i, end, toks

def decide_from_generated_mc(new_token_ids, step_scores, tokenizer, choices):
    """
    Decide A/B/C (or Unknown) from generated tokens and return:
    (letter, choice_text, logit, prob)
    The logit/prob are for the *decisive* token (the one that made us choose).
    """
    start, end, toks = first_content_step(new_token_ids, step_scores, tokenizer)
    decoded_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in new_token_ids[0]]
    # 1) Direct letter (A/B/C)
    for step in range(start, end):
        tok_id = int(new_token_ids[0, step].item())
        vis = tokenizer.convert_tokens_to_string([toks[step]]).strip()
        if LETTER_FULL.fullmatch(vis):
            letter = vis[0].upper()
            if letter in choices:
                score_vec = step_scores[step][0]            # [vocab]
                logit = score_vec[tok_id].item()               # pre-softmax
                prob  = torch.softmax(score_vec, -1)[tok_id].item()
                return letter, choices[letter], logit, prob, step
        if "unknown" in vis.lower():
            letter = next((L for L,t in choices.items() if t.lower()=="unknown"), "C")
            score_vec = step_scores[step][0]
            tok_id = int(new_token_ids[0, step].item())
            logit = score_vec[tok_id].item()
            prob  = torch.softmax(score_vec, -1)[tok_id].item()
            return letter, choices.get(letter, "Unknown"), logit, prob, step

    # 2) Word answer mapping: map first content word to closest option text
    decoded_str = tokenizer.decode(new_token_ids[0, start:end], skip_special_tokens=True)
    wm = WORD_RE.search(decoded_str)
    if wm:
        first_word = wm.group(0).lower()
        # pick the best matching option by containment / similarity
        best_letter, best_score = None, 0.0
        for L, txt in choices.items():
            low = txt.lower()
            score = 1.0 if first_word in low else SequenceMatcher(None, first_word, low).ratio()
            if score > best_score:
                best_score, best_letter = score, L

        # take the logit/prob of the token that produced the first *word* char
        for step, tok_id in enumerate(new_token_ids[0].tolist()):
            tok = decoded_tokens[step]
            vis = first_visible_token_text(tokenizer, tok)
            if WORD_RE.search(vis):
                score_vec = step_scores[step][0]
                logit = score_vec[tok_id].item()
                prob = torch.softmax(score_vec, dim=-1)[tok_id].item()
                return best_letter or "C", choices.get(best_letter, ""), logit, prob, step

    # 3) Fallback: mark unknown; still provide numbers from the first step
    tok_id0 = int(new_token_ids[0, start].item())
    score_vec0 = step_scores[start][0]          # <-- align with `start`
    logit0 = score_vec0[tok_id0].item()
    prob0 = torch.softmax(score_vec0, -1)[tok_id0].item()
    return "unknown", "", logit0, prob0, start


# def decide_from_generated_mc_generic(new_token_ids, step_scores, tokenizer, choices, key_full_regex):
#     """
#     choices: dict like {'1': 'text', '2': 'text'} or {'A':'...'}
#     key_full_regex: compiled regex that full-matches a key token (e.g., '1', '1.', 'A', 'A)').
#     Returns: (key, choice_text, logit, prob, decisive_step_index)
#     """
#     start, end, toks = first_content_step(new_token_ids, step_scores, tokenizer)

#     # 1) Direct key detection (e.g., '1' / '1.' / 'A' / 'A)')
#     for step in range(start, end):
#         tok_id = int(new_token_ids[0, step].item())
#         vis = tokenizer.convert_tokens_to_string([toks[step]]).strip()
#         m = key_full_regex.fullmatch(vis)
#         if m:
#             key = m.group("key")
#             if key in choices:
#                 score_vec = step_scores[step][0]
#                 logit = score_vec[tok_id].item()
#                 prob  = torch.softmax(score_vec, -1)[tok_id].item()
#                 return key, choices[key], logit, prob, step

#         # Optional: catch 'unknown'
#         if "unknown" in vis.lower():
#             # Prefer a key whose text is exactly 'unknown' if present
#             key = next((K for K, t in choices.items() if t.lower() == "unknown"), None)
#             if key is None:
#                 continue
#             score_vec = step_scores[step][0]
#             logit = score_vec[tok_id].item()
#             prob  = torch.softmax(score_vec, -1)[tok_id].item()
#             return key, choices[key], logit, prob, step

#     # 2) Fallback: map first content word to closest option text (same logic you use now)
#     decoded_str = tokenizer.decode(new_token_ids[0, start:end], skip_special_tokens=True)
#     wm = WORD_RE.search(decoded_str)
#     if wm:
#         first_word = wm.group(0).lower()
#         best_key, best_score = None, 0.0
#         for K, txt in choices.items():
#             low = txt.lower()
#             score = 1.0 if first_word in low else SequenceMatcher(None, first_word, low).ratio()
#             if score > best_score:
#                 best_score, best_key = score, K

#         # take numbers for the first word-like token
#         for step in range(start, end):
#             tok_id = int(new_token_ids[0, step].item())
#             vis = tokenizer.convert_tokens_to_string([toks[step]])
#             if WORD_RE.search(vis):
#                 score_vec = step_scores[step][0]
#                 logit = score_vec[tok_id].item()
#                 prob  = torch.softmax(score_vec, dim=-1)[tok_id].item()
#                 return best_key, choices.get(best_key, ""), logit, prob, step

#     # 3) Fallback: keep metrics defined
#     tok_id0 = int(new_token_ids[0, start].item())
#     score_vec0 = step_scores[start][0]
#     logit0 = score_vec0[tok_id0].item()
#     prob0 = torch.softmax(score_vec0, -1)[tok_id0].item()
#     return "unknown", "", logit0, prob0, start


def rstrip_punct(s: str) -> str:
    """Remove any trailing punctuation/symbols, keep inner hyphens/apostrophes."""
    i = len(s)
    while i > 0 and unicodedata.category(s[i-1])[0] in ("P", "S"):
        i -= 1
    return s[:i]


def decide_first_word(new_token_ids, step_scores, tokenizer):
    start, end, toks = first_content_step(new_token_ids, step_scores, tokenizer)

    decoded_span = tokenizer.decode(new_token_ids[0, start:end], skip_special_tokens=True)
    m = WORD_RE.search(decoded_span)
    pred_text = m.group(0) if m else ""
    pred_text = rstrip_punct(pred_text)   # <-- strip trailing punctuation

    ans_step = None
    pred_logit, pred_prob = float("-inf"), 0.0
    for step in range(start, end):
        tok_id = int(new_token_ids[0, step].item())
        vis = tokenizer.convert_tokens_to_string([toks[step]])
        if WORD_RE.search(vis):
            score_vec = step_scores[step][0]
            pred_logit = score_vec[tok_id].item()
            pred_prob  = torch.softmax(score_vec, dim=-1)[tok_id].item()
            ans_step   = step
            break

    if ans_step is None and start < end:
        tok_id0 = int(new_token_ids[0, start].item())
        score_vec0 = step_scores[start][0]
        pred_logit = score_vec0[tok_id0].item()
        pred_prob  = torch.softmax(score_vec0, dim=-1)[tok_id0].item()
        ans_step   = start

    return pred_text, pred_logit, pred_prob, ans_step


def answer_generation(args, model, tokenizer, dataset):
        all_pred_logit = []
        all_pred_prob = []
        all_pred_hs = []
        all_input = []

        for i in tqdm(range(len(dataset))):
            line = dataset.iloc[i]
            context = line["context"]
            query = line["query"]

            if args.dataset == 'strategyqa':
                system_prompt = "Answer the following query considering the provided context. Answer in either of these two: True or False. If you are unable to answer the query, respond Unknown."
                if args.prompt_mode == "prior_wo_context":
                    user_prompt = """Query: %s\n""" % (query.strip())
                    system_prompt = "Answer the following query. Answer in either of these two: True or False. If you are unable to answer the query, respond Unknown."
                elif args.prompt_mode == "prior_only" or args.prompt_mode == "context_only" or args.prompt_mode == "both_w_instruction":
                    user_prompt = """Context: %s\nInstruction: %s\nQuery: %s\n""" % (context.strip(), PROMPT_MODE_TO_INSTRUCTION[args.prompt_mode].strip(), query.strip())
                elif args.prompt_mode == 'both_wo_instruction':
                    user_prompt = """Context: %s\nQuery: %s\n""" % (context.strip(), query.strip())

            else:
                system_prompt = "Answer the following query considering the provided context. Answer with only one word."
                if args.prompt_mode == "prior_wo_context":
                    user_prompt = """Query: %s\n""" % (query.strip())
                    system_prompt = "Answer the following query. Answer with only one word."
                elif args.prompt_mode == "prior_only" or args.prompt_mode == "context_only" or args.prompt_mode == "both_w_instruction":
                    user_prompt = """Context: %s\nInstruction: %s\nQuery: %s\n""" % (context.strip(), PROMPT_MODE_TO_INSTRUCTION[args.prompt_mode].strip(), query.strip())
                elif args.prompt_mode == 'both_wo_instruction':
                    user_prompt = """Context: %s\nQuery: %s\n""" % (context.strip(), query.strip())

            if 'gemma' in args.model_name:
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

            tokens = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
            breakpoint()
            attn_mask = torch.ones_like(tokens)

            outputs  = model.generate(tokens, attention_mask=attn_mask, max_new_tokens=30, do_sample=False, temperature=None, top_p=None, return_dict_in_generate=True, output_scores=True, output_hidden_states=True)

            # New tokens only (exclude the prompt)
            sequences = outputs.sequences  # shape: [batch, prompt+new]
            new_token_ids = sequences[:, tokens.shape[1]:]  # [1, T_new]
            step_scores = outputs.scores  # list length T_new, each [batch, vocab]

            # Decode text if you want to keep it for debugging/inspection
            decoded = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0]
            # Iterate step-by-step to align each generated token with its score vector
            if args.dataset == 'strategyqa':
                # ---- Parse prediction token + collect its logit/prob ----
                # We scan generated tokens in order and pick the first containing "true" or "false".
                pred_label = "Unknown"

                for step, tok_id in enumerate(new_token_ids[0].tolist()):
                    token_str = tokenizer.convert_ids_to_tokens(tok_id)
                    # scores for this step: shape [vocab], for batch item 0
                    score_vec = step_scores[step][0]  # tensor [vocab]
                    # logit/probability of the token that was actually chosen
                    this_logit = score_vec[tok_id].item()
                    this_prob = torch.softmax(score_vec, dim=-1)[tok_id].item()

                    t = token_str.lower()
                    if ("true" in t) or ("false" in t):
                        pred_label = "True" if "true" in t else "False"
                        pred_logit = this_logit
                        pred_prob = this_prob
                        ans_step = step
                        break

                if pred_label == "Unknown":
                    start, end, toks = first_content_step(new_token_ids, step_scores, tokenizer)
                    tok_id0 = int(new_token_ids[0, start].item())
                    score_vec0 = step_scores[start][0]          # <-- align with `start`
                    pred_logit = score_vec0[tok_id0].item()
                    pred_prob = torch.softmax(score_vec0, -1)[tok_id0].item()
                    ans_step = start
                    

                last_vec, _ = get_answer_hidden(outputs, ans_step)  # may be None if unknown

            elif args.dataset == 'basefakepedia' or args.dataset == 'multihopfakepedia':
                pred_text, pred_logit, pred_prob, ans_step = decide_first_word(
                    new_token_ids, step_scores, tokenizer
                )
                last_vec, _ = get_answer_hidden(outputs, ans_step)
                pred_label = rstrip_punct(pred_text)  # redundant but safe

            
            elif args.dataset == 'openbookqa' or args.dataset == 'musique':
                choices = parse_choices_from_query(line["query"])
                # Decide prediction and grab logit/prob of the decisive token
                pred_label, pred_text, pred_logit, pred_prob, ans_step = decide_from_generated_mc(
                    new_token_ids, step_scores, tokenizer, choices
                )
                last_vec, _ = get_answer_hidden(outputs, ans_step)

            # elif args.dataset == 'musique':
            #     choices = parse_numbered_choices_from_query(query)        # {'1': '...', '2': '...'}
            #     key_regex = make_choice_full_regex(choices.keys())        # matches '1', '1.', '2', '2)'
            #     # Decide prediction and grab logit/prob of the decisive token
            #     pred_key, pred_text, pred_logit, pred_prob, ans_step = decide_from_generated_mc_generic(
            #         new_token_ids, step_scores, tokenizer, choices, key_regex
            #     )
            #     last_vec, _ = get_answer_hidden(outputs, ans_step)
            #     pred_label = pred_key               # e.g., '1' (you could store pred_text too)
            
            all_pred_logit.append(pred_logit)
            all_pred_prob.append(pred_prob)
            all_pred_hs.append([] if last_vec is None else last_vec.tolist())
            all_input.append({'context': line['context'], 'query': line['query'], 'label': str(line['label']), 'ctx_label': str(line['ctx_label']), 'prior_label': str(line['prior_label']), f'{args.prompt_mode}_ans': pred_label})

        return all_pred_logit, all_pred_prob, all_pred_hs, all_input


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    else:
        print('No GPU available, using the CPU instead.')


def load_dataset(args):
    dataset = None
    if args.dataset == "strategyqa":
        unique_df = pd.read_csv(f"../data/strategyqa/strategyqa_qa_data.csv")
        dataset = unique_df
        print('Length of strategyqa dataset: ', len(dataset))
        print('Labels: ', set(dataset['label'].tolist()))

    elif args.dataset == "basefakepedia":
        unique_df = pd.read_csv(f"../data/BaseFakepedia/basefakepedia_qa_data.csv")
        dataset = unique_df
        print('Length of BaseFakepedia dataset: ', len(dataset))

    elif args.dataset == "multihopfakepedia":
        unique_df = pd.read_csv(f"../data/MultihopFakepedia/multihopfakepedia_qa_data.csv")
        dataset = unique_df
        print('Length of MultihopFakepedia dataset: ', len(dataset))

    elif args.dataset == "musique":
        unique_df = pd.read_csv(f"../data/echoqa_dataset/musique_data.csv")
        dataset = unique_df
        print('Length of MuSiQue dataset: ', len(dataset))

    elif args.dataset == "openbookqa":
        unique_df = pd.read_csv(f"../data/echoqa_dataset/openbookqa_data.csv")
        dataset = unique_df
        print('Length of OpenBookQA dataset: ', len(dataset))

    return dataset


def main(args):

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args)

    all_pred_logit, all_pred_prob, all_pred_hs, all_input = answer_generation(args, model, tokenizer, dataset)

    #Write results
    model_name = args.model_name.strip().split('/')[-1]
    os.makedirs(f"{args.data_dir}{args.dataset}/{model_name}", exist_ok=True)
    os.makedirs(f"{args.output_dir}{args.dataset}/{model_name}", exist_ok=True)
    
    with open(f"{args.output_dir}{args.dataset}/{model_name}/all_pred_logit_{args.prompt_mode}_no_explanation.pkl", 'wb') as f:
        pickle.dump(all_pred_logit, f)
    with open(f"{args.output_dir}{args.dataset}/{model_name}/all_pred_prob_{args.prompt_mode}_no_explanation.pkl", 'wb') as f:
        pickle.dump(all_pred_prob, f)
    with open(f"{args.output_dir}{args.dataset}/{model_name}/all_pred_hs_{args.prompt_mode}_no_explanation.pkl", 'wb') as f:
        pickle.dump(all_pred_hs, f)
    with open(f"{args.data_dir}{args.dataset}/{model_name}/all_input_{args.prompt_mode}_no_explanation.json", 'w') as f:
        json.dump(all_input, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct") # meta-llama/Meta-Llama-3.1-8B-Instruct / google/gemma-2-9b-it / mistralai/Mistral-7B-Instruct-v0.3
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--dataset", type=str, default="strategyqa") # strategyqa / basefakepedia / multihopfakepedia / openbookqa / musique
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--prompt_mode", type=str, help = "prior_wo_context/prior_only/context_only/both_w_instruction/both_wo_instruction", default="both_w_instruction")
    parser.add_argument("--output_dir", type=str, default="../results/")
    args = parser.parse_args()
    print(args)

    set_seed(args)

    main(args)
