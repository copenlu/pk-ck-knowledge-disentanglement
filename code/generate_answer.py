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


MODE_TO_INSTRUCTION = {
    "prior": "Ignore the context in answering the query.",
    "context": "Only consider the context in answering the query.",
    "both": "Answer the following query considering the provided context."
}

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


def answer_generation(args, model, tokenizer, dataset):
        all_pred_logit = []
        all_pred_prob = []
        all_pred_hs = []
        all_input = []
        for i in tqdm(range(len(dataset))):
            line = dataset.iloc[i]

            if args.dataset == 'strategyqa':
                system_prompt = "Answer the following query considering the provided context. Answer in either of these two: True, or False."
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": """Context: %s\nInstruction: %s\nQuery: %s\n""" % (line["context"].strip(), MODE_TO_INSTRUCTION[args.mode].strip(), line["query"].strip())}
                    ]

            elif args.dataset == 'openbookqa-complementary':
                system_prompt = "Answer the following query considering the provided context. Answer with only one word."
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": """Context: %s\nInstruction: %s\nQuery: %s\n""" % (line["context"].strip(), MODE_TO_INSTRUCTION[args.mode].strip(), line["query"].strip())}
                    ]
            input_msgs = msgs
            encoded = tokenizer.apply_chat_template(input_msgs, return_tensors="pt").to(args.device)
            outputs  = model.generate(encoded, max_new_tokens=10, do_sample=False, return_dict_in_generate=True, output_scores=True, output_hidden_states=True)

            # New tokens only (exclude the prompt)
            sequences = outputs.sequences  # shape: [batch, prompt+new]
            new_token_ids = sequences[:, encoded.shape[1]:]  # [1, T_new]
            step_scores = outputs.scores  # list length T_new, each [batch, vocab]

            # Decode text if you want to keep it for debugging/inspection
            decoded = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0]
            # Iterate step-by-step to align each generated token with its score vector
            if args.dataset == 'strategyqa':
                # ---- Parse prediction token + collect its logit/prob ----
                # We scan generated tokens in order and pick the first containing "true" or "false".
                pred_label = "unknown"
                pred_logit = float("-inf")   # pre-softmax logit
                pred_prob = 0.0              # optional: post-softmax
                ans_step   = None

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

                if pred_label == "unknown":
                    # Fall back to the first token’s score (or leave as unknown with -inf)
                    # Here we keep unknown with logit -inf and prob 0.0 to make it explicit.
                    pass

                last_vec, _ = get_answer_hidden(outputs, ans_step)  # may be None if unknown
            
            elif args.dataset == 'openbookqa-complementary':
                choices = parse_choices_from_query(line["query"])
                # Decide prediction and grab logit/prob of the decisive token
                pred_label, pred_text, pred_logit, pred_prob, ans_step = decide_from_generated_mc(
                    new_token_ids, step_scores, tokenizer, choices
                )
                last_vec, _ = get_answer_hidden(outputs, ans_step)
                print(decoded, pred_label, pred_text, pred_logit, pred_prob, last_vec.shape, '\n')
            
            all_pred_logit.append(pred_logit)
            all_pred_prob.append(pred_prob)
            all_pred_hs.append([] if last_vec is None else last_vec.tolist())
            all_input.append({'context': line['context'], 'query': line['query'], 'label': str(line['label']), 'pred': pred_label})

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
        with open('../data/strategyqa/train.json', 'r') as fp:
            examples_train = json.load(fp)
        with open('../data/strategyqa/dev.json', 'r') as fp:
            examples_dev = json.load(fp)
        examples = examples_train + examples_dev
        
        context_list, query_list, label_list = [], [], []
        for example in examples:
            context_list.append(' '.join(example['facts']))
            query_list.append(example['question'])
            label_list.append(example['answer'])
        df_filtered = pd.DataFrame({'context': context_list, 'query': query_list, 'label': label_list})
        unique_df = df_filtered.drop_duplicates()
        dataset = unique_df

        print('Length of strategyqa dataset: ', len(dataset))
        print('Labels: ', set(dataset['label'].tolist()))

    elif args.dataset == "politihop":
        dataset = pd.read_csv('../data/politihop/politihop_unique.csv', sep = '\t')
        print('Length of politihop dataset: ', len(dataset))
        print('Labels: ', set(dataset['label'].tolist()))

    elif args.dataset == "musique":
        with open('./Knowledge_Interplay/echoqa_dataset/LLAMA_3_8B/MuSiQue/complementary.json', 'r') as fp:
            examples_dev = json.load(fp)
        examples = examples_dev

        context_list, q_list, label_list = [], [], []
        for example in examples:
            contexts = example['ck']
            try:
                context_str = ' '.join(contexts)
            except:
                context_str = ' '.join(contexts[0])
            context_list.append(context_str)
            q_list.append(example['question'])
            label_list.append(example['answer'])
        df_filtered = pd.DataFrame({'context': context_list, 'input': q_list, 'label': label_list})
        unique_df = df_filtered.drop_duplicates()
        dataset = unique_df

        print('Length of MuSiWue dataset: ', len(dataset))
        #print('Labels: ', set(dataset['label'].tolist()))

    elif args.dataset == "openbookqa-complementary":
        with open('../data/echoqa_dataset/LLAMA_3_8B/OpenbookQA/complementary.json', 'r') as fp:
            examples_dev = json.load(fp)
        examples = examples_dev

        context_list, query_list, label_list = [], [], []
        for example in examples:
            contexts = example['ck']
            try:
                context_str = ' '.join(contexts)
            except:
                context_str = ' '.join(contexts[0])
            context_list.append(context_str)
            query_list.append(example['question'])
            label_list.append(example['answer'])
        df_filtered = pd.DataFrame({'context': context_list, 'query': query_list, 'label': label_list})
        unique_df = df_filtered.drop_duplicates()
        dataset = unique_df

        print('Length of OpenBookQA Complementary dataset: ', len(dataset))

    return dataset


def main(args):

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args)

    all_pred_logit, all_pred_prob, all_pred_hs, all_input = answer_generation(args, model, tokenizer, dataset)
    exit(0)

    #Write results
    model_name = args.model_name.strip().split('/')[-1]
    os.makedirs(f"{args.data_dir}{args.dataset}/{model_name}", exist_ok=True)
    os.makedirs(f"{args.output_dir}{args.dataset}/{model_name}", exist_ok=True)
    
    with open(f"{args.output_dir}{args.dataset}/{model_name}/all_pred_logit_{args.mode}.pkl", 'wb') as f:
        pickle.dump(all_pred_logit, f)
    with open(f"{args.output_dir}{args.dataset}/{model_name}/all_pred_prob_{args.mode}.pkl", 'wb') as f:
        pickle.dump(all_pred_prob, f)
    with open(f"{args.output_dir}{args.dataset}/{model_name}/all_pred_hs_{args.mode}.pkl", 'wb') as f:
        pickle.dump(all_pred_hs, f)
    with open(f"{args.data_dir}{args.dataset}/{model_name}/all_input_{args.mode}.json", 'w') as f:
        json.dump(all_input, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # model : meta-llama/Llama-Guard-3-8B ,
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--dataset", type=str, default="strategyqa")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--mode", type=str, help = "prior/context/both", default="both") # prior only / context only / both
    parser.add_argument("--output_dir", type=str, default="../results/")
    args = parser.parse_args()
    print(args)

    set_seed(args)

    main(args)
