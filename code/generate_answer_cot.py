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


# PROMPT_MODE_TO_INSTRUCTION = {
#     "prior_only": "Ignore the context in answering the query.",
#     "context_only": "Only consider the context in answering the query.",
#     "both_w_instruction": "Consider the context in answering the query."
# }

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


def find_first_json_object(text: str):
    """Return (json_substring, start_idx, end_idx) for the first complete {...} found in text.
    Handles quotes/escapes so braces inside strings don't break the scan.
    end_idx is exclusive."""
    i = text.find('{')
    if i == -1:
        return None, -1, -1
    depth = 0
    in_str = False
    esc = False
    start = i
    for j in range(i, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:j+1], start, j+1
    return None, -1, -1

def locate_json_string_value_span(json_text: str, key: str):
    """Given a JSON object string, return (inner_start, inner_end) character
    indices (no quotes) of the string value for 'key'. Returns (None, None) if not found.
    Assumes well-formed JSON for that field."""
    m = re.search(rf'"{re.escape(key)}"\s*:\s*"', json_text)
    if not m:
        return None, None
    i = m.end() - 1  # position of opening quote
    # Scan JSON string handling escapes
    j = i + 1
    esc = False
    while j < len(json_text):
        ch = json_text[j]
        if esc:
            esc = False
        elif ch == '\\':
            esc = True
        elif ch == '"':
            # return inner span (exclude quotes)
            return i + 1, j
        j += 1
    return None, None

def token_texts_from_ids(tokenizer, token_ids_1d):
    """Return (token_strs, piece_strs, joined_text) for a 1D list/array of ids.
    piece_str is the visible text produced by each token."""
    toks = [tokenizer.convert_ids_to_tokens(int(t)) for t in token_ids_1d]
    pieces = [tokenizer.convert_tokens_to_string([t]) for t in toks]
    joined = ''.join(pieces)
    return toks, pieces, joined

def build_char_spans(pieces):
    """Given per-token visible pieces, return a list of (start,end) char indices per token."""
    spans = []
    pos = 0
    for p in pieces:
        start = pos
        end = pos + len(p)
        spans.append((start, end))
        pos = end
    return spans

def collect_token_stats(step, new_token_ids, step_scores, outputs, piece_text):
    """Return a dict with token stats for a given generated step index."""
    tok_id = int(new_token_ids[0, step].item())
    score_vec = step_scores[step][0]  # [vocab]
    logit = score_vec[tok_id].item()
    prob = torch.softmax(score_vec, dim=-1)[tok_id].item()
    hs, _ = get_answer_hidden(outputs, step)
    return {
        "step": step,
        "token_id": tok_id,
        "piece": piece_text,
        "logit": logit,
        "prob": prob,
        "hidden": None if hs is None else hs.tolist(),
    }


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
                system_prompt = """Answer the following query considering the provided context. Generate your final answer in either of these two: True or False. If you are unable to answer the query, generate your final answer as "Unknown". Also, generate an explanation to determine your final answer. Return your output in JSON format: { "explanation": "your explanation here", "answer": "your final response here" }. Only include the JSON object in your response."""

            else:
                system_prompt = """Answer the following query considering the provided context. Generate your final answer with only one word. If you are unable to answer the query, generate your final answer as "Unknown". Also, generate an explanation to determine your final answer. Return your output in JSON format: { "explanation": "your explanation here", "answer": "your final response here" }. Only include the JSON object in your response."""
            
            user_prompt = """Context: %s\nQuery: %s\nGive your answer by analyzing step by step.\n""" % (context.strip(), query.strip())

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
            attn_mask = torch.ones_like(tokens)

            outputs  = model.generate(tokens, attention_mask=attn_mask, max_new_tokens=300, do_sample=False, temperature=None, top_p=None, return_dict_in_generate=True, output_scores=True, output_hidden_states=True)

            # New tokens only (exclude the prompt)
            sequences = outputs.sequences  # shape: [batch, prompt+new]
            new_token_ids = sequences[:, tokens.shape[1]:]  # [1, T_new]
            step_scores = outputs.scores  # list length T_new, each [batch, vocab]
            # Visible text per generated token (aligned to scores/hidden_states)
            tok_ids_1d = new_token_ids[0].tolist()
            tok_strs, piece_strs, joined = token_texts_from_ids(tokenizer, tok_ids_1d)
            token_char_spans = build_char_spans(piece_strs)

            # Find the first JSON object in the generated text
            json_str, json_start, json_end = find_first_json_object(joined)

            explanation_text, answer_text = "", ""
            ex_span_abs = (None, None)
            ans_span_abs = (None, None)

            if json_str is not None:
                # Try to parse; if it fails, we'll still locate spans
                try:
                    obj = json.loads(json_str)
                    explanation_text = obj.get("explanation", "")
                    answer_text = obj.get("answer", "")
                except Exception:
                    # fall back to regex (non-strict JSON)
                    m_ex = re.search(r'"explanation"\s*:\s*"(?P<v>.*?)"', json_str, flags=re.DOTALL)
                    m_an = re.search(r'"answer"\s*:\s*"(?P<v>.*?)"', json_str, flags=re.DOTALL)
                    explanation_text = m_ex.group("v") if m_ex else ""
                    answer_text = m_an.group("v") if m_an else ""

                # Locate exact character spans (inner, without quotes) for explanation/answer in the JSON
                ex_inner = locate_json_string_value_span(json_str, "explanation")
                an_inner = locate_json_string_value_span(json_str, "answer")

                if ex_inner != (None, None):
                    ex_span_abs = (json_start + ex_inner[0], json_start + ex_inner[1])
                if an_inner != (None, None):
                    ans_span_abs = (json_start + an_inner[0], json_start + an_inner[1])

            # Map char spans to token indices
            def span_to_token_indices(span_abs, spans):
                if span_abs == (None, None):
                    return []
                s_abs, e_abs = span_abs
                idxs = []
                for k, (ts, te) in enumerate(spans):
                    # overlap test
                    if not (te <= s_abs or ts >= e_abs):
                        idxs.append(k)
                return idxs

            ex_token_idxs = span_to_token_indices(ex_span_abs, token_char_spans)
            ans_token_idxs = span_to_token_indices(ans_span_abs, token_char_spans)

            # Answer token stats: use the FIRST token of the answer string (if any)
            if ans_token_idxs:
                ans_step = ans_token_idxs[0]
            else:
                # fallback: first content step so metrics are defined
                ans_step = first_content_step(new_token_ids, step_scores, tokenizer)[0]

            answer_token_info = collect_token_stats(ans_step, new_token_ids, step_scores, outputs, piece_strs[ans_step])

            # Explanation tokens: collect stats for ALL tokens overlapping the explanation string
            explanation_tokens_info = [collect_token_stats(s, new_token_ids, step_scores, outputs, piece_strs[s])
                                    for s in ex_token_idxs]

            # For your existing arrays, store answer's stats
            pred_label = answer_text  # you can rstrip punctuation if you want
            pred_logit = answer_token_info["logit"]
            pred_prob  = answer_token_info["prob"]
            last_vec   = answer_token_info["hidden"]
            
            all_pred_logit.append(pred_logit)
            all_pred_prob.append(pred_prob)
            all_pred_hs.append([] if last_vec is None else last_vec)
            all_input.append({'context': line['context'], 'query': line['query'], 'label': str(line['label']), 'ctx_label': str(line['ctx_label']), 'prior_label': str(line['prior_label']), f'cot_ans': pred_label, 'explanation': explanation_text, 'explanation_tokens': explanation_tokens_info})

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
    
    with open(f"{args.output_dir}{args.dataset}/{model_name}/all_pred_logit_{args.prompt_mode}_cot.pkl", 'wb') as f:
        pickle.dump(all_pred_logit, f)
    with open(f"{args.output_dir}{args.dataset}/{model_name}/all_pred_prob_{args.prompt_mode}_cot.pkl", 'wb') as f:
        pickle.dump(all_pred_prob, f)
    with open(f"{args.output_dir}{args.dataset}/{model_name}/all_pred_hs_{args.prompt_mode}_cot.pkl", 'wb') as f:
        pickle.dump(all_pred_hs, f)
    with open(f"{args.data_dir}{args.dataset}/{model_name}/all_input_{args.prompt_mode}_cot.json", 'w') as f:
        json.dump(all_input, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct") # meta-llama/Meta-Llama-3.1-8B-Instruct / google/gemma-2-9b-it / mistralai/Mistral-7B-Instruct-v0.3
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--dataset", type=str, default="strategyqa") # strategyqa / basefakepedia / multihopfakepedia / openbookqa / musique
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--prompt_mode", type=str, help = "prior_wo_context/prior_only/context_only/both_w_instruction/both_wo_instruction", default="both_wo_instruction")
    parser.add_argument("--output_dir", type=str, default="../results/")
    args = parser.parse_args()
    print(args)

    set_seed(args)

    main(args)
