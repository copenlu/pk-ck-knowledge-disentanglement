import sys
# sys.path.insert(0, '../transformers/src')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from torch.nn import functional as F
from tqdm import tqdm
import pdb
import pickle
import argparse
from nnpatch.subspace import LowRankOrthogonalProjection
import os

parser = argparse.ArgumentParser(description='Script for processing data and models.')
parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
parser.add_argument(
    '--dataset', 
    type=str, 
    default="ragtruth", 
    help='ragtruth, dolly'
)
parser.add_argument("--output_dir", type=str, default="../results/")
args = parser.parse_args()
if args.dataset == "ragtruth":
    response_path = "./datasets/response_with_llama3_8b.jsonl"
elif args.dataset == "dolly":
    response_path = "./datasets/response_dolly.jsonl"

response = []
with open(response_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        response.append(data)
if args.dataset == "ragtruth":
    source_info_path = "./datasets/source_info.jsonl"
elif args.dataset == "dolly":
    source_info_path = "./datasets/source_info_dolly.jsonl"
source_info_dict = {}

with open(source_info_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        source_info_dict[data['source_id']] = data


model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
device = "cuda"

tokenizer_for_temp = tokenizer

topk_head_path = "./ReDeEP/log/test_llama3_8B/topk_heads.json"

with open(topk_head_path,'r') as f:
    # [(layer, head)...]
    copy_heads = json.load(f)


def calculate_dist(sep_vocabulary_dist, sep_attention_dist):
    softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)  
    softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)  

    M = 0.5 * (softmax_mature_layer + softmax_anchor_layer) 

    # 4. Calculate log-softmax for the KL divergence
    log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)  
    log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1) 

    # 5. Calculate the KL divergences and then the JS divergences
    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').mean(-1)  
    kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none').mean(-1)  
    # # Fix bug: https://github.com/Jeryi-Sun/ReDEeP-ICLR/issues/2 but for stable calculation, we maintain the original implementation of JSD.
    # kl1 = F.kl_div(M.log(), softmax_mature.unsqueeze(0),  reduction='none').mean(-1)
    # kl2 = F.kl_div(M.log(), softmax_anchor,  reduction='none').mean(-1)
    js_divs = 0.5 * (kl1 + kl2) 
        
    return js_divs.cpu().item()*10e5

def calculate_ma_dist(sep_vocabulary_dist, sep_attention_dist):
    sep_vocabulary_dist = F.softmax(sep_vocabulary_dist, dim=-1)

    dist_diff = sep_vocabulary_dist - sep_attention_dist
    # 取绝对值
    abs_diff = torch.abs(dist_diff)

    # 计算 Manhattan 距离
    manhattan_distance = torch.sum(abs_diff)
    
    return manhattan_distance.cpu().item()

def is_hallucination_token(token_id, hallucination_spans):
    for span in hallucination_spans:
        if token_id >= span[0] and token_id <= span[1]:
            return True
    return False
def calculate_hallucination_spans(response, text, response_rag, tokenizer, prefix_len):
    hallucination_span = []
    if "dolly" in source_info_path:
        return hallucination_span
    for item in response:
        start_id = item['start']
        end_id = item['end']
        start_text = text+response_rag[:start_id]
        end_text = text+response_rag[:end_id]
        start_text_id = tokenizer(start_text, return_tensors="pt").input_ids
        end_text_id = tokenizer(end_text, return_tensors="pt").input_ids
        start_id = start_text_id.shape[-1]
        end_id = end_text_id.shape[-1]
        hallucination_span.append([start_id, end_id])
    return hallucination_span

select_response = []
data_type =  "llama-3-8b-instruct" 

proj_rank2 = LowRankOrthogonalProjection.from_pretrained("../context-vs-prior-finetuning/analysis/projections/Meta-Llama-3.1-8B-Instruct-L17").to('cuda')

all_hal_rank2_contrib, all_nonhal_rank2_contrib = [], []
for i in tqdm(range(len(response))):
    if response[i]['model'] == data_type and response[i]["split"] == "test":
        response_rag = response[i]['response']
        source_id = response[i]['source_id']
        temperature = response[i]['temperature']
        prompt =  source_info_dict[source_id]['prompt']
        messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt[:12000]}
                ]
        text = tokenizer_for_temp.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # print(text)
        input_text = text+response_rag
        # print("all_text_len:", len(input_text))
        # print("prompt_len", len(prompt))
        # print("respond_len", len(response_rag))
        input_ids = tokenizer([input_text], return_tensors="pt").input_ids.to('cuda')
        prefix_ids = tokenizer([text], return_tensors="pt").input_ids
        continue_ids = input_ids[0, prefix_ids.shape[-1]:] # todo 这边要改成幻觉 token 的起止位置
        if "labels" in response[i].keys():
            if len(response[i]['labels']) > 0:
                hal_label = True
            else:
                hal_label = False
        else:
            continue

        with torch.no_grad():
            outputs = model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=True
                )

        # skip tokens without hallucination
        hidden_states = outputs["hidden_states"] # tuple ([batch, seq_len, vocab_size], ..., ) 
        last_hidden = hidden_states[-1][0] # [prefix_len, hidden_size]
        response_hs = last_hidden[prefix_ids.shape[-1]:]

        response_rank2_contrib = response_hs @ proj_rank2.weight
    
        if hal_label:
            all_hal_rank2_contrib.append(response_rank2_contrib.detach().cpu().numpy())
        else:
            all_nonhal_rank2_contrib.append(response_rank2_contrib.detach().cpu().numpy())

model_name = args.model_name.strip().split('/')[-1]
os.makedirs(f"{args.output_dir}{args.dataset}/{model_name}", exist_ok=True)
    
with open(f"{args.output_dir}{args.dataset}/{model_name}/all_hal_rank2_contrib_{args.dataset}.pkl", 'wb') as f:
    pickle.dump(all_hal_rank2_contrib, f)
with open(f"{args.output_dir}{args.dataset}/{model_name}/all_nonhal_rank2_contrib_{args.dataset}.pkl", 'wb') as f:
    pickle.dump(all_nonhal_rank2_contrib, f)