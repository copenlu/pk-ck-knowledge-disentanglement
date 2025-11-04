import argparse
import os
import torch
from loguru import logger
import pandas as pd

from nnpatch import activation_patch, attribution_patch, Site, Sites, activation_zero_patch, attribution_zero_patch
from nnpatch.api.llama import Llama3
from nnpatch.site import HeadSite, MultiSite

from model_utils.utils import load_model_and_tokenizer, construct_paths_and_dataset_kwargs, construct_test_results_dir
from model_utils.utils import construct_query_with_demonstrations, MODEL_ID_TO_TEMPLATES_DICT
from sklearn.model_selection import train_test_split


PROMPT_MODE_TO_INSTRUCTION = {
    "prior_only": "Ignore the context in answering the query.",
    "context_only": "Only consider the context in answering the query.",
    "both_w_instruction": "Consider the context in answering the query."
}

def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-store", default="/dlabscratch1/public/llm_weights/llama3.1_hf/")
    parser.add_argument("--model-id", default="Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--context-weight-format", "-CWF", default="float", choices=["float", "instruction"])
    parser.add_argument("--finetune-configuration", "-FTC", default="peftq_proj_k_proj_v_proj_o_proj")
    parser.add_argument("--finetune-training-args", "-FTCA", default=None, type=str)
    parser.add_argument("--finetune-seed", "-FTS", default=3, type=int)
    parser.add_argument("--finetune-training-samples", "-FTTS", default=2048, type=int)
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--shots", default=10, type=int)
    parser.add_argument("--dataset", "-DS", type=str, help="Name of the dataset class", default="BaseFakepedia")
    parser.add_argument("--eval-dataset", "-EDS", type=str, help="Name of the evaluation dataset", default="BaseFakepedia")
    parser.add_argument("--dataset-index", default=0, type=int)
    parser.add_argument("--n-samples", default=-1, type=int)
    parser.add_argument("--output-dir", default="patching_results")
    parser.add_argument("--name", default="")
    parser.add_argument("--context-info-flow", action="store_true")
    parser.add_argument("--prior-info-flow", action="store_true")
    parser.add_argument("--context-to-prior", action="store_true")
    parser.add_argument("--prior-to-context", action="store_true")
    parser.add_argument("--both_to_prior", action="store_true")
    parser.add_argument("--both_to_context", action="store_true")
    parser.add_argument("--topk", default=10, type=int)
    parser.add_argument("--new-few-shots", default=None, type=int)
    parser.add_argument("--zero", action="store_true")
    parser.add_argument("--no-filtering", action="store_true")
    parser.add_argument("--batch-size", default=-1, type=int)
    parser.add_argument("--force-model-confidence", action="store_true")
    parser.add_argument("--heads", default=["o"], nargs="+")
    parser.add_argument("--source-heads", default=["o", "q"], nargs="+")
    parser.add_argument("--layer-range", "-LR", default=[0, -1], nargs=2, type=int)
    parser.add_argument("--layers", default=None, nargs="+", type=int)
    parser.add_argument(
        "-SP",
        "--subsplit",
        type=str,
        default="nodup_relpid",
        # choices=[
        #     "nodup_relpid",
        #     "nodup_relpid_obj",
        #     "nodup_relpid_subj",
        #     "nodup_s_or_rel_or_obj",
        #     "base",
        # ],
        help="Name of the dataset subsplit to use.",
    )
    parser.add_argument(
        "--eval-subsplit",
        type=str,
        default="nodup_relpid",
        # choices=[
        #     "nodup_relpid",
        #     "nodup_relpid_obj",
        #     "nodup_relpid_subj",
        #     "nodup_s_or_rel_or_obj",
        #     "base",
        # ],
        help="Name of the dataset subsplit to use.",
    )
    return parser

def get_instruct_str(args):
    if "mistral" in args.model_id.lower() or "llama" in args.model_id.lower():
        return "Instruct"
    else:
        return "it"

def dir_is_instruct(args, dir_name):
    if get_instruct_str(args) in args.model_id:
        return get_instruct_str(args) in dir_name
    else:
        return get_instruct_str(args) not in dir_name

def paths_from_args(args):
    BASE_MODEL = os.path.join(args.model_store, args.model_id)
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    DATAROOT = os.path.join('/work/Work/sekh_phd/explanation_probe', "data", args.dataset, args.model_id)
    df = pd.read_csv(f'{DATAROOT}/projection_data.csv')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=args.finetune_seed)
    train_df.to_csv(f'{DATAROOT}/train.csv', index=False)
    test_df.to_csv(f'{DATAROOT}/test.csv', index=False)
    TRAIN_DATA = os.path.join(DATAROOT, "train.csv")
    # DATASET_CONFIG_NAME = f"{args.dataset}_{args.subsplit}-ts{args.finetune_training_samples}"
    
    instruct_str = get_instruct_str(args)
    # models_dir = os.path.join(DATAROOT, DATASET_CONFIG_NAME, str(args.finetune_seed), "models")
    # finetuned_dir = next((d for d in os.listdir(models_dir) if d.startswith(f"{args.model_id}-") and d.endswith(f"-cwf_{args.context_weight_format}") and ((args.finetuned and "NT" not in d) or (not args.finetuned and "NT" in d)) and dir_is_instruct(args, d)), None)
    # if args.finetuned:
    #     if finetuned_dir and (args.finetune_training_args is None or args.finetune_training_args == "None"):
    #         MODEL_NAME = finetuned_dir  
    #     else:
    #         MODEL_NAME = f"{args.model_id}-{args.finetune_configuration}-{args.finetune_training_args}-cwf_{args.context_weight_format}"
    # else:
    #     if finetuned_dir and (args.finetune_training_args is None or args.finetune_training_args == "None"):
    #         MODEL_NAME = finetuned_dir
    #     else:
    #         MODEL_NAME = f"{args.model_id}-NT"

    MODEL_NAME = f"{args.model_id}"
    print(MODEL_NAME)
    
    # MERGED_MODEL = os.path.join(DATAROOT, DATASET_CONFIG_NAME, str(args.finetune_seed), "models", MODEL_NAME, "merged")
    # PEFT_MODEL = os.path.join(DATAROOT, DATASET_CONFIG_NAME, str(args.finetune_seed), "models", MODEL_NAME, "model")
    # VAL_DATA_ALL = os.path.join(DATAROOT, "splits", args.subsplit, "val.csv")
    TRAIN_DATA_ALL = os.path.join(DATAROOT, "train.csv")
    
    # sp_args = ""
    # sp_args = f"-sp_{args.eval_subsplit}"

    RESULTS_DIR = os.path.join('/work/Work/sekh_phd/explanation_probe', "results", args.dataset, MODEL_NAME)
    # FEW_SHOT_SAMPLE = os.path.join(RESULTS_DIR, "few_shot_sample.csv")
    TEST_DATA = os.path.join(DATAROOT, "test.csv")
    
    return {
        "BASE_MODEL": BASE_MODEL,
        "MODEL_NAME": MODEL_NAME,
        "DATAROOT": DATAROOT,
        "TRAIN_DATA": TRAIN_DATA,
        # "DATASET_CONFIG_NAME": DATASET_CONFIG_NAME,
        # "PEFT_MODEL": PEFT_MODEL,
        # "MERGED_MODEL": MERGED_MODEL,
        # "VAL_DATA_ALL": VAL_DATA_ALL,
        "TRAIN_DATA_ALL": TRAIN_DATA_ALL,
        "RESULTS_DIR": RESULTS_DIR,
        # "FEW_SHOT_SAMPLE": FEW_SHOT_SAMPLE,
        "TEST_DATA": TEST_DATA
    }
    

def load_model_and_tokenizer_from_args(paths, args):
    attention_implementation = "eager" if "gemma" in args.model_id.lower() else "sdpa"
    if args.finetuned:
        print("Loading finetuned model:", paths["MERGED_MODEL"])
        model, tokenizer = load_model_and_tokenizer(paths["MERGED_MODEL"], args.load_4bit, False, False, padding_side="left", attn_implementation=attention_implementation)
    else:
        print("Loading base model:", paths["BASE_MODEL"])
        model, tokenizer = load_model_and_tokenizer(paths["BASE_MODEL"], args.load_4bit, False, False, padding_side="left", attn_implementation=attention_implementation)
    return model, tokenizer


def filter_for_true_pairs(data):
    # recompute the is_correct
    data["is_correct"] = data.apply(lambda x: x["predictions"].startswith(x["answer"]), axis=1)
    # filter out the false samples (both odd and even indices need to be true)
    trues = data[data.is_correct]
    even_true = trues[trues.index % 2 == 0 ].index
    odd_true = trues[trues.index % 2 == 1 ].index

    true_indices = set(even_true) & set([i-1 for i in odd_true])
    true_indices = sorted(list(true_indices) + [i+1 for i in true_indices])
    data = data.iloc[true_indices]
    return data

def encode_answer(answers_source, answers_target, tokenizer, device, args):
    # convert to strings
    if not isinstance(answers_source, list):
        answers_source = answers_source.astype(str).tolist()
        answers_target = answers_target.astype(str).tolist()
    else:
        answers_source = [str(a) for a in answers_source]
        answers_target = [str(a) for a in answers_target]
    # test whether we need to add a newline before the answer
    prefix= ""
    idx = 0
    if MODEL_ID_TO_TEMPLATES_DICT[args.model_id][0]["ROUND"].replace("{}", "")[-1] == "\n":
        logger.info(f"Round ends with newline, testing if we need to add it")
        test_toks = tokenizer.encode("\n" + answers_source[0])
        if tokenizer.decode(test_toks)[0] == "\n":
            logger.info("Tokenizer merges newline and first token, adding it before the answer")
            prefix = "\n"
            idx = 1
        else:
            logger.info("Tokenizer does not merge newline and first token, not adding it")
    # elif "mistral" in args.model_id.lower():
    #     prefix = "[/INST]" # mistral tokenizer tokenizes differently when there is a [\INST] in front of the answer
    #     idx = 1

    target_answer_index = torch.tensor([tokenizer.encode(prefix + a, add_special_tokens=False)[idx] for a in answers_target]).to(device)
    source_answer_index = torch.tensor([tokenizer.encode(prefix + a, add_special_tokens=False)[idx] for a in answers_source]).to(device)
    return source_answer_index, target_answer_index

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



def collect_data(args, PATHS, tokenizer, device):
    print("Reading test data from ", PATHS["TEST_DATA"])
    test_data = pd.read_csv(PATHS["TEST_DATA"])
    
    # filter test data for correct predictions
    n_before = len(test_data)
    # args.n_samples = -1
    if not args.no_filtering:
        test_data = filter_for_true_pairs(test_data)
        # reset index
        test_data.reset_index(drop=True, inplace=True)
        logger.info(f"Filtered {n_before - len(test_data)} samples")

    prior_only_texts, context_only_texts, both_texts = [], [], []
    for row in test_data.itertuples(index=False):
        text_prior_only = apply_chat_template(PATHS, tokenizer, getattr(row, "query"), getattr(row, "context"), prompt_mode='prior_only', dataset=args.dataset)
        text_context_only = apply_chat_template(PATHS, tokenizer, getattr(row, "query"), getattr(row, "context"), prompt_mode='context_only', dataset=args.dataset)
        text_both = apply_chat_template(PATHS, tokenizer, getattr(row, "query"), getattr(row, "context"), prompt_mode='both_w_instruction', dataset=args.dataset)
        prior_only_texts.append(text_prior_only)
        context_only_texts.append(text_context_only)
        both_texts.append(text_both)
    test_data["prior_only"] = prior_only_texts
    test_data["context_only"] = context_only_texts
    test_data["both"] = both_texts    

    if args.n_samples == -1:
        args.n_samples = len(test_data)  
        
    if args.context_info_flow:
        target_df = test_data[test_data.weight_context == 1.0].reset_index(drop=True)
    elif args.prior_info_flow:
        target_df = test_data[test_data.weight_context == 0.0]
        target_df = target_df.groupby("answer").first().reset_index()
    else:
        target_df = test_data

    if args.n_samples == -1:
        args.n_samples = len(target_df)
    elif args.n_samples > len(target_df):
        logger.warning(f"Only {len(target_df)} samples available, reducing n_samples to this value")
        args.n_samples = len(target_df)
    
    if args.context_to_prior:
        source_df = target_df.copy()[target_df.weight_context == 0.0]
        target_df = target_df[target_df.weight_context == 1.0]
    elif args.prior_to_context:
        source_df = target_df.copy()[target_df.weight_context == 1.0]
        target_df = target_df[target_df.weight_context == 0.0]
    elif args.both_to_prior:
        mask = (target_df.weight_prior == 0.0)# & (target_df.weight_context == 1.0)
        source_df = target_df.copy()[mask]
        target_df = target_df[mask]
    elif args.both_to_context:
        mask = (target_df.weight_context == 0.0)# & (target_df.weight_prior == 1.0)
        source_df = target_df.copy()[mask]
        target_df = target_df[mask]
    elif args.context_info_flow or args.prior_info_flow:
        clean_to_corrupted_index =  [(i + 1) % len(target_df) for i in range(len(target_df))]
        source_df = target_df.iloc[clean_to_corrupted_index]
    else:
        clean_to_corrupted_index =  [(i + 1) if (i % 2) == 0 else i-1 for i in range(len(target_df))]
        source_df = target_df.iloc[clean_to_corrupted_index]
    
    target_df = target_df.iloc[args.dataset_index: args.dataset_index+args.n_samples]
    source_df = source_df.iloc[args.dataset_index: args.dataset_index+args.n_samples]

    if args.both_to_prior:
        source_answer_index, target_answer_index = encode_answer(source_df.prior_answer.tolist(), target_df.answer.tolist(), tokenizer, device, args)
    if args.both_to_context:
        source_answer_index, target_answer_index = encode_answer(source_df.ctx_answer.tolist(), target_df.answer.tolist(), tokenizer, device, args)

    same_answer_indices = source_answer_index == target_answer_index
    target_df = target_df[~same_answer_indices.cpu().numpy()]
    source_df = source_df[~same_answer_indices.cpu().numpy()]
    source_answer_index = source_answer_index[~same_answer_indices]
    target_answer_index = target_answer_index[~same_answer_indices]

    target_df.reset_index(drop=True, inplace=True)
    source_df.reset_index(drop=True, inplace=True)

    if args.both_to_prior:
        target_text = target_df.both.tolist()
        source_text = source_df.prior_only.tolist()
    if args.both_to_context:
        target_text = target_df.both.tolist()
        source_text = source_df.context_only.tolist()
    
    target_tokens = tokenizer(target_text, return_tensors="pt", padding=True)
    attention_mask_target = target_tokens["attention_mask"].to(device)
    source_tokens = tokenizer(source_text, return_tensors="pt", padding=True)
    attention_mask_source = source_tokens["attention_mask"].to(device)
    target_tokens = target_tokens["input_ids"].to(device)
    source_tokens = source_tokens["input_ids"].to(device)
    
    return target_df, source_df, target_tokens, source_tokens, target_answer_index, source_answer_index, attention_mask_target, attention_mask_source


def combine_batches(list_of_outs):
    combined = {}
    # combine batches
    total = 0
    for out_batch, batch_len in list_of_outs:
        for head_name, value in out_batch.items():
            if torch.any(value.isnan()):
                logger.warning(f"NaN in {head_name}")
                continue
            if head_name not in combined:
                combined[head_name] = value * batch_len
            else:
                combined[head_name] += value * batch_len
            total += batch_len
                
    for key in combined:
        combined[key] /= total
    return combined


def batch_act_patch(args, nnmodel, sites, clean_tokens, corrupted_tokens, correct_index, incorrect_index, attention_mask_clean, attention_mask_corrupted, force_model_confidence=False):
    if args.batch_size == -1:
        args.batch_size = len(clean_tokens)
    logger.info(f"ACT PATCH with batch size {args.batch_size}")
    results = []
    for i in range(0, len(clean_tokens), args.batch_size):
        HeadSite.reset()
        logger.info(f"Batch {i}")
        if args.zero:
            activation_zero_patch(nnmodel, Llama3, sites, corrupted_tokens[i:i+args.batch_size], incorrect_index[i:i+args.batch_size], target_attention_mask=attention_mask_corrupted[i:i+args.batch_size])
        else:
            out = activation_patch(nnmodel, Llama3, sites, clean_tokens[i:i+args.batch_size], corrupted_tokens[i:i+args.batch_size], correct_index[i:i+args.batch_size], incorrect_index[i:i+args.batch_size], source_attention_mask=attention_mask_clean[i:i+args.batch_size], target_attention_mask=attention_mask_corrupted[i:i+args.batch_size], force_model_confidence=args.force_model_confidence)
        results.append((out, len(clean_tokens[i:i+args.batch_size])))
        
    all_results = combine_batches(results)
    return all_results

def ptd(toks):
    print(tokenizer.decode(toks))