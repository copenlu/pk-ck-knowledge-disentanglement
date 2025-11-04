# %load_ext autoreload
# %autoreload 2
import sys
sys.path.append("..")
from nnsight import NNsight
import torch
import os
from tqdm import tqdm, trange

from nnsight import NNsight

from analysis.circuit_utils.visualisation import *
from analysis.circuit_utils.model import *
from analysis.circuit_utils.validation import *
from analysis.circuit_utils.decoding import *
from analysis.circuit_utils.utils import *
from analysis.circuit_utils.decoding import get_decoding_args, get_data, generate_title, get_plot_prior_patch, get_plot_context_patch, get_plot_weightcp_patch, get_plot_weightpc_patch, get_plot_weightbp_patch, get_plot_weightbc_patch
from analysis.circuit_utils.das import *

from nnpatch.subspace.interventions import train_projection, create_dataset, LowRankOrthogonalProjection

# from main import load_model_and_tokenizer


from nnpatch.api.gemma import Gemma2

# jupyter_enable_mathjax()

def main(args):
    model_split_list = args.model_name.strip().split('/')
    MODEL_STORE=f"{model_split_list[0]}/"
    plot_dir = f"/work/Work/sekh_phd/explanation_probe/results/{args.dataset}" # provide the plot path here
    DATASET_NAME = args.dataset

    os.makedirs(plot_dir, exist_ok=True)
    num_sample = int(args.num_sample)

    PATHS, args_path = get_decoding_args(finetuned=False, no_filtering=True, load_in_4bit=False, both_to_prior=True, both_to_context=False, cwf="instruction", model_id=model_split_list[-1], model_store=MODEL_STORE, dataset=DATASET_NAME, n_samples=num_sample)
    model, tokenizer = load_model_and_tokenizer_from_args(PATHS, args_path)
    nnmodel = NNsight(model)


    ## Patch
    target_df, source_df, target_tokens, source_tokens, target_answer_index, source_answer_index, attention_mask_target, attention_mask_source = get_data(args_path, PATHS, tokenizer)
    both_to_prior_args = [target_tokens, attention_mask_target, target_tokens, source_tokens, attention_mask_target, attention_mask_source, target_answer_index, source_answer_index]

    PATHS, args_path = get_decoding_args(finetuned=False, no_filtering=True, load_in_4bit=False, both_to_prior=False, both_to_context=True, cwf="instruction", model_id=model_split_list[-1], model_store=MODEL_STORE, dataset=DATASET_NAME, n_samples=num_sample)
    target_df, source_df, target_tokens, source_tokens, target_answer_index, source_answer_index, attention_mask_target, attention_mask_source = get_data(args_path, PATHS, tokenizer)
    both_to_context_args = [target_tokens, attention_mask_target, target_tokens, source_tokens, attention_mask_target, attention_mask_source, target_answer_index, source_answer_index]


    ## Auto search
    both_to_prior_range = auto_search(model, tokenizer, both_to_prior_args, n_layers=42, phi=0.05, eps=0.3, thres=args.threshold_bp, batch_size=args.batch_size_patch, api=Gemma2)
    both_to_context_range = auto_search(model, tokenizer, both_to_context_args, n_layers=42, phi=0.05, eps=0.3, thres=args.threshold_bc, batch_size=args.batch_size_patch, api=Gemma2)
    # both_to_prior_range = range(16, 32)
    # both_to_context_range = range(13, 32)
    print(both_to_prior_range)
    print(both_to_context_range)


    site_1_config = { 
        "o":
        {
            "layers": both_to_prior_range,
        },
    }
    figr, figp = get_plot_weightbp_patch(nnmodel, tokenizer, *both_to_prior_args, site_1_config, N_LAYERS=42, batch_size=args.batch_size_plot, output_dir=plot_dir, api=Gemma2, title=generate_title(site_1_config, f"BP - {model_split_list[-1]}"))

    site_1_config = { 
        "o":
        {
            "layers": both_to_context_range,
        },
    }
    figr, figp = get_plot_weightbc_patch(nnmodel, tokenizer, *both_to_context_args, site_1_config, N_LAYERS=42, batch_size=args.batch_size_plot, output_dir=plot_dir, api=Gemma2, title=generate_title(site_1_config, f"BC - {model_split_list[-1]}"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it") # meta-llama/Meta-Llama-3.1-8B-Instruct / google/gemma-2-9b-it / mistralai/Mistral-7B-Instruct-v0.3
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--dataset", type=str, default="openbookqa") # strategyqa / basefakepedia / multihopfakepedia / openbookqa / musique
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--num_sample", type=int, default=200)
    parser.add_argument("--batch_size_patch", type=int, default=10)
    parser.add_argument("--batch_size_plot", type=int, default=20)
    parser.add_argument("--threshold_bp", type=float, default=0.75)
    parser.add_argument("--threshold_bc", type=float, default=0.85)
    # parser.add_argument("--prompt_mode", type=str, help = "prior_wo_context/prior_only/context_only/both_w_instruction/both_wo_instruction", default="both_w_instruction")
    # parser.add_argument("--output_dir", type=str, default="../results/")
    args = parser.parse_args()
    print(args)

    # set_seed(args)

    main(args)





