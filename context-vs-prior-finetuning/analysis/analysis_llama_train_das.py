# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append("..")
from analysis.circuit_utils.das import *
from functools import partial
from torch.utils.data import DataLoader, random_split
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
from analysis.circuit_utils.decoding import get_decoding_args, get_data, generate_title, get_plot_prior_patch, get_plot_context_patch, get_plot_weightcp_patch, get_plot_weightpc_patch

# from main import load_model_and_tokenizer
from nnpatch.subspace.interventions import train_projection, LowRankOrthogonalProjection


from nnpatch.api.llama import Llama3

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

def main(args):
    model_split_list = args.model_name.strip().split('/')
    MODEL_STORE=f"{model_split_list[0]}/"
    plot_dir = f"/work/Work/sekh_phd/explanation_probe/results/{args.dataset}" # provide the plot path here
    device = args.device

    DATASET_NAME = args.dataset

    os.makedirs(plot_dir, exist_ok=True)
    num_sample = int(args.num_sample)

    PATHS, args_path = get_decoding_args(finetuned=False, no_filtering=True, load_in_4bit=False, both_to_prior=True, both_to_context=False, cwf="instruction", model_id=model_split_list[-1], model_store=MODEL_STORE, dataset=DATASET_NAME, n_samples=num_sample)
    model, tokenizer = load_model_and_tokenizer_from_args(PATHS, args_path)
    
    st_1, st_2, tt, si_1, si_2, ti, ams_1, ams_2, amt, tit, amti = prepare_train_data(args_path, PATHS, tokenizer, device, same_query=True, remove_weight=False)

    confident_indices = filter_confident_samples(args_path, model, tt, st_1, st_2, ti, si_1, si_2, amt, ams_1, ams_2, batch_size=32)
    print(f"Number of confident samples: {len(confident_indices)}/{tt.shape[0]}")
    train_dataset = create_dataset(st_1[confident_indices], st_2[confident_indices], tt[confident_indices], si_1[confident_indices], si_2[confident_indices], ti[confident_indices], ams_1[confident_indices], ams_2[confident_indices], amt[confident_indices])
    # print_random_sample(train_dataset, tokenizer, prefix="Train")

    st_1_test, st_2_test, tt_test, si_1_test, si_2_test, ti_test, ams_1_test, ams_2_test, amt_test, tit_test, amti_test = prepare_test_data(args_path, PATHS, tokenizer, device, same_query=True, remove_weight=False)
    test_dataset = create_dataset(st_1_test, st_2_test, tt_test, si_1_test, si_2_test, ti_test, ams_1_test, ams_2_test, amt_test)
    # print_random_sample(test_dataset, tokenizer, prefix="Test")

    proj = LowRankOrthogonalProjection(embed_dim=4096, rank=2)

    proj = train_projection(model, proj, layer=17, train_dataset=train_dataset, val_dataset=test_dataset, epochs=2, batch_size=8)

    # Save
    proj.save_pretrained("projections_v2/Meta-Llama-3.1-8B-Instruct-L17")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct") # meta-llama/Meta-Llama-3.1-8B-Instruct / google/gemma-2-9b-it / mistralai/Mistral-7B-Instruct-v0.3
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--dataset", type=str, default="openbookqa") # strategyqa / basefakepedia / multihopfakepedia / openbookqa / musique
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--num_sample", type=int, default=-1)
    # parser.add_argument("--batch_size_patch", type=int, default=24)
    # parser.add_argument("--batch_size_plot", type=int, default=33)
    # parser.add_argument("--threshold_bp", type=float, default=0.88)
    # parser.add_argument("--threshold_bc", type=float, default=0.88)
    # parser.add_argument("--prompt_mode", type=str, help = "prior_wo_context/prior_only/context_only/both_w_instruction/both_wo_instruction", default="both_w_instruction")
    # parser.add_argument("--output_dir", type=str, default="../results/")
    args = parser.parse_args()
    print(args)

    set_seed(args)

    main(args)





