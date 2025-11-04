import os
import argparse
import pickle
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
# from matplotlib import pyplot as plt  # unused; comment out unless you need plots
from nnpatch.subspace import LowRankOrthogonalProjection

# --------- utils ---------
def set_seed(args):
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if ("cuda" in args.device) and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@torch.no_grad()
def assign_pk_ck_axes(
    u1: torch.Tensor,                 # (d,)
    u2: torch.Tensor,                 # (d,)
    pk_hidden: List[torch.Tensor],    # list of (N_i, d) PK-only batches
    ck_hidden: List[torch.Tensor],    # list of (N_i, d) CK-only batches
) -> Dict:
    """
    Determine which axis (u1 or u2) corresponds to Parametric (PK) vs Context (CK).
    Uses contrastive anchoring by PK-only vs CK-only mean projections.
    """
    assert u1.ndim == 1 and u2.ndim == 1, "u1/u2 must be shape (d,)"
    device = u1.device
    u2 = u2.to(device)

    # concat and move to device
    pk_cat = torch.cat([x.to(device) for x in pk_hidden], dim=0)  # (N_pk, d)
    ck_cat = torch.cat([x.to(device) for x in ck_hidden], dim=0)  # (N_ck, d)

    # Orthonormalize (Gram–Schmidt) to be safe
    def normalize(v):
        n = torch.linalg.norm(v)
        return v / (n + 1e-12)

    # u1 = normalize(u1)
    # u2 = u2 - (u1 @ u2) * u1
    # u2 = normalize(u2)

    # Means
    mu_pk = pk_cat.mean(dim=0)  # (d,)
    mu_ck = ck_cat.mean(dim=0)  # (d,)

    # Projections
    def proj2(mu):
        return torch.stack([u1 @ mu, u2 @ mu])  # (2,)
    p_pk = proj2(mu_pk)  # [on u1, on u2]
    p_ck = proj2(mu_ck)

    # Assign axes: whichever |projection| is larger for PK mean becomes PK axis
    pk_axis_idx = torch.argmax(torch.abs(p_pk)).item()  # 0->u1, 1->u2
    ck_axis_idx = 1 - pk_axis_idx

    which_pk = "u1" if pk_axis_idx == 0 else "u2"
    which_ck = "u2" if pk_axis_idx == 0 else "u1"

    U = [u1, u2]
    u_pk = U[pk_axis_idx].clone()
    u_ck = U[ck_axis_idx].clone()

    # Make anchors positive along their own axis (sign convention)
    if (u_pk @ mu_pk) < 0:
        u_pk = -u_pk
        p_pk[pk_axis_idx] = -p_pk[pk_axis_idx]
        p_ck[pk_axis_idx] = -p_ck[pk_axis_idx]
    if (u_ck @ mu_ck) < 0:
        u_ck = -u_ck
        p_pk[ck_axis_idx] = -p_pk[ck_axis_idx]
        p_ck[ck_axis_idx] = -p_ck[ck_axis_idx]

    return {
        "u_pk": u_pk,                     # (d,)
        "u_ck": u_ck,                     # (d,)
        "pk_proj": torch.stack([p_pk[pk_axis_idx], p_pk[ck_axis_idx]]),  # [on u_pk, on u_ck]
        "ck_proj": torch.stack([p_ck[pk_axis_idx], p_ck[ck_axis_idx]]),  # [on u_pk, on u_ck]
        "which_pk": which_pk,
        "which_ck": which_ck,
    }

# --------- I/O helpers ---------
def load_hidden_numpy(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        arr = pickle.load(f)
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected shape (N, d), got {arr.shape}")
    return arr

def to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)

def load_vector_any(path: str, device: str) -> torch.Tensor:
    """
    Load a (d,) vector from .pkl/.npy/.npz or .pt/.pth. Returns float32 tensor on device.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    vec = None
    try:
        if ext in [".pt", ".pth"]:
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, torch.Tensor):
                vec = obj
            else:
                vec = torch.as_tensor(obj)
        elif ext in [".npy"]:
            vec = np.load(path)
            vec = torch.from_numpy(vec)
        elif ext in [".npz"]:
            npz = np.load(path)
            # try first array inside
            key = list(npz.keys())[0]
            vec = torch.from_numpy(npz[key])
        else:
            # assume pickle of numpy array/vector
            with open(path, "rb") as f:
                obj = pickle.load(f)
            vec = torch.as_tensor(obj)
    except Exception as e:
        raise RuntimeError(f"Failed to load vector from {path}: {e}")

    vec = vec.squeeze()
    if vec.ndim != 1:
        raise ValueError(f"{path}: expected vector (d,), got shape {tuple(vec.shape)}")
    return vec.to(device=device, dtype=torch.float32)

# --------- main ---------
def main(args):
    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    model_dirname = args.model_name.strip().split("/")[-1]

    # Datasets to aggregate for anchoring
    datasets = ["strategyqa", "basefakepedia", "multihopfakepedia", "openbookqa"]

    # ---- Load or define u1/u2 ----
    if "mistral" in args.model_name.lower():
        proj_path = "../context-vs-prior-finetuning/analysis/projections_v2/Mistral-7B-Instruct-v0.3-L16_epoch_1_v2"
        try:
            proj = LowRankOrthogonalProjection.from_pretrained(proj_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load projection from {proj_path}: {e}")
        u = proj.weight  # expect (r, d)
        if isinstance(u, np.ndarray):
            u = torch.from_numpy(u)
        if not torch.is_tensor(u):
            raise TypeError(f"proj.weight is type {type(u)}; expected torch.Tensor or np.ndarray")
        u = u.to(device=device, dtype=torch.float32)
        if u.ndim != 2 or u.shape[0] < 2:
            raise ValueError(f"Projection weight must be (r, d) with r>=2; got {tuple(u.shape)}")
        u1, u2 = u[:, 0], u[:, 1]
        cos_sim = torch.dot(u1, u2) / (u1.norm() * u2.norm() + 1e-12)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # numerical safety
        angle_rad = torch.acos(cos_sim)
        print(torch.rad2deg(angle_rad).item())
    else:
        if not (args.u1_path and args.u2_path):
            raise RuntimeError("Non-llama model: please provide --u1_path and --u2_path.")
        u1 = load_vector_any(args.u1_path, device)
        u2 = load_vector_any(args.u2_path, device)

    # Collect PK-only and CK-only hidden states across datasets
    pk_hidden_runs: List[torch.Tensor] = []
    ck_hidden_runs: List[torch.Tensor] = []

    for ds in datasets:
        path_prior_only = os.path.join(
            args.output_dir, ds, model_dirname, "all_pred_hs_prior_only_no_explanation.pkl"
        )
        # If your CK files are named "prior_only_in_context", try that if "context_only" missing
        path_context_only = os.path.join(
            args.output_dir, ds, model_dirname, "all_pred_hs_context_only_no_explanation.pkl"
        )

        try:
            arr_pk = load_hidden_numpy(path_prior_only)  # (N_pk, d)
        except FileNotFoundError:
            print(f"[WARN] Missing PK-only file for {ds}: {path_prior_only}; skipping dataset.")
            continue

        try:
            arr_ck = load_hidden_numpy(path_context_only)  # (N_ck, d)
        except FileNotFoundError:
            alt = os.path.join(args.output_dir, ds, model_dirname, "all_pred_hs_prior_only_in_context_no_explanation.pkl")
            if os.path.isfile(alt):
                print(f"[INFO] Using prior_only_in_context as CK for {ds}")
                arr_ck = load_hidden_numpy(alt)
            else:
                print(f"[WARN] Missing CK-only file for {ds} (tried {path_context_only} and {alt}); skipping dataset.")
                continue

        # Convert to tensors
        pk_hidden_runs.append(to_tensor(arr_pk, device))
        ck_hidden_runs.append(to_tensor(arr_ck, device))

    if len(pk_hidden_runs) == 0 or len(ck_hidden_runs) == 0:
        raise RuntimeError("No PK-only or CK-only tensors loaded. Check your file paths/names.")

    # ---- Assign axes ----
    out = assign_pk_ck_axes(u1, u2, pk_hidden_runs, ck_hidden_runs)
    print(f"PK axis is {out['which_pk']}; CK axis is {out['which_ck']}")
    print("PK mean projections [on u_pk, on u_ck]:", out["pk_proj"].tolist())
    print("CK mean projections [on u_pk, on u_ck]:", out["ck_proj"].tolist())

    # Export labeled axes if you want to reuse downstream
    if args.save_axes_dir:
        os.makedirs(args.save_axes_dir, exist_ok=True)
        torch.save(out["u_pk"].cpu(), os.path.join(args.save_axes_dir, f"{model_dirname}_u_pk.pt"))
        torch.save(out["u_ck"].cpu(), os.path.join(args.save_axes_dir, f"{model_dirname}_u_ck.pt"))
        print(f"Saved u_pk/u_ck to {args.save_axes_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="../results/")
    # parser.add_argument("--u1_path", type=str, default="", help="Path to saved u1 (vector: .pt/.npy/.npz/.pkl)")
    # parser.add_argument("--u2_path", type=str, default="", help="Path to saved u2 (vector: .pt/.npy/.npz/.pkl)")
    # parser.add_argument("--proj_path", type=str, default="",
    #                     help="If model_name contains 'llama', optional path to LowRankOrthogonalProjection")
    parser.add_argument("--save_axes_dir", type=str, default="", help="Optional dir to save labeled axes (u_pk/u_ck)")
    args = parser.parse_args()
    print(args)

    set_seed(args)
    main(args)
