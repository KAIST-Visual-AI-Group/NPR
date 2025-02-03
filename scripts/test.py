"""
test.py

A script for transferring poses using learned pose representations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import igl
from jaxtyping import jaxtyped
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typeguard import typechecked
import tyro

from src.dataset.jacobians import Jacobians
from src.geometry.poisson_system import PoissonSystem
from src.networks.model import Deformation_Networks
from src.utils.geometry_utils import compute_centroids
from src.utils.random_utils import seed_everything


@dataclass
class Args:

    train_dset_cfg_file: Path
    """Path to the train dataset config file."""
    test_dset_cfg_file: Path
    """Path to the test dataset config file."""
    exp_dir: Path
    """Path to experiment directory."""
    tag: Optional[str] = None
    """Tag for the experiment."""

    device_type: Literal["cpu", "cuda"] = "cuda"
    """Device type."""
    seed: int = 2024
    """Random seed"""

@jaxtyped(typechecker=typechecked)
def main(args: Args) -> None:

    # Reproducibility
    seed_everything(args.seed)

    device = torch.device(args.device_type)

    # Locate experiment directory and latest checkpoints
    exp_dir = args.exp_dir
    assert exp_dir.exists(), f"Directory {str(args.exp_dir)} does not exist"
    ckpt_dir = exp_dir / "checkpoints"
    assert ckpt_dir.exists(), f"Directory {str(ckpt_dir)} does not exist"
    ckpt_file = sorted(list(ckpt_dir.glob("*.pth")))[-1]
    print(f"Checkpoint file: {str(ckpt_file)}")
    ckpt = torch.load(ckpt_file)

    # Initialize model
    model = Deformation_Networks().to(device)
    print("Initialized model")

    # Load checkpoint
    model.load_state_dict(ckpt["model"])
    print("Loaded checkpoint")

    # ===============================================================================
    # Load dataset
    train_dset = Jacobians(args.train_dset_cfg_file, device)
    test_dset = Jacobians(args.test_dset_cfg_file, device)
    train_loader = DataLoader(
        train_dset,
        batch_size=1,
        shuffle=False,
    )
    print(f"# train: {len(train_dset)}")
    print(f"# test: {len(test_dset)}")

    train_instance_name = train_dset[0]["instance_name"]
    test_instance_name = test_dset[0]["instance_name"]

    # Loader for testing the same category
    test_loader0 = DataLoader(
        train_dset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    # Loader for testing the other category
    test_loader = DataLoader(
        test_dset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    # ===============================================================================

    # ===============================================================================
    # Initialize Poisson system for training
    train_key = list(train_dset.tmpl_files.keys())[0]
    train_tmpl_file = Path(train_dset.tmpl_files[train_key][0])
    assert train_tmpl_file.exists(), (
        f"Template file {train_tmpl_file} does not exist"
    )
    (
        train_v_tmpl, train_f_tmpl
    ) = igl.read_triangle_mesh(str(train_tmpl_file))
    train_poisson = PoissonSystem(
        torch.from_numpy(train_v_tmpl).to(device),
        torch.from_numpy(train_f_tmpl).to(device),
        device=device,
        anchor_inds=torch.tensor([0], dtype=torch.int32).to(device),
    )
    train_tmpl_J = train_poisson.J.clone()
    print("Initialized train poisson")

    # Initialize Poisson system for testing
    test_key = list(test_dset.tmpl_files.keys())[0]
    test_tmpl_file = Path(test_dset.tmpl_files[test_key][0])
    assert test_tmpl_file.exists(), (
        f"Template file {test_tmpl_file} does not exist"
    )
    (
        test_v_tmpl, test_f_tmpl
    ) = igl.read_triangle_mesh(str(test_tmpl_file))

    test_poisson = PoissonSystem(
        torch.from_numpy(test_v_tmpl).to(device),
        torch.from_numpy(test_f_tmpl).to(device),
        device=device,
        anchor_inds=torch.tensor([0], dtype=torch.int32).to(device),
    )
    test_tmpl_J = test_poisson.J.clone()
    print("Initialized test poisson")
    # ===============================================================================

    out_dir = args.exp_dir / "test"
    if not args.tag is None:
        out_dir = out_dir / f"{args.tag}"
    else:
        out_dir = out_dir / "default_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {str(out_dir)}")

    for batch_idx, (test_same_batch, test_diff_batch) in tqdm(
        enumerate(zip(test_loader0, test_loader))
    ):
        # Prepare network input
        # ===========================================================
        same_enc_in = test_same_batch["J"].type(torch.float32)
        B = same_enc_in.shape[0]
        N = same_enc_in.shape[1]
        same_enc_in = same_enc_in.reshape(B, N, -1)

        same_v_tgts = test_same_batch["v_tgt"].type(torch.float32)
        same_f_tgts = test_same_batch["f_tgt"].type(torch.float32)
        same_fc_tgt = compute_centroids(same_v_tgts, same_f_tgts)                

        same_enc_in = torch.cat([same_fc_tgt, same_enc_in], dim=-1)
        
        same_v_srcs = test_same_batch["v_src"].type(torch.float32)
        same_f_srcs = test_same_batch["f_src"].type(torch.float32)
        same_query_pts = compute_centroids(same_v_srcs, same_f_srcs)
        # ===========================================================

        # ===========================================================
        diff_enc_in = test_diff_batch["J"].type(torch.float32)
        B = diff_enc_in.shape[0]
        N = diff_enc_in.shape[1]
        diff_enc_in = diff_enc_in.reshape(B, N, -1)

        diff_v_tgts = test_diff_batch["v_tgt"].type(torch.float32)
        diff_f_tgts = test_diff_batch["f_tgt"].type(torch.float32)
        diff_fc_tgt = compute_centroids(diff_v_tgts, diff_f_tgts)                
        
        diff_v_srcs = test_diff_batch["v_src"].type(torch.float32)
        diff_f_srcs = test_diff_batch["f_src"].type(torch.float32)
        diff_query_pts = compute_centroids(diff_v_srcs, diff_f_srcs)
        # ===========================================================

        # Forward pass
        with torch.no_grad():
            same_enc_out = model.encode(same_enc_in)
            diff_dec_out = model.decode(same_enc_out, diff_query_pts)
        mesh_dir = out_dir / "meshes"
        mesh_dir.mkdir(parents=True, exist_ok=True)

        # Poisson Solve: Jacobian Field -> Vertices
        diff_f_src = test_diff_batch["f_src"][0]
        diff_dec_out_ = diff_dec_out[0].clone().reshape(-1, 3, 3)
        test_poisson.J = diff_dec_out_
        diff_dec_out_, _ = test_poisson.get_current_mesh(
            torch.zeros((1, 3), device=device),
        )
        diff_dec_out_ = diff_dec_out_ - torch.mean(diff_dec_out_, dim=0, keepdim=True)

        # Save meshes
        igl.write_triangle_mesh(
            str(mesh_dir / f"out_{batch_idx:05d}.obj"),
            diff_dec_out_.cpu().numpy(),
            diff_f_src.cpu().numpy(),
        )


if __name__ == "__main__":
    main(
        tyro.cli(Args)
    )
