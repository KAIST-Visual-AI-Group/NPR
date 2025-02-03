"""
train.py

A script for learning pose representation from a collection of shape examples.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Literal

import igl
from jaxtyping import jaxtyped
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typeguard import typechecked
import tyro
import yaml

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
    out_dir: Path = Path("outputs/train_ae")
    """Output directory."""
    wandb_grp: str = "train_ae"
    """WandB group name."""

    device_type: Literal["cpu", "cuda"] = "cuda"
    """Device type."""
    dset_type: Literal["deform4d", "smpl"] = "deform4d"
    """Type of the dataset."""
    use_poisson: bool = True
    """Use Poisson system for training."""
    seed: int = 2024
    """Random seed"""

    lr: float = 1e-4
    """Learning rate."""
    batch_size: int = 1
    """Batch size."""
    shuffle: bool = True
    """Shuffle the dataset."""
    n_epoch: int = 300
    """Number of epochs."""
    test_every: int = 10
    """Number of epochs between testing"""
    n_test_sample: int = 1
    """Number of test examples"""

    save_every: int = 10
    """Number of epochs between checkpoints"""


@jaxtyped(typechecker=typechecked)
def main(args: Args) -> None:

    # Reproducibility
    seed_everything(args.seed)

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {str(args.out_dir.resolve())}")

    # Set the device to use
    device = torch.device(args.device_type)

    # Save the command
    with open(args.out_dir / "cmd.txt", mode="w") as file:
        file.write(" ".join(sys.argv))

    # Save the config
    with open(args.out_dir / "config.yaml", mode="w") as file:
        yaml.dump(asdict(args), file)

    # Initialize dataset
    train_dset = Jacobians(args.train_dset_cfg_file, device)
    test_dset = Jacobians(args.test_dset_cfg_file, device)
    train_loader = DataLoader(
        train_dset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
    )
    print(f"# train: {len(train_dset)}")
    print(f"# test: {len(test_dset)}")

    # Loader for testing the same category
    test_loader0 = DataLoader(
        train_dset,
        batch_size=args.n_test_sample,
        shuffle=args.shuffle,
        drop_last=True,
    )

    # Loader for testing the other category
    test_loader = DataLoader(
        test_dset,
        batch_size=args.n_test_sample,
        shuffle=args.shuffle,
        drop_last=True,
    )
    
    print(f"Initialized dataset and loader")

    # Initialize model
    model = Deformation_Networks().to(device)
    print("Initialized model")

    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )
    print("Initialized optimizer")

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

    # Initialize Tensorboard
    tb_dir = args.out_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    # Check if checkpoint directory exists
    start_epoch = 0
    ckpt_dir = args.out_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpt_files = sorted(ckpt_dir.glob("*.pth"))
        if len(ckpt_files) > 0:
            ckpt_file = ckpt_files[-1]
            ckpt = torch.load(ckpt_file)
            model.load_state_dict(ckpt["model"])
            optim.load_state_dict(ckpt["optim"])
            start_epoch = ckpt["epoch"]
            print(f"Loaded checkpoint {str(ckpt_file)}")

    pbar = tqdm(range(start_epoch, args.n_epoch), leave=True)
    for epoch_idx in pbar:

        # Train one epoch
        inner_pbar = tqdm(train_loader, leave=True)
        for train_idx, train_batch in enumerate(inner_pbar):

            glob_idx = epoch_idx * len(train_loader) + train_idx

            # Prepare network input
            enc_in = train_batch["J"].type(torch.float32)
            enc_in_v = train_batch["v_tgt"].type(torch.float32)
            B = enc_in.shape[0]
            N = enc_in.shape[1]
            enc_in = enc_in.reshape(B, N, -1)

            v_tgts = train_batch["v_tgt"].type(torch.float32)
            f_tgts = train_batch["f_tgt"].type(torch.float32)
            fc_tgt = compute_centroids(v_tgts, f_tgts)                
            enc_in = torch.cat([fc_tgt, enc_in], dim=-1)

            v_srcs = train_batch["v_src"].type(torch.float32)
            f_srcs = train_batch["f_src"].type(torch.float32)
            query_pts = compute_centroids(v_srcs, f_srcs)

            # Forward pass
            dec_out = model.decode(model.encode(enc_in), query_pts)

            # Compute loss
            if args.use_poisson:  # NOTE: Compute reconstruction loss in vertex space
                assert args.batch_size == 1, "Batch size must be 1 for Poisson system"
                train_poisson.J = dec_out[0, ...].reshape(-1, 3, 3)
                dec_out, _ = train_poisson.get_current_mesh(
                    enc_in_v[0, 0:1, ...],
                )
                loss = F.mse_loss(dec_out[None], enc_in_v)
            else:
                loss = F.mse_loss(dec_out, enc_in[..., 3:])

            # Backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Log current loss
            tb_writer.add_scalar("train/loss", loss.item(), glob_idx)
            pbar.set_description(f"Loss: {loss.item():.5f}")

        # Test
        if (epoch_idx + 1) % args.test_every == 0:        

            test_same_batch = next(iter(test_loader0))
            test_diff_batch = next(iter(test_loader))
            
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
            diff_enc_in = torch.cat([diff_fc_tgt, diff_enc_in], dim=-1)
            
            diff_v_srcs = test_diff_batch["v_src"].type(torch.float32)
            diff_f_srcs = test_diff_batch["f_src"].type(torch.float32)
            diff_query_pts = compute_centroids(diff_v_srcs, diff_f_srcs)
            # ===========================================================

            # Forward pass
            with torch.no_grad():
                same_dec_out = model.decode(model.encode(same_enc_in), same_query_pts)

            # Log reconstruction loss
            same_loss = F.mse_loss(same_dec_out, same_enc_in[..., 3:]).item()
            tb_writer.add_scalar("test/same_loss", same_loss, epoch_idx)

        # Save checkpoint
        if (epoch_idx + 1) % args.save_every == 0:
            ckpt_dir = args.out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "epoch": epoch_idx,
            }
            torch.save(
                ckpt,
                ckpt_dir / f"ckpt_epoch-{epoch_idx:06d}.pth"
            )


if __name__ == "__main__":
    main(
        tyro.cli(Args)
    )
