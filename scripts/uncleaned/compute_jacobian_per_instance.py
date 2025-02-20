"""
compute_jacobian_per_instance.py

A script for computing Jacobians of different deformations of an instance.
"""

import dataclasses
from dataclasses import asdict, dataclass, fields
from PIL import Image
from pathlib import Path
from typing import Any, Literal

import igl
from jaxtyping import jaxtyped
import json
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typeguard import typechecked
import tyro
import wandb
import yaml

from src.dataset.deform4d import Deform4D
from src.dataset.smpl import SMPL

from src.geometry.poisson_system import PoissonSystem
from src.utils.geometry_utils import (
    load_obj,
)
from src.utils.random_utils import seed_everything


@dataclass
class Args:

    train_dset_cfg_file: Path
    """Path to the train dataset config file"""
    test_dset_cfg_file: Path
    """Path to the test dataset config file"""
    dset_type: Literal["deform4d", "smpl"]
    """Type of the dataset"""
    data_out_dir: Path
    """Directory to place processed data"""
    cfg_out_dir: Path
    """Directory to place config files"""

    # Poisson system
    constraint_lambda: float = 1e5
    """Constraint coefficient for linear system"""
    dnet_width: int = 256
    """Width of DiffusionNet. TODO: Drop DiffusionNet from Poisson System"""
    proj_to_local: bool = False
    """A flag to enable projection of transform matrices to local bases."""

    device_type: Literal["cpu", "cuda"] = "cuda"
    """Device to use"""
    seed: int = 2024
    """Random seed"""


@jaxtyped(typechecker=typechecked)
def main(args: Args) -> None:

    # Reproducibility
    seed_everything(args.seed)

    # Create output directory
    args.data_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data output directory: {str(args.data_out_dir.resolve())}")

    args.cfg_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Config output directory: {str(args.cfg_out_dir.resolve())}")

    # Set the device to use
    device = torch.device(args.device_type)

    # Initialize dataset
    train_dset, test_dset = init_dataset(args, device)

    # Initialize Poisson system
    # NOTE: This is required for Jacobian computation
    poisson = init_poisson(args, train_dset, device)

    # Load dataset config files
    with open(args.train_dset_cfg_file, "r") as f:
        train_dset_cfg = json.load(f)
    with open(args.test_dset_cfg_file, "r") as f:
        test_dset_cfg = json.load(f)

    # Process training set
    for i in range(len(train_dset)):
        # Parse each sample in the dataset
        if args.dset_type == "deform4d":
            # Parse geometry data
            v_src = train_dset[i]["v_src"]
            f_src = train_dset[i]["f_src"]
            v_tgt = train_dset[i]["v_tgt"]
            f_tgt = train_dset[i]["f_tgt"]
            anchor_inds = train_dset[i]["handle_inds"]
            anchor_pos = v_tgt[anchor_inds.tolist(), :]

            # Parse metadata
            src_motion_name = train_dset[i]["src_motion_name"]
            src_motion_idx = train_dset[i]["src_motion_idx"]
            tgt_motion_name = train_dset[i]["tgt_motion_name"]
            tgt_motion_idx = train_dset[i]["tgt_motion_idx"]
        elif args.dset_type == "smpl":
            handle, anchor, tgt = train_dset[i]
            anchor_inds, anchor_pos = anchor
            v_tgt, f_tgt, vc_tgt = tgt["v"], tgt["f"], tgt["vc"]
        else:
            raise ValueError(f"Unknown dataset type: {args.dset_type}")

        # Compute Jacobian
        J_tgt = poisson.compute_per_triangle_jacobian(v_tgt)
        J_tgt = J_tgt.cpu().numpy()

        # Save Jacobian
        out_dir = args.data_out_dir / tgt_motion_name / tgt_motion_idx
        out_dir.mkdir(parents=True, exist_ok=True)
        J_file = out_dir / "J_tgt.npy"
        np.save(J_file, J_tgt)

        # Update current item of dataset config
        curr_cfg_item = train_dset_cfg[i]
        src_motion_idx_ = curr_cfg_item["src"].strip().split("/")[-2]
        tgt_motion_idx_ = curr_cfg_item["tgt"].strip().split("/")[-2]
        assert curr_cfg_item["src_motion"] == src_motion_name.split("_")[1], (
            f"Motion name mismatch: {curr_cfg_item['src_motion']} != {src_motion_name}"
        )
        assert curr_cfg_item["tgt_motion"] == tgt_motion_name.split("_")[1], (
            f"Motion name mismatch: {curr_cfg_item['tgt_motion']} != {tgt_motion_name}"
        )
        assert src_motion_idx_ == src_motion_idx, (
            f"Motion index mismatch: {src_motion_idx_} != {src_motion_idx}"
        )
        assert tgt_motion_idx_ == tgt_motion_idx, (
            f"Motion index mismatch: {tgt_motion_idx_} != {tgt_motion_idx}"
        )
        curr_cfg_item["J_file"] = str(J_file)
        train_dset_cfg[i] = curr_cfg_item

    # Process test set
    for i in range(len(test_dset)):

        # Parse each sample in the dataset
        if args.dset_type == "deform4d":
            # Parse geometry data
            v_src = test_dset[i]["v_src"]
            f_src = test_dset[i]["f_src"]
            v_tgt = test_dset[i]["v_tgt"]
            f_tgt = test_dset[i]["f_tgt"]
            anchor_inds = test_dset[i]["handle_inds"]
            anchor_pos = v_tgt[anchor_inds.tolist(), :]

            # Parse metadata
            src_motion_name = test_dset[i]["src_motion_name"]
            src_motion_idx = test_dset[i]["src_motion_idx"]
            tgt_motion_name = test_dset[i]["tgt_motion_name"]
            tgt_motion_idx = test_dset[i]["tgt_motion_idx"]
        elif args.dset_type == "smpl":
            handle, anchor, tgt = test_dset[i]
            anchor_inds, anchor_pos = anchor
            v_tgt, f_tgt, vc_tgt = tgt["v"], tgt["f"], tgt["vc"]
        else:
            raise ValueError(f"Unknown dataset type: {args.dset_type}")

        # Compute Jacobian
        J_tgt = poisson.compute_per_triangle_jacobian(v_tgt)
        J_tgt = J_tgt.cpu().numpy()

        # Save Jacobian
        out_dir = args.data_out_dir / tgt_motion_name / tgt_motion_idx
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "J_tgt.npy", J_tgt)

        # Update current item of dataset config
        curr_cfg_item = test_dset_cfg[i]
        src_motion_idx_ = curr_cfg_item["src"].strip().split("/")[-2]
        tgt_motion_idx_ = curr_cfg_item["tgt"].strip().split("/")[-2]
        assert curr_cfg_item["src_motion"] == src_motion_name.split("_")[1], (
            f"Motion name mismatch: {curr_cfg_item['src_motion']} != {src_motion_name}"
        )
        assert curr_cfg_item["tgt_motion"] == tgt_motion_name.split("_")[1], (
            f"Motion name mismatch: {curr_cfg_item['tgt_motion']} != {tgt_motion_name}"
        )
        assert src_motion_idx_ == src_motion_idx, (
            f"Motion index mismatch: {src_motion_idx_} != {src_motion_idx}"
        )
        assert tgt_motion_idx_ == tgt_motion_idx, (
            f"Motion index mismatch: {tgt_motion_idx_} != {tgt_motion_idx}"
        )
        curr_cfg_item["J_file"] = str(J_file)
        test_dset_cfg[i] = curr_cfg_item

    # Save updated config files
    if args.dset_type == "deform4d":
        instance_name = train_dset_cfg[0]["name"]
        instance_name_ = test_dset_cfg[0]["name"]
        assert instance_name == instance_name_, (
            f"Instance name mismatch: {instance_name} != {instance_name_}"
        )
        out_train_cfg_file = args.cfg_out_dir / f"{instance_name}_train.json"
        out_test_cfg_file = args.cfg_out_dir / f"{instance_name}_test.json"

        assert out_train_cfg_file.resolve() != args.train_dset_cfg_file.resolve(), (
            f"Output train config file is the same as input train config file"
        )
        assert out_test_cfg_file.resolve() != args.test_dset_cfg_file.resolve(), (
            f"Output test config file is the same as input test config file"
        )
        with open(out_train_cfg_file, "w") as f:
            json.dump(train_dset_cfg, f, indent=4)
        with open(out_test_cfg_file, "w") as f:
            json.dump(test_dset_cfg, f, indent=4)

    elif args.dset_type == "smpl":
        raise NotImplementedError("SMPL dataset type not supported yet.")
    else:
        raise ValueError(f"Unknown dataset type: {args.dset_type}")

    print("[!] Done")


@jaxtyped(typechecker=typechecked)
def init_dataset(
    args, device: torch.device
) -> Any:
    """
    Initializes the dataset.
    """
    print(f"Dataset type: {str(args.dset_type)}")
    if args.dset_type == "deform4d":
        train_dset = Deform4D(
            args.train_dset_cfg_file,
            device,
        )
        test_dset = Deform4D(
            args.test_dset_cfg_file,
            device,
        )
    elif args.dset_type == "smpl":
        train_dset = SMPL(
            args.train_dset_cfg_file,
            device,
        )
        test_dset = SMPL(
            args.test_dset_cfg_file,
            device,
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dset_type}")
    assert len(train_dset) > 0, "Train dataset is empty."
    assert len(test_dset) > 0, "Test dataset is empty."
    print("Initialized datasets.")
    print(f"Train size: {len(train_dset)}")
    print(f"Test size: {len(test_dset)}")

    return train_dset, test_dset

@jaxtyped(typechecker=typechecked)
def init_poisson(
    args, dset: Dataset, device: torch.device
) -> PoissonSystem:
    """
    Initializes the Poisson system.

    TODO: Extend to the case with multiple templates.
    """
    if args.dset_type == "deform4d":
        assert isinstance(dset, Deform4D), (
            f"Expected Deform4D dataset, got {type(dset)}"
        )

        key = list(dset.tmpl_files.keys())[0]
        tmpl_file = Path(dset.tmpl_files[key][0])
        tmpl_anchor = dset.tmpl_anchors[key]
        assert tmpl_file.exists(), (
            f"Template file {str(tmpl_file)} does not exist."
        )
        print(
            "TODO: Support multiple templates in a dataset by creating multiple Poisson system"
        )
        (
            v_tmpl, f_tmpl,
            vcs_tmpl, uvs_tmpl, vns_tmpl,
            tex_inds_tmpl, vn_inds_tmpl, tex_tmpl,
        ) = load_obj(tmpl_file)
        v_tmpl = torch.from_numpy(v_tmpl).to(device)
        f_tmpl = torch.from_numpy(f_tmpl).to(device)

        poisson = PoissonSystem(
            v_tmpl, f_tmpl,
            device=device,
            anchor_inds=tmpl_anchor,
            constraint_lambda=args.constraint_lambda,
            train_J=False,  # NOTE: We are not using Poisson solve in this experiment
            dnet_width=args.dnet_width,
            proj_to_local=args.proj_to_local,
        )

    elif args.dset_type == "smpl":
        assert isinstance(dset, SMPL), (
            f"Expected SMPL dataset, got {type(dset)}"
        )
        tmpl = dset.template
        (
            v_tmpl, f_tmpl, vc_tmpl, wks_tmpl
        ) = tmpl["v"], tmpl["f"], tmpl["vc"], tmpl["wks"]
        poisson = PoissonSystem(
            v_tmpl, f_tmpl,
            device=device,
            anchor_inds=dset.anchors[0][0],
            constraint_lambda=args.constraint_lambda,
            train_J=False,  # NOTE: We are not using Poisson solve in this experiment
        )
    print("Initialized Poisson system.")

    return poisson


if __name__ == "__main__":
    main(
        tyro.cli(Args)
    )
