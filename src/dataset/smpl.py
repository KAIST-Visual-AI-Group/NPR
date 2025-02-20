"""
smpl.py

A dataset of SMPL template-target pairs.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

from jaxtyping import Int, Float, Shaped, jaxtyped
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from typeguard import typechecked

from ..utils.geometry_utils import load_obj


class SMPL(Dataset):

    cfg_file: Path
    """Path to the dataset config file."""
    device: torch.device
    """The device to use for the dataset."""

    @jaxtyped(typechecker=typechecked)
    def __init__(self, cfg_file: Path, device: torch.device) -> None:
        super().__init__()
        
        # load data in the config file
        self.cfg_file = cfg_file
        assert self.cfg_file.exists(), (
            f"Config file {self.cfg_file} does not exist."
        )

        # set the device        
        self.device = device

        # build the dataset
        self._build_dataset()

    @jaxtyped(typechecker=typechecked)
    def _build_dataset(self) -> None:

        self.tmpl_files = []
        self.handle_files = []
        self.anchor_files = []
        self.tgt_files = []
        # self.fps_ind_files = []

        self.handles = []
        self.anchors = []
        self.targets = []
        # self.fps_inds = []

        # Load the config and identify files to load
        with open(self.cfg_file, "r") as f:
            lines = f.readlines()
            for l in lines:
                ####
                # tmpl_file, handle_file, anchor_file, tgt_file, fps_ind_file = l.strip().split(",")
                tmpl_file, handle_file, anchor_file, tgt_file, _ = l.strip().split(",")
                ####
                tmpl_file = Path(tmpl_file.strip())
                handle_file = Path(handle_file.strip())
                anchor_file = Path(anchor_file.strip())
                tgt_file = Path(tgt_file.strip())
                # fps_ind_file = Path(fps_ind_file.strip())

                assert tmpl_file.exists(), f"File {tmpl_file} does not exist."
                assert handle_file.exists(), f"File {handle_file} does not exist."
                assert anchor_file.exists(), f"File {anchor_file} does not exist."
                assert tgt_file.exists(), f"File {tgt_file} does not exist."
                # assert fps_ind_file.exists(), f"File {fps_ind_file} does not exist."
                
                self.tmpl_files.append(tmpl_file)
                self.handle_files.append(handle_file)
                self.anchor_files.append(anchor_file)
                self.tgt_files.append(tgt_file)
                # self.fps_ind_files.append(fps_ind_file)
        
        # Load template mesh
        assert len(set(self.tmpl_files)) == 1, "All template files must be the same."
        tmpl_file = self.tmpl_files[0]
        sample_pts_file = tmpl_file.parent / "sample_1024_pts.npy"
        assert sample_pts_file.exists(), (
            f"Sample points file {sample_pts_file} does not exist."
        )
        (
            v, f, vc, uvs, vns, tex_inds, vn_inds, tex
        ) = load_obj(tmpl_file)
        sampled_pts = np.load(sample_pts_file)
        wks_file = tmpl_file.parent / "wks_top-k-eig-100_T-100.npy"
        if wks_file.exists():
            wks = np.load(wks_file).astype(np.float32)
        assert wks.shape[0] == v.shape[0], (
            f"Number of WKS values {wks.shape[0]} does not match the number of vertices {v.shape[0]}."
        )
        self.template = {
            "v": torch.from_numpy(v).to(self.device),
            "f": torch.from_numpy(f).to(self.device),
            "vc": torch.from_numpy(vc).to(self.device),
            "sampled_pts": torch.from_numpy(sampled_pts).to(self.device),
            "wks": torch.from_numpy(wks).to(self.device),
        }

        # Load data
        print("Loading handles...")
        for file in tqdm(self.handle_files):  # Handles
            inds = []
            pos = []
            with open(file, "r") as f:
                lines = f.readlines()
                for l in lines:
                    ind, x, y, z = l.strip().split()
                    inds.append(int(ind))
                    pos.append([float(x), float(y), float(z)])
            inds = torch.tensor(inds, device=self.device)
            pos = torch.tensor(pos, device=self.device)
            self.handles.append((inds, pos))
        
        print("Loading anchors...")
        for file in tqdm(self.anchor_files):  # Anchors
            inds = []
            pos = []
            with open(file, "r") as f:
                lines = f.readlines()
                for l in lines:
                    ind, x, y, z = l.strip().split()
                    inds.append(int(ind))
                    pos.append([float(x), float(y), float(z)])
            inds = torch.tensor(inds, device=self.device)
            pos = torch.tensor(pos, device=self.device)
            self.anchors.append((inds, pos))

        print("Loading target meshes...")
        for file in tqdm(self.tgt_files):  # Target meshes
            (
                v, f, vc, uvs, vns, tex_inds, vn_inds, tex
            ) = load_obj(file)
            v = torch.from_numpy(v).to(self.device)
            f = torch.from_numpy(f).to(self.device)
            vc = torch.from_numpy(vc).to(self.device)
            # TODO: Hold texture data
            self.targets.append(
                {
                    "v": v,
                    "f": f,
                    "vc": vc,
                }
            )

        assert len(self.handles) == len(self.anchors) == len(self.targets), (
            f"Number of data: {len(self.handles)}, {len(self.anchors)}, {len(self.targets)}"
        )

        print(f"Number of samples: {len(self.tmpl_files)}")

    @jaxtyped(typechecker=typechecked)
    def __len__(self) -> int:
        return len(self.targets)
    
    @jaxtyped(typechecker=typechecked)
    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[
        Tuple[Int[Tensor, "H"], Float[Tensor, "H 3"]],
        Tuple[Int[Tensor, "A"], Float[Tensor, "A 3"]],
        # Tuple[Int[Tensor, "FPS"], Float[Tensor, "FPS 3"]],
        Dict[str, Tensor],
    ]:
        """
        Retrieves a sample from the dataset by the given index.
        
        Args:
            idx: The index of the sample to retrieve.
        
        Returns:
            handle: A tuple of handle indices and positions.
            anchor: A tuple of anchor indices and positions.
            fps_inds: A dictionary of FPS indices and positions.
            target: A tuple of target mesh vertices and faces.
        """
        handle = self.handles[idx]
        anchor = self.anchors[idx]
        ####
        # fps = self.fps_inds[idx]
        ####
        target = self.targets[idx]

        ####
        # return (handle, anchor, fps, target)
        return (handle, anchor, target)
        ####