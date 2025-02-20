"""
deform4d.py

A dataset for DeformingThings4D dataset.
"""

from pathlib import Path
from typing import Dict, List, Union

import igl
from jaxtyping import Shaped, jaxtyped
import json
import torch
from torch.utils.data import Dataset
from typeguard import typechecked


class Deform4D(Dataset):

    cfg_file: Path
    """Path to the dataset config file."""
    device: torch.device
    """Device to load data."""

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        cfg_file: Path,
        device: torch.device,
    ) -> None:
        super().__init__()
        
        # load data in the config file
        self.cfg_file = cfg_file
        assert self.cfg_file.exists(), (
            f"Config file {self.cfg_file} does not exist."
        )
        assert self.cfg_file.suffix == ".json", (
            f"Config file {self.cfg_file} is not a JSON file."
        )

        # set the device        
        self.device = device

        # build the dataset
        self._build_dataset()

    @jaxtyped(typechecker=typechecked)
    def _build_dataset(self) -> None:

        with open(self.cfg_file, "r") as f:
            self.deform_pairs = json.load(f)
        cfg_name = self.cfg_file.stem

        # TODO: clean up this part
        self.tmpl_files = {}
        self.tmpl_anchors = {}
        self.tmpl_files[cfg_name] = [str(self.deform_pairs[0]["src"])]
        self.tmpl_anchors[cfg_name] = torch.tensor(
            self.deform_pairs[0]["fps_inds"],
            device=self.device,
        ).type(torch.int32)

        print("Built dataset.")

    @jaxtyped(typechecker=typechecked)
    def __len__(self) -> int:
        return len(self.deform_pairs)

    @jaxtyped(typechecker=typechecked)
    def __getitem__(
        self, idx: int
    ) -> Dict[str, Union[str, Shaped[torch.Tensor, "..."]]]:
        """
        Retrieves a data sample from the dataset.
        """
        pair = self.deform_pairs[idx]
        src_mesh_file, tgt_mesh_file = Path(pair["src"]), Path(pair["tgt"])
        assert src_mesh_file.exists(), (
            f"Source mesh file {str(src_mesh_file)} does not exist."
        )
        assert tgt_mesh_file.exists(), (
            f"Target mesh file {str(tgt_mesh_file)} does not exist."
        )
        handle_inds = pair["fps_inds"]

        # Retrieve the name and index of source and target motions
        src_motion_name = str(src_mesh_file.parents[1].stem)
        src_motion_idx = str(src_mesh_file.parents[0].stem)
        tgt_motion_name = str(tgt_mesh_file.parents[1].stem)
        tgt_motion_idx = str(tgt_mesh_file.parents[0].stem)

        # Read mesh
        # By now, the consistency of topology is guaranteed
        # by the preprocessing script
        v_src, f_src = igl.read_triangle_mesh(str(src_mesh_file))
        v_tgt, f_tgt = igl.read_triangle_mesh(str(tgt_mesh_file))
        v_src = torch.from_numpy(v_src).to(self.device).float()
        f_src = torch.from_numpy(f_src).to(self.device).long()
        v_tgt = torch.from_numpy(v_tgt).to(self.device).float()
        f_tgt = torch.from_numpy(f_tgt).to(self.device).long()
        handle_inds = torch.tensor(handle_inds).to(self.device).long()

        # Compile the list into a dictionary
        sample_dict = {
            "v_src": v_src,
            "f_src": f_src,
            "v_tgt": v_tgt,
            "f_tgt": f_tgt,
            "handle_inds": handle_inds,
            "src_motion_name": src_motion_name,
            "src_motion_idx": src_motion_idx,
            "tgt_motion_name": tgt_motion_name,
            "tgt_motion_idx": tgt_motion_idx,
        }
        
        return sample_dict
