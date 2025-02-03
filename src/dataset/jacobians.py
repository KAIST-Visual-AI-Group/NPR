"""
jacobians.py

A dataset for Jacobians of meshes.
"""

from pathlib import Path
from typing import Dict, List, Union

import igl
from jaxtyping import Shaped, jaxtyped
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typeguard import typechecked


class Jacobians(Dataset):

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
        """
        Constructs a dataset.
        """

        with open(self.cfg_file, "r") as f:
            self.deform_pairs = json.load(f)
        cfg_name = self.cfg_file.stem

        # Collect templates of instances
        self._collect_templates()

        # Set path to template
        self.tmpl_files = {}
        self.tmpl_anchors = {}
        self.tmpl_files[cfg_name] = [str(self.deform_pairs[0]["src"])]
        self.tmpl_anchors[cfg_name] = torch.tensor(
            self.deform_pairs[0]["fps_inds"],
            device=self.device,
        ).type(torch.int32)

        print("Built dataset.")

    @jaxtyped(typechecker=typechecked)
    def _collect_templates(self) -> None:
        """
        Collects templates of instances.
        """
        self.templates = {}
        for pair in self.deform_pairs:
            name = pair["name"]
            if not name in self.templates:
                # Load mesh
                src_mesh_file = Path(pair["src"])
                assert src_mesh_file.exists(), (
                    f"Source mesh file {str(src_mesh_file)} does not exist."
                )
                v, f = igl.read_triangle_mesh(str(src_mesh_file))

                # Load handles
                handle_inds = torch.tensor(
                    pair["fps_inds"],
                    dtype=torch.int32,
                    device=self.device,
                )

                # Register the template of the current instance
                self.templates[name] = {
                    "v": torch.from_numpy(v).to(self.device),
                    "f": torch.from_numpy(f).to(self.device),
                    "handle_inds": handle_inds,
                }

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
        # Identify the pair
        pair = self.deform_pairs[idx]
        src_mesh_file, tgt_mesh_file = Path(pair["src"]), Path(pair["tgt"])
        assert src_mesh_file.exists(), (
            f"Source mesh file {str(src_mesh_file)} does not exist."
        )
        assert tgt_mesh_file.exists(), (
            f"Target mesh file {str(tgt_mesh_file)} does not exist."
        )

        # Retrieve instance name
        instance_name = str(pair["name"])

        # Retrieve the name and index of source and target motions
        src_motion_name = str(src_mesh_file.parents[1].stem)
        src_motion_idx = str(src_mesh_file.parents[0].stem)
        tgt_motion_name = str(tgt_mesh_file.parents[1].stem)
        tgt_motion_idx = str(tgt_mesh_file.parents[0].stem)

        # Load meshes
        v_src, f_src = igl.read_triangle_mesh(str(src_mesh_file))
        v_tgt, f_tgt = igl.read_triangle_mesh(str(tgt_mesh_file))
        assert v_src.shape[0] > 0, f"Failed to load {str(src_mesh_file)}"
        assert f_src.shape[0] > 0, f"Failed to load {str(src_mesh_file)}"
        assert v_tgt.shape[0] > 0, f"Failed to load {str(tgt_mesh_file)}"
        assert f_tgt.shape[0] > 0, f"Failed to load {str(tgt_mesh_file)}"
        v_src = torch.from_numpy(v_src).to(self.device).float()
        f_src = torch.from_numpy(f_src).to(self.device).long()
        v_tgt = torch.from_numpy(v_tgt).to(self.device).float()
        f_tgt = torch.from_numpy(f_tgt).to(self.device).long()

        # Load Jacobian
        J_file = Path(pair["J_file"])
        assert J_file.exists(), (
            f"Jacobian file {J_file} does not exist."
        )
        J = torch.from_numpy(np.load(J_file)).to(self.device)

        # Retrieve the handle indices
        handle_inds = torch.tensor(
            pair["fps_inds"],
            device=self.device,
            dtype=torch.int32,
        )

        sample_dict = {
            "J": J,
            "instance_name": instance_name,
            "src_motion_name": src_motion_name,
            "src_motion_idx": src_motion_idx,
            "tgt_motion_name": tgt_motion_name,
            "tgt_motion_idx": tgt_motion_idx,
            "v_src": v_src,
            "f_src": f_src,
            "v_tgt": v_tgt,
            "f_tgt": f_tgt,
            "handle_inds": handle_inds,
        }

        # Load source surface points if available
        src_srf_file = src_mesh_file.parent / "surface_points.npz"
        if src_srf_file.exists():
            # flow_dict = np.load(path)
            # points = flow_dict['points'].astype(np.float32)
            # normals = flow_dict['normals'].astype(np.float32)
            src_srf_dict = np.load(src_srf_file)
            src_srf_points = torch.from_numpy(
                src_srf_dict["points"]
            ).to(self.device)
            src_srf_normals = torch.from_numpy(
                src_srf_dict["normals"]
            ).to(self.device)

            sample_dict["src_srf_points"] = src_srf_points
            sample_dict["src_srf_normals"] = src_srf_normals

        return sample_dict
