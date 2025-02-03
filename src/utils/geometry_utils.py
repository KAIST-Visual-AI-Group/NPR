"""
geometry_utils.py

Utility functions for geometry processing.
"""

from PIL import Image
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fpsample
import imageio
from jaxtyping import Int32, Float, Float32, Int32, Shaped, jaxtyped
import numpy as np
from numpy import ndarray
import scipy
import torch
import trimesh
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
def compute_centroids(
    v: Shaped[torch.Tensor, "B V 3"],
    f: Shaped[torch.Tensor, "B F 3"],
) -> Shaped[torch.Tensor, "B F 3"]:
    v_gather = v.unsqueeze(-1).expand(-1, -1, -1, 3)
    f_gather = f.unsqueeze(2).expand(-1, -1, v.shape[-1], -1)
    centroids = torch.gather(v_gather, 1, f_gather.type(torch.int64))
    centroids = torch.mean(centroids, dim=-1)
    return centroids

@jaxtyped(typechecker=typechecked)
def columnize_rotations(
    R: Shaped[torch.Tensor, "N 3 3"],
) -> Shaped[torch.Tensor, "Nx9"]:
    """
    Converts a batch of columnized rotation matrices to a batch of rotation matrices.
    """
    n_v = R.shape[0]

    R_col = torch.zeros((n_v * 9), dtype=R.dtype, device=R.device)
    R_col[:n_v] = R[:, 0, 0]
    R_col[n_v : 2 * n_v] = R[:, 1, 0]
    R_col[2 * n_v : 3 * n_v] = R[:, 2, 0]
    R_col[3 * n_v : 4 * n_v] = R[:, 0, 1]
    R_col[4 * n_v : 5 * n_v] = R[:, 1, 1]
    R_col[5 * n_v : 6 * n_v] = R[:, 2, 1]
    R_col[6 * n_v : 7 * n_v] = R[:, 0, 2]
    R_col[7 * n_v : 8 * n_v] = R[:, 1, 2]
    R_col[8 * n_v : 9 * n_v] = R[:, 2, 2]
    
    return R_col

@jaxtyped(typechecker=typechecked)
def fps_pointcloud(
    p: Shaped[np.ndarray, "N 3"],
    n_sample: int,
) -> Tuple[Shaped[np.ndarray, "N_S 3"], Int32[np.ndarray, "N_S"]]:
    """
    Samples a point cloud using farthest point sampling.
    """
    inds = fpsample.bucket_fps_kdline_sampling(
        p, n_sample, h=3
    ).astype(np.int32)
    pts = p[inds]

    return pts, inds

@jaxtyped(typechecker=typechecked)
def save_mesh(
    path: Path,
    v: Shaped[ndarray, "V 3"],
    f: Shaped[ndarray, "F 3"],
    uv: Optional[Shaped[ndarray, "V 2"]]=None,
    tex: Optional[Shaped[ndarray, "H W C"]]=None,
) -> None:
    # create mesh
    mesh = trimesh.Trimesh(vertices=v, faces=f)

    # handle texture
    if not tex is None:
        assert not uv is None, "UVs must be provided if texture exists."
        mesh.visual.uv = uv
        if tex.dtype == np.float32:
            tex = (tex * 255).astype(np.uint8)
        tex_img = Image.fromarray(tex)
        color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=tex_img)
        mesh.visual = color_visuals
    
    # write mesh
    mesh.export(str(path))

@jaxtyped(typechecker=typechecked)
def load_mesh(path: Path):
    assert path.exists(), f"No mesh file: {path}"

    mesh = trimesh.load(str(path), process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        print(path)
        assert len(mesh.geometry) == 1, f"Number of geometry: {len(mesh.geometry)}"
        mesh = mesh.geometry["geometry_0"]

    # load geometry
    v = np.array(mesh.vertices, dtype=np.float32)
    f = np.array(mesh.faces, dtype=np.int32)

    # load texture
    # NOTE: Flip texture image when loading
    # TODO: handle textureless meshes
    uvs = np.array(mesh.visual.uv, dtype=np.float32)

    material = mesh.visual.material
    if isinstance(material, trimesh.visual.material.SimpleMaterial):
        tex = np.array(material.image)
    elif isinstance(material, trimesh.visual.material.PBRMaterial):
        tex = np.array(material.baseColorTexture)
    else:
        raise NotImplementedError(f"Unsupported material: {type(material)}")
    tex = (tex / 255.0).astype(np.float32)
    tex = np.flip(tex, axis=0).copy()  # NOTE: resolve error converting array to tensor

    return v, f, uvs, tex

@jaxtyped(typechecker=typechecked)
def load_obj(path: Path):
    """
    Loads an .obj file.
    """
    assert path.suffix == ".obj", f"Not an .obj file. Got {path.suffix}"
    assert path.exists(), f"File not found: {str(path)}"

    v = []
    f = []
    vc = None
    uvs = []
    vns = []
    tex_inds = []
    vn_inds = []
    tex = None

    with open(path, "r") as file:

        for line in file.readlines():

            # ===================================================================
            # parse vertex coordinates
            if line.startswith("v "):
                vertices_ = _parse_vertex(line)
                v.append(vertices_)
            # ===================================================================

            # ===================================================================
            # parse faces
            elif line.startswith("f "):
                (
                    face_indices_,
                    tex_coord_indices_,
                    vertex_normal_indices_
                ) = _parse_face(line)
                f.append(face_indices_)
                tex_inds.append(tex_coord_indices_)
                vn_inds.append(vertex_normal_indices_)
            # ===================================================================

            # ===================================================================
            # parse texture coordinates
            elif line.startswith("vt "):
                tex_coordinates_ = _parse_tex_coordinates(line)
                uvs.append(tex_coordinates_)
            # ===================================================================

            # ===================================================================
            # parse vertex normals
            elif line.startswith("vn "):
                vertex_normals_ = _parse_vertex_normal(line)
                vns.append(vertex_normals_)
            # ===================================================================

            else:
                pass  # ignore
    v = np.array(v, dtype=np.float32)
    f = np.array(f, dtype=np.int32)

    # ==========================================================================
    # load texture
    tex = _load_texture_image(path)
    tex_exists = (
        len(uvs) > 0 \
        and tex_inds[0][0] is not None \
        and tex is not None
    )
    if tex_exists:
        uvs = np.array(uvs, dtype=np.float32)
        tex_inds = np.array(tex_inds, dtype=np.int32)
    else:
        uvs = None
        tex_inds = None
        vc = np.ones_like(v) * 0.75
    # ==========================================================================

    # ==========================================================================
    # load vertex normals
    if len(vns) > 0:
        vns = np.array(vns, dtype=np.float32)
    else:
        vns = None

    if vn_inds[0][0] is not None:
        vn_inds = np.array(vn_inds, dtype=np.int32)
    else:
        vn_inds = None
    # ==========================================================================
    
    return v, f, vc, uvs, vns, tex_inds, vn_inds, tex

@typechecked
def _parse_vertex(line: str) -> List[float]:
    coords = [float(x) for x in line.split()[1:]]
    coords = coords[:3]  # ignore the rest
    return coords

@typechecked
def _parse_face(
    line: str,
) -> Tuple[
    List[int],
    Union[List[int], List[None]],
    Union[List[int], List[None]],
]:
    """
    Parses a line starts with 'f' that contains face information.

    NOTE: face indices must be offset by 1 because OBJ files are 1-indexed.
    """

    space_splits = line.split()[1:]

    face_indices = []
    tex_coord_indices = []
    vertex_normal_indices = []

    for space_split in space_splits:
        slash_split = space_split.split("/")

        if len(slash_split) == 1:  # f v1 v2 v3 ...
            face_indices.append(int(slash_split[0]) - 1)
            tex_coord_indices.append(None)
            vertex_normal_indices.append(None)

        elif len(slash_split) == 2:  # f v1/vt1 v2/vt2 v3/vt3 ...
            face_indices.append(int(slash_split[0]) - 1)
            tex_coord_indices.append(int(slash_split[1]) - 1)
            vertex_normal_indices.append(None)

        elif len(slash_split) == 3:  # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 ...
            face_indices.append(int(slash_split[0]) - 1)
            tex_coord_index = None
            if slash_split[1].isnumeric():
                tex_coord_index = int(slash_split[1]) - 1
            tex_coord_indices.append(tex_coord_index)
            vertex_normal_indices.append(int(slash_split[2]) - 1)

        else:
            raise NotImplementedError("Unsupported feature")

    return (
        face_indices,
        tex_coord_indices,
        vertex_normal_indices,
    )

@typechecked
def _parse_tex_coordinates(line: str) -> List[float]:
    return [float(x) for x in line.split()[1:]]

@typechecked
def _parse_vertex_normal(line: str) -> List[float]:
    return [float(x) for x in line.split()[1:]]

@jaxtyped(typechecker=typechecked)
def _load_texture_image(
    obj_path: Path,
    vertical_flip: bool = True,
    default_bg_color: Float[ndarray, "3"] = np.zeros(3, dtype=np.float32),
) -> Optional[Float[np.ndarray, "image_height image_width 3"]]:
    """
    Loads the texture image associated with the given .obj file.

    Args:
        obj_path: Path to the .obj file whose texture is being loaded.
        vertical_flip: Whether to flip the texture image vertically.
            This is necessary for rendering systems following OpenGL conventions.
        default_bg_color: The default background color to use 
            if the loaded image has an alpha channel.
            
    Returns:
        A texture image if it exists, otherwise None.
    """
    img_path = obj_path.parent / f"{obj_path.stem}_texture.png"
    tex = None
    if img_path.exists():
        tex = imageio.imread(img_path)

        num_channel = tex.shape[-1]
        if num_channel == 4:  # RGBA
            tex, alpha = np.split(tex, [3], axis=2)
            alpha_mask = (alpha > 0.0).astype(np.float32)
        elif num_channel == 3:  # RGB
            alpha_mask = np.ones([*tex.shape[:2], 1], dtype=np.float32)
        else:
            raise AssertionError(f"Invalid texture image shape: {tex.shape}")

        tex = tex.astype(np.float32) / 255.0
        tex = alpha_mask * tex + \
            (1 - alpha_mask) * default_bg_color

        if vertical_flip:
            tex = np.flip(tex, axis=0).copy()
    return tex

@jaxtyped(typechecker=typechecked)
def save_obj(
    out_path: Path,
    v: Shaped[np.ndarray, "V 3"],
    f: Shaped[np.ndarray, "F 3"],
    vc: Optional[Shaped[np.ndarray, "* 3"]] = None,
    uvs: Optional[Shaped[np.ndarray, "..."]] = None,  # FIXME: Add type annotation
    vns: Optional[Shaped[np.ndarray, "* 3"]] = None,
    tex_inds: Optional[Shaped[np.ndarray, "..."]] = None,  # FIXME: Add type annotation
    vn_inds: Optional[Shaped[np.ndarray, "* 3"]] = None,
    tex: Optional[Shaped[np.ndarray, "..."]] = None,  # FIXME: Add type annotation
    flip_tex: bool = True,
) -> None:
    """
    Saves a triangle mesh as an .obj file.
    """

    assert out_path.parent.exists(), f"Directory not found: {str(out_path.parent)}"
    assert out_path.suffix == ".obj", f"Not an .obj file. Got {out_path.suffix}"

    with open(out_path, "w") as obj_file:
        v_ = v.tolist()
        f_ = f.tolist()

        # check whether mesh has texture
        if (not tex_inds is None) and (not tex is None) and (not uvs is None):
            tex_ = tex
            tex_inds_ = tex_inds.tolist()
            uvs_ = uvs.tolist()
        else:
            tex_ = None
            uvs_ = None
            tex_inds_ = None
        
        # check whether mesh has per-vertex normal
        if (not vn_inds is None) and (not vns is None):
            vns_ = vns.tolist()
            vn_inds_ = vn_inds.tolist()
        else:
            vns_ = None
            vn_inds_ = None 

        obj_file.write("\n# vertices\n")
        for vertex in v_:
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        if (not tex_inds is None) and (not tex is None) and (not uvs is None):
            obj_file.write("\n# texture coordinates\n")
            for uv in uvs_:
                obj_file.write(f"vt {uv[0]} {uv[1]}\n")

            if flip_tex:
                tex_ = np.flip(tex_, axis=0)
            if np.max(tex_) <= 1.0:
                tex_ = (tex_ * 255.0).astype(np.uint8)
            imageio.imwrite(
                out_path.parent / f"{out_path.stem}_texture.png",
                tex_,
            )

        if (not vn_inds is None) and (not vns is None):
            obj_file.write("\n# vertex normals\n")
            for vn in vns_:
                obj_file.write(
                    f"vn {vn[0]} {vn[1]} {vn[2]}\n"
                )

        obj_file.write("\n# faces\n")
        for face_index, face in enumerate(f_):
            face_i_str = str(face[0] + 1)
            face_j_str = str(face[1] + 1)
            face_k_str = str(face[2] + 1)

            # TODO: Clean up this code
            if (not tex_inds is None) and (not tex is None) and (not uvs is None):
                face_i_str += f"/{tex_inds_[face_index][0] + 1}"
                face_j_str += f"/{tex_inds_[face_index][1] + 1}"
                face_k_str += f"/{tex_inds_[face_index][2] + 1}"

                if (not vn_inds is None) and (not vns is None):
                    face_i_str += "/"
                    face_j_str += "/"
                    face_k_str += "/"
            else:
                if (not vn_inds is None) and (not vns is None):
                    face_i_str += "//"
                    face_j_str += "//"
                    face_k_str += "//"

            if (not vn_inds is None) and (not vns is None):
                face_i_str += f"{vn_inds_[face_index][0] + 1}"
                face_j_str += f"{vn_inds_[face_index][1] + 1}"
                face_k_str += f"{vn_inds_[face_index][2] + 1}"

            obj_file.write(f"f {face_i_str} {face_j_str} {face_k_str}\n")

@jaxtyped(typechecker=typechecked)
def find_nearest_vertices(
    v: Shaped[np.ndarray, "N 3"],
    query_pts: Shaped[np.ndarray, "N_Q 3"],
) -> Shaped[np.ndarray, "N_Q"]:
    """
    Finds the nearest vertices in a point cloud.
    """
    dists = scipy.spatial.distance.cdist(query_pts, v)
    nn_inds = np.argmin(dists, axis=1)
    return nn_inds

@jaxtyped(typechecker=typechecked)
def normalize_mesh(
    v: Union[Shaped[np.ndarray, "V 3"], Shaped[torch.Tensor, "V 3"]],
) -> Union[Shaped[np.ndarray, "V 3"], Shaped[torch.Tensor, "V 3"]]:
    """
    Normalizes a mesh to fit into a unit cube centered at the origin.
    """
    # normalize the vertices
    if isinstance(v, np.ndarray):
        v_min = np.min(v, axis=0, keepdims=True)
        v_max = np.max(v, axis=0, keepdims=True)
        v_range = v_max - v_min
        v_center = (v_min + v_max) / 2
        v_normalized = (v - v_center) / v_range.max()
    else:
        v_min = torch.min(v, dim=0, keepdim=True).values
        v_max = torch.max(v, dim=0, keepdim=True).values
        v_range = v_max - v_min
        v_center = (v_min + v_max) / 2
        v_normalized = (v - v_center) / v_range.max()
 
    return v_normalized

@jaxtyped(typechecker=typechecked)
def normalize_mesh_batch(
    v: Union[Shaped[np.ndarray, "B V 3"], Shaped[torch.Tensor, "B V 3"]],
) -> Union[Shaped[np.ndarray, "B V 3"], Shaped[torch.Tensor, "B V 3"]]:
    """
    Normalizes a mesh to fit into a unit cube centered at the origin.
    """
    # normalize the vertices
    if isinstance(v, np.ndarray):
        v_min = np.min(v, axis=1, keepdims=True)
        v_max = np.max(v, axis=1, keepdims=True)
        v_range = v_max - v_min
        v_center = (v_min + v_max) / 2
        v_normalized = (v - v_center) / v_range.max(axis=-1, keepdims=True)
    else:
        v_min = torch.min(v, dim=1, keepdim=True).values
        v_max = torch.max(v, dim=1, keepdim=True).values
        v_range = v_max - v_min
        v_center = (v_min + v_max) / 2
        v_normalized = (v - v_center) / v_range.max(dim=-1, keepdim=True).values
 
    return v_normalized
