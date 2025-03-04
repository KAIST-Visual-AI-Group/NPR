"""
math_utils.py
"""

from typing import Union

from jaxtyping import Shaped, jaxtyped
import numpy as np
import torch
import torch.nn.functional as F
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
def normalize_vs(
    vs: Union[Shaped[np.ndarray, "* D"], Shaped[torch.Tensor, "* D"]]
) -> Union[Shaped[np.ndarray, "* D"], Shaped[torch.Tensor, "* D"]]:
    """
    Normalize a batch of vectors.
    """
    eps = 1e-8

    # PyTorch
    if isinstance(vs, torch.Tensor):
        is_batch = False
        if vs.ndim == 2:
            is_batch = True
        else:
            assert vs.ndim == 1, f"Expected 1D or 2D array, got {vs.ndim}D"
            vs = vs[None, ...]
        norm = torch.norm(vs, dim=1, keepdim=True)
        vs = vs / (norm + eps)
        if not is_batch:
            vs = vs.squeeze(0)
    
    # Numpy
    else:
        is_batch = False
        if vs.ndim == 2:
            is_batch = True
        else:
            assert vs.ndim == 1, f"Expected 1D or 2D array, got {vs.ndim}D"
            vs = vs[np.newaxis, ...]
        norm = np.linalg.norm(vs, axis=1, keepdims=True)
        vs = vs / (norm + eps)
        if not is_batch:
            vs = vs.squeeze(0)
    
    return vs

@jaxtyped(typechecker=typechecked)
def euler_to_mat(
    theta_x: float, theta_y: float, theta_z: float
) -> Shaped[np.ndarray, "3 3"]:
    """
    Compute a rotation matrix specified by Euler angles.
    """
    thetas = [theta_x, theta_y, theta_z]
    assert all([x >= -2 * np.pi and x <= 2 * np.pi for x in thetas]), (
        f"Expected Euler angles to be in [0, 2pi], got {thetas}"
    )

    # rotation around X-axis
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ],
    )

    # rotation around Y-axis
    rot_y = np.array(
        [
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ],
    )

    # rotation around Z-axis
    rot_z = np.array(
        [
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ],
    )

    # combine rotations
    rot = rot_z @ rot_y @ rot_x

    return rot

@jaxtyped(typechecker=typechecked)
def mat_to_quat(
    rot: Shaped[np.ndarray, "3 3"]
) -> Shaped[np.ndarray, "4"]:
    """
    Convert a rotation matrix to a unit quaternion (W, X, Y, Z)
    """
    trace = np.trace(rot)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (rot[2, 1] - rot[1, 2]) / S
        qy = (rot[0, 2] - rot[2, 0]) / S
        qz = (rot[1, 0] - rot[0, 1]) / S
    elif (rot[0, 0] > rot[1, 1]) and (rot[0, 0] > rot[2, 2]):
        S = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        qw = (rot[2, 1] - rot[1, 2]) / S
        qx = 0.25 * S
        qy = (rot[0, 1] + rot[1, 0]) / S
        qz = (rot[0, 2] + rot[2, 0]) / S
    elif rot[1, 1] > rot[2, 2]:
        S = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        qw = (rot[0, 2] - rot[2, 0]) / S
        qx = (rot[0, 1] + rot[1, 0]) / S
        qy = 0.25 * S
        qz = (rot[1, 2] + rot[2, 1]) / S
    else:
        S = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        qw = (rot[1, 0] - rot[0, 1]) / S
        qx = (rot[0, 2] + rot[2, 0]) / S
        qy = (rot[1, 2] + rot[2, 1]) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])

@jaxtyped(typechecker=typechecked)
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.

    Brought from: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

@jaxtyped(typechecker=typechecked)
def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

@jaxtyped(typechecker=typechecked)
def mat_to_quat_torch(
    rot: Shaped[torch.Tensor, "... 3 3"]
) -> Shaped[torch.Tensor, "... 4"]:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).

    Brought from: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    """
    if rot.size(-1) != 3 or rot.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {rot.shape}.")

    batch_dim = rot.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        rot.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

@jaxtyped(typechecker=typechecked)
def quat_to_mat(
    quats: Shaped[np.ndarray, "* 4"]
) -> Shaped[np.ndarray, "* 3 3"]:
    """
    Convert a batch of quaternions to a batch of rotation matrices.
    """
    is_batch = False
    if quats.ndim == 2:  # received a batch of quaternions
        is_batch = True
    else:  # received a single quaternion
        assert quats.ndim == 1, f"Expected 1D or 2D array, got {quats.ndim}D"
        quats = quats[np.newaxis, ...]

    norm = np.sqrt(quats[:, 0] ** 2 + quats[:, 1] ** 2 + quats[:, 2] ** 2 + quats[:, 3] ** 2)
    assert np.allclose(norm, np.ones_like(norm)), f"Expected unit quaternions"

    B = quats.shape[0]
    mats = np.zeros((B, 3, 3))

    W, X, Y, Z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    XX = X * X
    XY = X * Y
    XZ = X * Z
    YY = Y * Y
    YZ = Y * Z
    ZZ = Z * Z
    WX = W * X
    WY = W * Y
    WZ = W * Z

    mats[:, 0, 0] = 1 - 2 * (YY + ZZ)
    mats[:, 0, 1] = 2 * (XY - WZ)
    mats[:, 0, 2] = 2 * (XZ + WY)
    mats[:, 1, 0] = 2 * (XY + WZ)
    mats[:, 1, 1] = 1 - 2 * (XX + ZZ)
    mats[:, 1, 2] = 2 * (YZ - WX)
    mats[:, 2, 0] = 2 * (XZ - WY)
    mats[:, 2, 1] = 2 * (YZ + WX)
    mats[:, 2, 2] = 1 - 2 * (XX + YY)

    if not is_batch:
        mats = mats.squeeze(0)

    return mats

@jaxtyped(typechecker=typechecked)
def quat_to_mat_torch(
    quats: Shaped[torch.Tensor, "* 4"]
) -> Shaped[torch.Tensor, "* 3 3"]:
    """
    Convert a batch of quaternions to a batch of rotation matrices.
    """
    is_batch = False
    if quats.ndim == 2:  # received a batch of quaternions
        is_batch = True
    else:  # received a single quaternion
        assert quats.ndim == 1, f"Expected 1D or 2D array, got {quats.ndim}D"
        quats = quats[None, ...]

    norm = torch.sqrt(quats[:, 0] ** 2 + quats[:, 1] ** 2 + quats[:, 2] ** 2 + quats[:, 3] ** 2)
    assert torch.allclose(norm, torch.ones_like(norm)), f"Expected unit quaternions"

    B = quats.shape[0]
    mats = torch.zeros(
        (B, 3, 3), dtype=quats.dtype, device=quats.device
    )

    W, X, Y, Z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    XX = X * X
    XY = X * Y
    XZ = X * Z
    YY = Y * Y
    YZ = Y * Z
    ZZ = Z * Z
    WX = W * X
    WY = W * Y
    WZ = W * Z

    mats[:, 0, 0] = 1 - 2 * (YY + ZZ)
    mats[:, 0, 1] = 2 * (XY - WZ)
    mats[:, 0, 2] = 2 * (XZ + WY)
    mats[:, 1, 0] = 2 * (XY + WZ)
    mats[:, 1, 1] = 1 - 2 * (XX + ZZ)
    mats[:, 1, 2] = 2 * (YZ - WX)
    mats[:, 2, 0] = 2 * (XZ - WY)
    mats[:, 2, 1] = 2 * (YZ + WX)
    mats[:, 2, 2] = 1 - 2 * (XX + YY)

    if not is_batch:
        mats = mats.squeeze(0)

    return mats

@jaxtyped(typechecker=typechecked)
def quat_slerp(
    Q1: np.ndarray, Q2: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """
    Spherical interpolation of two quaternion rotations.
    """
    raise NotImplementedError("TODO")
    