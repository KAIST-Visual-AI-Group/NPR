"""
poisson_system.py

An implementation of Poisson system solver.
"""

from typing import Any, Dict, Optional, Tuple, Union

import cholespy
import igl
from jaxtyping import Int, Shaped, jaxtyped
import numpy as np
from numpy import ndarray
from scipy.sparse import (
    csc_matrix,
    csr_matrix,
    coo_matrix,
    diags,
    kron,
    hstack,
    vstack,
)
import torch
from torch import Tensor
from typeguard import typechecked

from .PoissonSystem import SPLUSolveLayer, SparseMat
from ..utils.geometry_utils import columnize_rotations


MUL_DIAG = False

class PoissonSystem:
    
    v_src: Shaped[Tensor, "num_vertex 3"]
    """The vertices of the source mesh"""
    f_src: Shaped[Tensor, "num_face 3"]
    """The faces of the source mesh"""
    anchor_inds: Optional[Int[Tensor, "num_handle"]]
    """The indices of the handle vertices used as the constraints"""
    constraint_lambda: float
    """The weight of the constraint term in the system"""
    is_constrained: bool
    """The Boolean flag indicating the presence of the constraint term in the system"""
    device: torch.device
    """The device where the computation is performed"""
    torch_dtype: Any
    """The data type of PyTorch tensors used in Poisson system"""
    is_sparse: bool
    """The flag for enabling computation of sparse tensors"""
    cpu_only: bool
    """The flag for forcing the computation on CPU"""

    # differential operators
    grad: SparseMat
    """The gradient operator computed using the source mesh"""
    rhs: SparseMat
    """The right-hand side of the Poisson system"""
    w: Shaped[ndarray, "num_face 3 2"]
    """The local basis of the source mesh"""
    L: SparseMat
    """The Laplacian operator computed using the source mesh"""
    L_fac: cholespy.CholeskySolverD
    """The factorization of the Laplacian operator"""
    J: Shaped[Tensor, "num_face 3 3"]
    """The per-triangle Jacobians computed using the source mesh"""
    indicator_matrix: Optional[SparseMat] = None
    """The indicator matrix for the constrained system"""
    indicator_product: Optional[SparseMat] = None
    """The product of the indicator matrix and its transpose"""
    local_basis: Shaped[Tensor, "num_vertex 3 3"]
    """The per-vertex local basis of the source mesh"""
    local_basis_per_f: Shaped[Tensor, "num_face 3 3"]
    """The per-vertex local basis of the source mesh"""

    f_to_v: SparseMat
    """The operator to convert face-based data to vertex-based data"""
    n_f_per_v: torch.Tensor
    """The number of faces per vertex"""

    # additional flags
    no_factorization: bool = False
    """The flag for disabling matrix factorization. Enabled when debugging the module"""

    @jaxtyped
    @typechecked
    def __init__(
        self,
        v: Shaped[Tensor, "num_vertex 3"],
        f: Shaped[Tensor, "num_face 3"],
        device: torch.device,
        anchor_inds: Optional[Int[Tensor, "num_handle"]] = None,
        constraint_lambda: float = 1e5,
        train_J: bool = False,
        is_sparse: bool = True,
        cpu_only: bool = False,
        no_factorization: bool = False,
    ) -> None:
        """
        Constructor of the Poisson system solver.
        """
        self.v_src: Shaped[Tensor, "num_vertex 3"] = v
        self.f_src: Shaped[Tensor, "num_face 3"] = f
        self.anchor_inds: Optional[Int[Tensor, "num_handle"]] = anchor_inds
        self.constraint_lambda: float = constraint_lambda
        self.train_J: bool = train_J
        self.device: torch.device = device
        self.torch_dtype = torch.float64  # NOTE: Use double precision for numerical stability
        self.is_sparse: bool = is_sparse
        self.cpu_only: bool = cpu_only
        self.no_factorization = no_factorization

        self._build_poisson_system()

    @jaxtyped(typechecker=typechecked)
    def _build_poisson_system(self) -> None:

        # check whether the system is constrained
        self.is_constrained = self.anchor_inds is not None
        if self.is_constrained:
            assert self.constraint_lambda > 0.0, (
                f"Invalid weight value: {self.constraint_lambda}"
            )

        # compute the operator 'f_to_v'
        self._compute_f_to_v()

        # compute gradient, Laplacian, and right-hand side of the system
        if self.is_constrained:
            self._compute_differential_operators_constrained()
        else:
            self._compute_differential_operators_unconstrained()

        # check whether the system is constrained
        if self.is_constrained:
            self._build_indicator_matrix()

        # pre-factorize the matrices used in the system
        if not self.no_factorization:
            self._factorize_matrices()
        else:
            print("[PoissonSystem] Coefficient matrix is not factorized")
            self.L_fac = None

        # initialize per-triangle Jacobians
        self._compute_per_triangle_jacobian()

        # set the learnable parameters (Jacobians)
        self.J.requires_grad_(self.train_J)

    @jaxtyped(typechecker=typechecked)
    def get_current_mesh(
        self,
        constraints: Optional[Shaped[Tensor, "num_handle 3"]] = None,
        trans_mats: Optional[Shaped[Tensor, "num_face 3 3"]] = None,
        return_float32: bool = True,
    ) -> Tuple[Shaped[Tensor, "num_vertex 3"], Shaped[Tensor, "num_face 3"]]:
        """
        Returns the mesh whose geometry is determined by the current values of Jacobians.
        """
        # create a mesh
        new_v = self.solve(constraints, trans_mats)
        new_f = self.f_src
        if return_float32:
            new_v = new_v.type(torch.float32)
        return new_v, new_f

    @jaxtyped(typechecker=typechecked)
    def solve(
        self,
        constraints: Optional[Shaped[Tensor, "num_handle 3"]] = None,
        trans_mats: Optional[Shaped[Tensor, "num_face 3 3"]] = None,
    ) -> Shaped[Tensor, "num_vertex 3"]:
        """
        Computes vertex positions according to the current per-triangle Jacobians.
        """
        self._check_jacobians()
        if self.is_constrained:
            assert constraints is not None
            return self._solve_constrained(constraints, trans_mats)
        else:
            return self._solve_unconstrained(trans_mats)

    @jaxtyped(typechecker=typechecked)
    def _solve_unconstrained(
        self,
        trans_mats: Optional[Shaped[Tensor, "num_face 3 3"]] = None,
    ) -> Shaped[Tensor, "num_vertex 3"]:

        # compute the RHS of the system
        jacobians_ = self.J
        if not trans_mats is None:
            jacobians_ = torch.matmul(trans_mats, jacobians_)
        jacobians_ = jacobians_.transpose(1, 2).reshape(-1, 3, 1).squeeze(2).contiguous()
        jacobians_ = rearrange_jacobian_elements(jacobians_)
        rhs = self.rhs.multiply_with_dense(jacobians_)

        # solve least squares
        v_sol: Shaped[Tensor, "num_vertex 3"] = SPLUSolveLayer.apply(
            self.L_fac,
            rhs[None, ...],
        )[0, ...]

        # Recenter the mesh to the origin
        v_sol = torch.cat(
            [
                torch.zeros(1, 3, dtype=v_sol.dtype, device=v_sol.device),
                v_sol,
            ],
            dim=0,
        )
        center = torch.mean(v_sol, dim=0, keepdim=True)
        v_sol = v_sol - center
        v_sol = v_sol.type_as(self.J)

        return v_sol

    @jaxtyped(typechecker=typechecked)
    def _solve_constrained(
        self,
        constraint: Shaped[Tensor, "num_handle 3"],
        trans_mats: Optional[Shaped[Tensor, "num_face 3 3"]] = None,
    ) -> Shaped[Tensor, "num_vertex 3"]:

        # compute the RHS of the system
        J_ = self.J
        if not trans_mats is None:
            J_ = torch.matmul(trans_mats, J_)
        J_ = J_.transpose(1, 2).reshape(-1, 3, 1).squeeze(2).contiguous()
        J_ = rearrange_jacobian_elements(J_)
        rhs = self.rhs.multiply_with_dense(J_)
        rhs = self.L.transpose().multiply_with_dense(rhs)
        rhs = rhs + self.constraint_lambda * (
            self.indicator_matrix.transpose().multiply_with_dense(constraint)
        )

        # solve the constrained least squares
        v_sol: Shaped[Tensor, "num_vertex 3"] = SPLUSolveLayer.apply(
            self.L_fac,
            rhs[None, ...],
        )[0, ...]
        v_sol = v_sol.type_as(self.J)

        return v_sol

    @jaxtyped(typechecker=typechecked)
    def _compute_differential_operators_unconstrained(self) -> None:
        """
        Computes operators involving in Poisson's equation from the given mesh.

        This code is borrowed from: https://github.com/threedle/TextDeformer/blob/main/NeuralJacobianFields/PoissonSystem.py
        """

        # prepare Numpy arrays
        v = self.v_src.clone().cpu().numpy()
        f = self.f_src.clone().cpu().numpy()

        # compute gradient
        grad: Shaped[ndarray, "3F V"] = igl.grad(v, f)
    
        # NOTE
        # the diagonal elements of the mass matrix are arranged
        # in a way that follows the convention how libigl stacks
        # the coefficients to compute x, y, z component of gradient vectors
        # refer to _convert_sparse_igl_grad_to_our_convention for details.
        mass: Shaped[ndarray, "F F"] = (
            compute_mass_matrix(v, f, self.is_sparse)
        )
    
        # compute Laplacian
        laplace: Shaped[ndarray, "V V"] = grad.T @ mass @ grad
        laplace = laplace[1:, 1:]  # Why? -> Probably related to the rest constraint?

        # compute right-hand side of the system
        rhs = grad.T @ mass
        b1, b2, _ = igl.local_basis(v, f)
        w = np.stack(
            (b1,b2),
            axis=-1,
        )
        rhs = rhs[1:,:]

        if self.is_sparse:
            laplace = laplace.tocoo()
            rhs = rhs.tocoo()
            grad = grad.tocsc()
        else:
            laplace = laplace.toarray()
            rhs = rhs.toarray()
            grad = grad.toarray()
    
        # rearrange the elements of the gradient matrix
        grad = rearrange_igl_grad_elements(grad)

        # construct 'back-prop'able sparse matrices
        grad = SparseMat.from_M(grad, self.torch_dtype).to(self.device)
        rhs = SparseMat.from_coo(rhs, self.torch_dtype).to(self.device)
        L = SparseMat.from_coo(laplace, self.torch_dtype).to(self.device)
    
        # register the computed operators
        self.grad = grad
        self.w = w
        self.rhs = rhs
        self.L = L

    @jaxtyped(typechecker=typechecked)
    def _compute_differential_operators_constrained(self) -> None:
        """
        Computes operators involving in Poisson's equation from the given mesh.

        This code is borrowed from: https://github.com/threedle/TextDeformer/blob/main/NeuralJacobianFields/PoissonSystem.py
        """

        # prepare Numpy arrays
        v = self.v_src.clone().cpu().numpy()
        f = self.f_src.clone().cpu().numpy()

        # compute gradient
        grad: Shaped[ndarray, "3F V"] = igl.grad(v, f)
    
        # NOTE
        # the diagonal elements of the mass matrix are arranged
        # in a way that follows the convention how libigl stacks
        # the coefficients to compute x, y, z component of gradient vectors
        # refer to _convert_sparse_igl_grad_to_our_convention for details.
        mass: Shaped[ndarray, "F F"] = (
            compute_mass_matrix(v, f, self.is_sparse)
        )
    
        # compute Laplacian
        # TODO: Figure out difference between libigl and our implementation
        laplace: Shaped[ndarray, "V V"] = grad.T @ mass @ grad
        self.laplace = laplace.copy()
        self.LTL = self.laplace.transpose() @ self.laplace

        # compute right-hand side of the system
        rhs = grad.T @ mass
        b1, b2, b3 = igl.local_basis(v, f)
        w = np.stack(
            (b1,b2),
            axis=-1,
        )

        # Compute per-vertex frame
        w_ = np.stack(
            [b1, b2, b3],
            axis=-1,
        )
        w_ = torch.from_numpy(w_).to(self.device)
        w_ = w_.reshape(-1, 9)
        w_per_v = self.f_to_v.multiply_with_dense(w_)
        w_per_v = w_per_v / self.n_f_per_v
        w_per_v_norm = torch.norm(w_per_v, dim=1, keepdim=True)
        w_per_v = w_per_v / (w_per_v_norm + 1e-6)
        self.local_basis = w_per_v.reshape(-1, 3, 3).type(torch.float32)
        self.local_basis_per_f = w_.reshape(-1, 3, 3).type(torch.float32)

        if self.is_sparse:
            laplace = laplace.tocoo()
            rhs = rhs.tocoo()
            grad = grad.tocsc()
        else:
            laplace = laplace.toarray()
            rhs = rhs.toarray()
            grad = grad.toarray()
    
        # rearrange the elements of the gradient matrix
        grad = rearrange_igl_grad_elements(grad)

        # construct 'back-prop'able sparse matrices
        grad = SparseMat.from_M(grad, self.torch_dtype).to(self.device)
        rhs = SparseMat.from_coo(rhs, self.torch_dtype).to(self.device)
        laplacian = SparseMat.from_coo(laplace, self.torch_dtype).to(self.device)

        # register the computed operators
        self.grad = grad
        self.w = w
        self.rhs = rhs
        self.L = laplacian

    @jaxtyped(typechecker=typechecked)
    def _build_indicator_matrix(self) -> None:
        """
        Builds a matrix that indicates the vertices being constrained
        in the system.
        """
        num_handle: int = int(len(self.anchor_inds))
        num_vertex: int = int(self.v_src.shape[0])
        anchor_inds: Int[ndarray, "num_handle"] = (
            self.anchor_inds.cpu().numpy()
        )

        # check whether handle indices are unique
        assert len(np.unique(anchor_inds)) == num_handle, (
            f"Handle indices must be unique. " 
            "Otherwise, one needs to compute a multiplication of large matrices"
        )

        # build a list of non-zero indices
        indicator_indices: Int[ndarray, "2 num_handle"] = (
            np.zeros((2, num_handle), dtype=np.int64)
        )
        indicator_indices[0, :] = np.arange(num_handle, dtype=np.int64)
        indicator_indices[1, :] = anchor_inds

        # create a sparse matrix
        self.indicator_matrix = SparseMat(
            indicator_indices,
            np.ones_like(anchor_inds),
            n=num_handle,
            m=num_vertex,
            ttype=self.torch_dtype,
        ).to(self.device)

        # create the product of the indicator transposed and itself
        product_indices: Int[ndarray, "2 num_handle"] = np.concatenate(
            (
                anchor_inds[None, :],
                anchor_inds[None, :],
            ),
            axis=0,
        )
        self.indicator_product = SparseMat(
            product_indices,
            np.ones_like(anchor_inds),
            n=num_vertex,
            m=num_vertex,
            ttype=self.torch_dtype,
        ).to(self.device)

    @jaxtyped(typechecker=typechecked)
    def _factorize_matrices(self) -> None:
        """
        Factorizes the large, sparse matrices used in the system.
        """
        # compute the matrix to factorize
        mat_to_fac: SparseMat = self._compute_matrix_to_factorize()
        self.lap_factored = mat_to_fac

        # Cholesky factorization
        self.L_fac: cholespy.CholeskySolverD = (
            mat_to_fac.to_cholesky()
        )

    @jaxtyped(typechecker=typechecked)
    def _compute_matrix_to_factorize(self):  # NOTE: Temporarily remove type annotation
        """Computes the matrix to be factorized"""
        # retrieve the Laplacian matrix
        mat_to_fac: SparseMat = self.L

        # add constraint term if necessary
        if self.is_constrained:
            mat_to_fac: coo_matrix = mat_to_fac.to("cpu").to_coo()

            # compute L^{T} @ L
            mat_to_fac = mat_to_fac.transpose() @ mat_to_fac

            # compute L^{T} @ L + lambda * K
            indicator_product: coo_matrix = self.indicator_product.to("cpu").to_coo()
            mat_to_fac: coo_matrix = (
                mat_to_fac + self.constraint_lambda * indicator_product
            )

            # convert the matrix back to the sparse matrix format
            mat_to_fac = SparseMat.from_coo(
                mat_to_fac.tocoo(),  # ensure COO-format to be passed
                self.torch_dtype,
            ).to(self.device)

        return mat_to_fac

    @jaxtyped(typechecker=typechecked)
    def _compute_per_triangle_jacobian(self) -> None:
        """
        Computes per-triangle Jacobians from the given vertices.
        """
        # retrieve operands
        grad: SparseMat = self.grad
        v: Shaped[Tensor, "num_vertex 3"] = self.v_src

        # compute Jacobians
        J = grad.multiply_with_dense(v)
        J = J[:, None]
        J = J.reshape(-1, 3, 3)
        J = J.transpose(1, 2)

        # register the computed Jacobians
        self.J = J

    @jaxtyped(typechecker=typechecked)
    def compute_per_triangle_jacobian(
        self,
        vertices: Shaped[Tensor, "num_vertex 3"],
    ) -> Shaped[Tensor, "num_face 3 3"]:
        """
        Computes per-triangle Jacobians from the given vertices.
        """
        J = self.grad.multiply_with_dense(vertices)
        J = J[:, None]
        J = J.reshape(-1, 3, 3)
        J = J.transpose(1, 2)
        return J

    @jaxtyped(typechecker=typechecked)
    def _compute_f_to_v(self) -> None:
        """
        Computes the sparse matrix that converts per-face data to per-vertex data.
        """
        assert not self.v_src is None, (
            "Source vertices must be set before calling this method."
        )
        assert not self.f_src is None, (
            "Source faces must be set before calling this method."
        )

        n_v = len(self.v_src)
        n_f = len(self.f_src)

        # Build a list of non-zero indices
        inds = []
        for i in range(n_f):
            for j in range(3):
                inds.append([int(self.f_src[i, j].item()), i])
        inds = np.array(inds).T

        # Compute the operator itself
        self.f_to_v = SparseMat(
            inds,
            np.ones(inds.shape[1]),
            n=n_v,
            m=n_f,
            ttype=self.torch_dtype,
        ).to(self.device)
        print("Computed f_to_v")

        # Compute the number of faces per-vertex
        f_to_v = self.f_to_v.to("cpu").to_coo()
        n_f_per_v = torch.from_numpy(np.array(f_to_v.sum(axis=1)))
        self.n_f_per_v = n_f_per_v.to(self.device)
        assert n_f_per_v.shape[0] == n_v, (
            f"{n_f_per_v.shape[0]} != {n_v}"
        )
        print("Computed n_f_per_v")

    @jaxtyped(typechecker=typechecked)
    def _check_jacobians(self) -> None:
        assert self.J.ndim == 3, f"Invalid shape: {self.J.shape}"
        assert self.J.shape[0] == self.f_src.shape[0], (
            f"Invalid shape: {self.J.shape[0]} != {self.f_src.shape[0]}"
        )
        assert self.J.shape[1] == 3, (
            f"Each Jacobian matrix must have 3 rows: {self.J.shape[1]} != 3"
        )
        assert self.J.shape[2] == 3, (
            f"Each Jacobian matrix must have 3 columns: {self.J.shape[2]} != 3"
        )

    @jaxtyped(typechecker=typechecked)
    def compute_coefficient_statistics(self):
        """Computes the statistics of the Laplacian matrix used in the system"""
        mat_to_fac: SparseMat = self._compute_matrix_to_factorize()
        mat_to_fac: coo_matrix = mat_to_fac.to("cpu").to_coo().todense()

        # compute rank
        rank = np.linalg.matrix_rank(mat_to_fac)

        # compute condition number
        cond_number = np.linalg.cond(mat_to_fac)

        return rank, cond_number

@jaxtyped
@typechecked
def compute_mass_matrix(
    vertices: Shaped[ndarray, "num_vertex 3"],
    faces: Shaped[ndarray, "num_face 3"],
    is_sparse: bool,
):
    """
    Computes the mass matrix of a mesh.
    """
    double_area = igl.doublearea(vertices, faces)
    double_area = np.hstack(
        (
            double_area,
            double_area,
            double_area,
        )
    )
    
    if is_sparse:
        return csc_matrix(diags(double_area))
    else:
        return diags(double_area)

@jaxtyped
# @typechecked
def rearrange_igl_grad_elements(input):
    """
    The grad operator computed from igl.grad() results in a matrix of shape (3*#tri x #verts).
    It is packed such that all the x-coordinates are placed first, followed by y and z. As shown below

    ----------           ----------
    | x1 ...             | x1 ...
    | x2 ...             | y1 ...
    | x3 ...             | z1 ...
    | .                  | .
    | .                  | .
    | y1 ...             | x2 ...
    | y2 ...      ---->  | y2 ...
    | y3 ...             | z2 ...
    | .                  | .
    | .                  | .
    | z1 ...             | x3 ...
    | z2 ...             | y3 ...
    | z3 ...             | z3 ...
    | .                  | .
    | .                  | .
    ----------           ----------

    Note that this functionality cannot be computed trivially if because igl.grad() is a sparse tensor and as such
    slicing is not well defined for sparse matrices. the following code performs the above conversion and returns a
    torch.sparse tensor.
    Set check to True to verify the results by converting the matrices to dense and comparing it.

    This code has been borrowed from: https://github.com/threedle/TextDeformer/blob/main/NeuralJacobianFields/PoissonSystem.py
    """
    assert type(input) == csc_matrix, 'Input should be a scipy csc sparse matrix'
    T = input.tocoo()

    r_c_data = np.hstack((T.row[..., np.newaxis], T.col[..., np.newaxis],
                          T.data[..., np.newaxis]))  # horizontally stack row, col and data arrays
    r_c_data = r_c_data[r_c_data[:, 0].argsort()]  # sort along the row column

    # Separate out x, y and z blocks
    # Note that for the grad operator there are exactly 3 non zero elements in a row
    L = T.shape[0]
    Tx = r_c_data[:L, :]
    Ty = r_c_data[L:2 * L, :]
    Tz = r_c_data[2 * L:3 * L, :]

    # align the y,z rows with x so that they too start from 0
    Ty[:, 0] -= Ty[0, 0]
    Tz[:, 0] -= Tz[0, 0]

    # 'strech' the x,y,z rows so that they can be interleaved.
    Tx[:, 0] *= 3
    Ty[:, 0] *= 3
    Tz[:, 0] *= 3

    # interleave the y,z into x
    Ty[:, 0] += 1
    Tz[:, 0] += 2

    Tc = np.zeros((input.shape[0] * 3, 3))
    Tc[::3] = Tx
    Tc[1::3] = Ty
    Tc[2::3] = Tz

    indices = Tc[:, :-1].astype(int)
    data = Tc[:, -1]

    return (indices.T, data, input.shape[0], input.shape[1])

@jaxtyped
# @typechecked
def rearrange_jacobian_elements(jacobians):
    """
    Rearranges the elements of the given per-triangle Jacobian matrix.

    Args:
        jacobians: Per-triangle Jacobian matrices.
    """
    jacobians_rearranged = torch.zeros_like(jacobians)
    num_face = jacobians.shape[0] // 3

    # rearrange the elements
    jacobians_rearranged[:num_face, :] = jacobians[::3, ...]
    jacobians_rearranged[num_face : 2 * num_face, :] = jacobians[1::3, ...]
    jacobians_rearranged[2 * num_face :, :] = jacobians[2::3, ...]

    return jacobians_rearranged