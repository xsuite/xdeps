import numpy as np
import xdeps as xd
from xdeps.optimize.matrixutils import SVD

def test_svd_initialization():
    A = np.array([[1, 2], [3, 4], [5, 6]])
    svd = SVD(A)

    assert svd.U.shape == A.shape
    assert svd.s.shape[0] == min(A.shape)
    assert svd.Vh.shape == (A.shape[1], A.shape[1])

    assert np.isclose(svd.cond, np.linalg.cond(A))
    assert svd.rank == np.linalg.matrix_rank(A)

    assert np.isclose(svd.U, np.linalg.svd(A, full_matrices=False)[0]).all()
    assert np.isclose(svd.s, np.linalg.svd(A, full_matrices=False)[1]).all()
    assert np.isclose(svd.Vh, np.linalg.svd(A, full_matrices=False)[2]).all()

def test_svd_lstsq():
    A = np.array([[2, 1], [1, 3], [0, 1]])
    b = np.array([1, 2, 3])
    svd = SVD(A)

    x = svd.lstsq(b)
    x_sol = np.array([-0.2, 1.])

    assert np.allclose(x, x_sol)

def test_svd_lstsq_cutoff():

    A = np.array([[1, 0, 0], [0, 1e-10, 0], [0, 0, 1]])
    b = np.array([1, 1, 1])
    svd = SVD(A, rcond=1e-5)
    svd2 = SVD(A, sing_val_cutoff=2)

    x = svd.lstsq(b)
    x2 = svd2.lstsq(b)

    s_i = np.array([1, 1, 0])
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    x_expected = Vh.T @ (np.diag(s_i) @ (U.T @ b))

    assert np.allclose(x, x_expected)
    assert np.allclose(x2, x_expected)