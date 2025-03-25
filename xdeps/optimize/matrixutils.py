# Create a class SVD which is a wrapper around numpy.linalg.svd

import numpy as np

class SVD:

    def __init__(self, matrix, rcond=1e-14, sing_val_cutoff = None):
        """
        A class for performing singular value decomposition on a matrix and
        solving least squares problems using the SVD and to keep metrics
        like the condition number and rank of the matrix.

        Parameters
        ----------
        matrix : array_like
            The matrix to perform the SVD on.
        rcond : float, optional
            Cutoff for small singular values. Singular values less than
            rcond * largest_singular_value are set to zero. The default
            is None.
        sing_val_cutoff : int, optional
            Number of singular values to use for least squares solution.
            The default is None.
        """
        self.matrix = matrix
        self.rcond = rcond
        self.empty = matrix.size == 0
        if not self.empty:
            self.U, self.s, self.Vh = np.linalg.svd(matrix, full_matrices=False)
            if self.s[-1] < 1e-16:
                self.cond = np.inf
            else:
                self.cond = self.s[0] / self.s[-1]
            self.rank = np.linalg.matrix_rank(matrix)
            if sing_val_cutoff is None:
                self.sing_val_cutoff = len(self.s)
            else:
                self.sing_val_cutoff = sing_val_cutoff
        else:
            self.cond = -1
            self.rank = -1
            self.sing_val_cutoff = None

    def lstsq(self, b, rcond = None, sing_val_cutoff = None):
        """
        Solve a least squares problem using the SVD.

        Parameters
        ----------
        b : array_like
            Right-hand side of the equation to solve.
        rcond : float, optional
            Cutoff for small singular values. Singular values less than
            rcond * largest_singular_value are set to zero. The default
            is None.
        sing_val_cutoff : int, optional
            Number of singular values to use for least squares solution.
            The default is None.
        """
        if self.empty:
            return np.array([])

        if rcond is None:
            rcond = self.rcond
        if sing_val_cutoff is None:
            sing_val_cutoff = self.sing_val_cutoff

        U = self.U[:, :sing_val_cutoff]
        Vh = self.Vh[:sing_val_cutoff, :]
        s = self.s[:sing_val_cutoff]

        s_inv = np.zeros_like(s)
        s_inv[s > 0] = 1 / s[s > 0]
        if rcond is not None:
            s_inv[s < rcond * s[0]] = 0

        x = Vh.T @ (np.diag(s_inv) @ (U.T @ b))
        return x
