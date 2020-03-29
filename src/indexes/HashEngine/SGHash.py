import time

import numpy as np
import scipy.linalg as alg
from nearpy.hashes import LSHash


class SGHash(LSHash):

    def __init__(self, name: str, training_data: np.ndarray, projections: int, num_bases: int = 300):
        super().__init__(hash_name=name)

        self.dim = training_data.shape[1]
        self.projections = projections
        self.num_bases = num_bases

        if training_data is None:
            raise Exception('training data must be provided to construct the hash function')

        t = time.time()

        # 0 center features
        self.feats_mean = np.mean(training_data, axis=0)
        centered_data = training_data - self.feats_mean

        # choose bases
        random_indexes = np.random.permutation(centered_data.shape[0])[:num_bases]
        self.bases = training_data[random_indexes]

        # precalculate bases square for kernel calculations
        self.bases_2 = np.reshape(np.sum(np.multiply(self.bases, self.bases), axis=1), (-1, 1))

        # train sgh
        self.kernel_delta = None
        self.kernel_mean = None
        self.wx = None
        self.train_sgh(data=centered_data)

        print(f'SGH trained with {training_data.shape[0]} vectors in {time.time() - t:.2f} seconds')

    def reset(self, dim):
        """ Resets / Initializes the hash for the specified dimension. """
        if self.dim != dim:
            raise Exception('SGH was trained for a different dimension!')

    def hash_vector(self, vector, querying=False):
        # center feats
        centered_vector = vector - self.feats_mean

        # calculate kernel
        kernel_test = self.dist_matrix(centered_vector)
        kernel_test = np.exp(- kernel_test / (2 * self.kernel_delta)) - self.kernel_mean

        # get projection and binarize
        projection = np.dot(kernel_test, self.wx)
        return [''.join(['1' if x > 0.0 else '0' for x in projection[0]])]

    def get_config(self):
        pass

    def apply_config(self, config):
        pass

    def train_sgh(self, data):
        # convert to unit vectors
        # data_norm = np.linalg.norm(data, axis=1, keepdims=True)
        # data /= data_norm

        n = data.shape[0]
        rho = 2

        # construct PX and QX
        e = np.exp(1)
        norm_ = np.linalg.norm(data, axis=1, keepdims=True) ** 2
        norm_ = np.exp(-norm_ / rho)

        alpha = np.sqrt(2 * (e * e - 1) / (e * rho))
        beta = np.sqrt((e * e + 1) / e)

        part2 = beta * norm_
        part1 = alpha * norm_ * data
        p_x = np.c_[part1, part2, np.ones((n, 1), dtype=np.float32)]
        q_x = np.c_[part1, part2, -1 * np.ones((n, 1), dtype=np.float32)]

        # construct kernel for training set
        kernel = self.dist_matrix(data)

        self.kernel_delta = np.mean(kernel)
        kernel = np.exp(- kernel / (2 * self.kernel_delta))
        self.kernel_mean = np.mean(kernel, axis=0)
        kernel -= self.kernel_mean

        # obtain weigth matrix
        self.wx = self.train_sgh_seq(kernel, p_x, q_x)
        return

    def train_sgh_seq(self, kernel, p_x, q_x):
        n_dims = kernel.shape[1]
        gamma = 1e-6

        a_1 = self.projections * np.dot(kernel.T, p_x).dot(np.dot(kernel.T, q_x).T)
        z = np.dot(kernel.T, kernel) + gamma * np.eye(n_dims)

        wx = np.random.randn(n_dims, self.projections)

        for i in range(self.projections):
            eigvalues, eigvectors = alg.eigh(a_1, z, eigvals=(n_dims - 1, n_dims - 1))
            wx_i = eigvectors
            vx = np.dot(kernel.T, np.sign(np.dot(kernel, wx_i)))
            a_1 = a_1 - np.dot(vx, vx.T)
            wx[:, i] = wx_i.squeeze()

        rand_perm = np.random.permutation(self.projections)

        iterations = 1
        for j in range(iterations):
            for i in range(self.projections):
                wx_i = wx[:, rand_perm[i]].reshape(-1, 1)
                vx = np.dot(kernel.T, np.sign(np.dot(kernel, wx_i)))
                a_1 = a_1 + np.dot(vx, vx.T)

                eigvalues, eigvectors = alg.eigh(a_1, z, eigvals=(n_dims - 1, n_dims - 1))
                wx_i = eigvectors
                vx = np.dot(kernel.T, np.sign(np.dot(kernel, wx_i)))
                a_1 = a_1 - np.dot(vx, vx.T)

                wx[:, rand_perm[i]] = wx_i.squeeze()

        return wx

    def dist_matrix(self, x):
        """ calculates distance matrix between x and the selected base vectors. """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        (rows_x, cols_x) = x.shape
        (rows_y, cols_y) = self.bases.shape
        assert cols_x == cols_y, f'vectors dimensions must match with the bases ({cols_x} != {cols_y})'

        xy = np.dot(x, self.bases.T)
        x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rows_x, 1)), repeats=rows_y, axis=1)
        y2 = np.repeat(self.bases_2, repeats=rows_x, axis=1).T
        return x2 + y2 - 2 * xy
