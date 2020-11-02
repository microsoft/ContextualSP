# coding: utf-8

import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


class BipartiteGraphSolver:
    solver = linear_sum_assignment

    @staticmethod
    def find_min(cost_matrix):
        row_ind, col_ind = BipartiteGraphSolver.solver(cost_matrix)
        sum_cost = cost_matrix[row_ind, col_ind].sum()
        return sum_cost, (row_ind, col_ind)

    @staticmethod
    def find_max(value_matrix):
        cost_matrix = -value_matrix
        sum_cost, (row_ind, col_ind) = BipartiteGraphSolver.find_min(cost_matrix)
        return -sum_cost, (row_ind, col_ind)


if __name__ == '__main__':
    # cost_mat = np.random.random((30, 50))
    cost_mat = np.array([[1, 3, 5, 7], [2, 6, 5, 8], [7, 0, 3, 6]])
    st_time = time.time()
    for _ in tqdm(range(100)):
        cost, (row_ind, col_ind) = BipartiteGraphSolver.find_max(cost_mat)
    ed_time = time.time()
    print(f'average calculation time = {(ed_time - st_time) / 10000}')
    print(cost)
    print(row_ind, col_ind)

    a = list(range(100))
    with tqdm(a) as tqdm_iter:
        for idx, _ in enumerate(tqdm_iter):
            print(f'{idx}: {_ ** 2}')
