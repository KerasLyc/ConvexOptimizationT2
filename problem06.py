#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Description:
    Author:LYC
    Time:2022/3/30 17:30
    Development Tool:PyCharm
"""
import numpy as np
from test_function import BasicFunction


class Func06(BasicFunction):
    def calculate(self, x: np.ndarray):
        return x[0] ** 2 + 4 * x[1] ** 2 - 4 * x[0] - 8 * x[1]

    def gradient(self, x):
        return np.stack((
            2 * x[0] - 4,
            8 * x[1] - 8
        ), axis=0)

    def __init__(self):
        super(Func06, self).__init__()


class BFGS:
    def __init__(self, func: BasicFunction, x1: np.ndarray, golden_sec_max_gamma=10, golden_sec_delta=0.001):
        self.func = func
        self.x1 = x1
        self.golden_sec_max_gamma = golden_sec_max_gamma
        self.golden_sec_delta = golden_sec_delta

    def cal_min(self, delta):
        x_j = self.x1
        k = 1
        g_x_j = self.func.gradient(x_j)
        g_x_ja1 = self.func.gradient(x_j)
        while np.linalg.norm(g_x_ja1) > delta:
            B_j = np.identity(self.x1.shape[0])
            for j in range(1, self.x1.shape[0] + 1):
                # g_x_j
                g_x_j = g_x_ja1
                # 计算搜索方向
                d_j = -np.linalg.inv(B_j) @ g_x_j
                # 计算搜索步长
                gamma_j = self.calc_gamma_j(x_j, d_j)
                # s_j
                s_j = gamma_j * d_j
                self.show_status(k, j, x_j, g_x_j, B_j, d_j, gamma_j, x_j + s_j)
                x_j = x_j + s_j
                g_x_ja1 = self.func.gradient(x_j)
                y_j = g_x_ja1 - g_x_j
                # B_ja1
                B_j = B_j - (B_j @ s_j @ s_j.T @ B_j) / (s_j.T @ B_j @ s_j) + (y_j @ y_j.T) / (y_j.T @ s_j)
            k += 1
        print(
            '******Final******\nx_j=({},{}),g_j=({},{}),\n\tB={}\n'.format(
                x_j[0, 0], x_j[1, 0], g_x_ja1[0, 0], g_x_ja1[1, 0], B_j
            ))
        return x_j, self.func.calculate(x_j)

    def calc_gamma_j(self, x_j, d_j):
        """
        使用一维精确搜索——黄金分割法求解gamma_j
        :param x_j:
        :param d_j:
        :return:
        """
        a_k = 0
        b_k = self.golden_sec_max_gamma
        while b_k > a_k + self.golden_sec_delta:
            gamma_k = a_k + (1 - 0.618) * (b_k - a_k)
            u_k = a_k + 0.618 * (b_k - a_k)
            f_gamma_k = self.func.calculate(x_j + gamma_k * d_j)
            f_u_k = self.func.calculate(x_j + u_k * d_j)
            if f_gamma_k > f_u_k:
                a_k = gamma_k
            else:
                b_k = u_k
        return (gamma_k + u_k) / 2

    def show_status(self, k, j, x_j, g_j, B, d_j, gamma_j, y_ja1):
        print(
            '******k={},j={}******\nx_j=({},{}),g_j=({},{}),f_j={},\n\tB={},\n\td_j=({},{}),gamma_j={},x_ja1=({},{})\n'.format(
                k, j, x_j[0, 0], x_j[1, 0], g_j[0, 0], g_j[1, 0], self.func.calculate(x_j)[0], B, d_j[0, 0], d_j[1, 0],
                gamma_j, y_ja1[0, 0],
                y_ja1[1, 0]
            ))


if __name__ == '__main__':
    f = Func06()
    x1 = np.array((0, 0), float).reshape((2, 1))
    bfgs = BFGS(f, x1, golden_sec_max_gamma=100, golden_sec_delta=0.01)
    min_x, minf = bfgs.cal_min(delta=0.01)
    print('最优解:({},{}),最优解为:{}'.format(min_x[0, 0], min_x[1, 0], minf[0]))
