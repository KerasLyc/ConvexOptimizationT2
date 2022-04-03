#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Description:
    Author:LYC
    Time:2022/3/30 14:40
    Development Tool:PyCharm
"""
import numpy as np
from test_function import BasicFunction


class testF(BasicFunction):
    def __init__(self):
        super(testF, self).__init__()

    def calculate(self, x: np.ndarray):
        """

        :param x:(2,1)
        :return:
        """
        return (x[0] - 2) ** 4 + (x[0] - 2 * x[1]) ** 2

    def gradient(self, x: np.ndarray):
        """

        :param x: (2,1)
        :return:
        """
        return np.stack((4 * (x[0] - 2) ** 3 + 2 * (x[0] - 2 * x[1]),
                         -4 * (x[0] - 2 * x[1])))


class Func05(BasicFunction):
    def calculate(self, x):
        return 10 * x[0] ** 2 + x[1] ** 2

    def gradient(self, x):
        return np.stack(
            (20 * x[0],
             2 * x[1])
        )

    def __init__(self):
        super(Func05, self).__init__()


class DFP:
    def __init__(self, func: BasicFunction, x1: np.ndarray, golden_sec_max_gamma=10, golden_sec_delta=0.001):
        self.func = func
        self.x1 = x1
        self.golden_sec_max_gamma = golden_sec_max_gamma
        self.golden_sec_delta = golden_sec_delta

    def cal_min(self, delta):
        x_j = self.x1
        k = 1
        g_x_j = self.func.gradient(x_j)
        g_x_ja1 = g_x_j
        while np.linalg.norm(g_x_ja1) > delta:
            # 初始化D为单位矩阵
            D_j = np.identity(self.x1.shape[0])
            for j in range(1, self.x1.shape[0] + 1):
                # g_x_j
                g_x_j = g_x_ja1
                # 如果g_x_j达到精度要求，停机
                if np.linalg.norm(g_x_j) <= delta:
                    break
                # 计算搜索方向
                d_j = -D_j @ g_x_j
                # 计算搜索步长
                gamma_j = self.cal_gamma_j(x_j, d_j)
                p_j = gamma_j * d_j
                self.show_status(k, j, x_j, g_x_j, D_j, d_j, gamma_j, x_j + p_j)
                # x_ja1
                x_j = x_j + p_j
                g_x_ja1 = self.func.gradient(x_j)
                q_j = g_x_ja1 - g_x_j
                D_j = D_j + (p_j @ p_j.T) / (p_j.T @ q_j) - (D_j @ q_j @ q_j.T @ D_j) / (q_j.T @ D_j @ q_j)
            k += 1
        print(
            '******Final******\ny_j=({},{}),g_j=({},{}),\n\tD={}\n'.format(
                x_j[0, 0], x_j[1, 0], g_x_ja1[0, 0], g_x_ja1[1, 0], D_j
            ))
        return x_j, self.func.calculate(x_j)

    def cal_gamma_j(self, x_j, d_j):
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

    def show_status(self, k, j, y_j, g_j, D, d_j, gamma_j, y_ja1):
        print(
            '******k={},j={}******\ny_j=({},{}),g_j=({},{}),\n\tD={},\n\td_j=({},{}),gamma_j={},y_ja1=({},{})\n'.format(
                k, j, y_j[0, 0], y_j[1, 0], g_j[0, 0], g_j[1, 0], D, d_j[0, 0], d_j[1, 0], gamma_j, y_ja1[0, 0],
                y_ja1[1, 0]
            ))


if __name__ == '__main__':
    f = Func05()
    x1 = np.array((1 / 10, 1), float).reshape((2, 1))
    dfp = DFP(f, x1, 10)
    min_x, minf = dfp.cal_min(delta=0.01)
    print('最优解:({},{}),最优解为:{}'.format(min_x[0, 0], min_x[1, 0], minf[0]))
