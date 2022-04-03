#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Description:
    Author:LYC
    Time:2022/4/1 16:33
    Development Tool:PyCharm
"""
import numpy as np

from test_function import BasicFunction
from problem05 import DFP
from problem06 import BFGS


class Func07(BasicFunction):
    def calculate(self, x: np.ndarray):
        return x[0] ** 2 + x[1] ** 2 - x[0] * x[1] - 10 * x[0] - 4 * x[1] + 60

    def gradient(self, x):
        return np.stack(
            (
                2 * x[0] - x[1] - 10,
                2 * x[1] - x[0] - 4
            ), axis=0
        )

    def __init__(self):
        super(Func07, self).__init__()
        self.H = np.array(
            [[2, -1],
             [-1, 2]]
        ).reshape((2, 2))
        self.c = np.array(
            [-10, -4]
        ).reshape((2, 1))


class FRConjugateGradientMethod:
    def __init__(self, func: Func07, x1: np.ndarray, golden_sec_max_gamma=10,
                 golden_sec_delta=0.001):
        self.x1 = x1
        self.func = func
        self.golden_sec_delta = golden_sec_delta
        self.golden_sec_max_gamma = golden_sec_max_gamma
        self.d1 = -self.func.gradient(self.x1)

    def calc_min(self):
        dk = self.d1
        xk = self.x1
        k = 1
        for i in range(self.x1.shape[0]):
            gamma_k = self.calc_gamma_j(xk, dk)
            g_k=self.func.gradient(xk)
            self.show_status(k, xk, dk, gamma_k)
            k += 1
            xk = xk + gamma_k * dk
            dk = self.calc_d_ka1(xk, dk,g_k)
        local_min = self.func.calculate(xk)
        return xk, local_min

    def show_status(self, k, xk, dk, gamma_k):
        print('k={},x_k=({},{})^T,f(x_k)={},d_k=({},{})^T,gamma_k={}'.format(
            k,
            xk[0, 0], xk[1, 0],
            self.func.calculate(xk)[0],
            dk[0, 0], dk[1, 0],
            gamma_k))

    def calc_gamma_j(self, x_j, d_j):
        return -(d_j.T @ (self.func.H @ x_j + self.func.c)) / (d_j.T @ self.func.H @ d_j)

    def calc_beta_k(self, x_ka1: np.ndarray, g_k:np.ndarray):
        """
        d_{k+1}=-g_k+beta_k*d_k
        根据关于H共轭计算beta_k
        :param x_ka1:
        :param d_k:
        :return:(1,1)
        """
        g_ka1 = self.func.gradient(x_ka1)
        return (g_ka1.T @ g_ka1) / (g_k.T @ g_k)

    def calc_d_ka1(self, x_ka1: np.ndarray, d_k: np.ndarray,g_k:np.ndarray):
        """
        计算下一个共轭向量d_{k+1}=-g_k+beta_k*d_k
        :param x_ka1:
        :param d_k:
        :return:(2,1)
        """
        beta_k = self.calc_beta_k(x_ka1, g_k)
        return -self.func.gradient(x_ka1) + beta_k * d_k


if __name__ == '__main__':
    x1 = np.array((0, 0), float).reshape((2, 1))
    f = Func07()
    dfp = DFP(f, x1.copy())
    bfgs = BFGS(f, x1.copy())
    fr = FRConjugateGradientMethod(f, x1.copy())
    print("==========DFP方法==========")
    min_x, minf = dfp.cal_min(delta=0.01)
    print('最优解:({},{}),最优解为:{}'.format(min_x[0, 0], min_x[1, 0], minf[0]))

    print("==========BFGS方法==========")
    min_x, minf = bfgs.cal_min(delta=0.01)
    print('最优解:({},{}),最优解为:{}'.format(min_x[0, 0], min_x[1, 0], minf[0]))
    print("==========FR共轭梯度法==========")
    min_x, minf = fr.calc_min()
    print('最优解:({},{}),最优解为:{}'.format(min_x[0, 0], min_x[1, 0], minf[0]))
