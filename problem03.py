#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Description:
    Author:LYC
    Time:2022/3/29 20:42
    Development Tool:PyCharm
"""
from test_function import BasicFunction

import numpy as np


class Func03(BasicFunction):
    def calculate(self, x: np.ndarray):
        """

        :param x:(2,1,N,N)
        :return: (1,N,N)
        """
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def gradient(self, x: np.ndarray):
        """

        :param x: (2,1)
        :return: (2,1)
        """
        g = 200 * (x[1] - x[0] ** 2)
        return np.stack((g * (-2 * x[0]) + 2 * (x[0] - 1), g), axis=0)

    def __init__(self):
        super(Func03, self).__init__()


class Goldstein:
    def __init__(self):
        pass

    def calculate(self, func: BasicFunction, x0: np.ndarray, d0: np.ndarray, p=0.1, alpha=1.5, beta=0.5, min_gamma=0,
                  max_gamma=10):
        assert min_gamma < max_gamma
        x_k = x0
        d_k = d0
        gamma_l = min_gamma
        gamma_r = max_gamma
        gamma_k = 1.
        g_x_k = func.gradient(x_k)
        f_x_k = func.calculate(x_k)
        while gamma_l < gamma_r:
            s_k = gamma_k * d_k
            f_x_ka1 = func.calculate(x_k + s_k)
            if f_x_ka1 - f_x_k <= p * g_x_k.T @ s_k:
                if f_x_ka1 - f_x_k >= (1 - p) * g_x_k.T @ s_k:
                    break
                else:
                    gamma_l = gamma_k
                    gamma_k *= alpha
            else:
                gamma_r = gamma_k
                gamma_k *= beta
        x_k += s_k
        return gamma_k, func.calculate(x_k)


class GoldsteinPrice:
    def __init__(self):
        pass

    def calculate(self, func: BasicFunction, x0: np.ndarray, d0: np.ndarray, p=0.1, tho=0.9, alpha=1.5, beta=0.5,
                  min_gamma=0,
                  max_gamma=10):
        assert min_gamma < max_gamma
        x_k = x0
        d_k = d0
        gamma_l = min_gamma
        gamma_r = max_gamma
        gamma_k = 1.
        g_x_k = func.gradient(x_k)
        f_x_k = func.calculate(x_k)
        while gamma_l < gamma_r:
            s_k = gamma_k * d_k
            f_x_ka1 = func.calculate(x_k + s_k)
            if f_x_ka1 - f_x_k <= p * g_x_k.T @ s_k:
                if f_x_ka1 - f_x_k >= tho * g_x_k.T @ s_k:
                    break
                else:
                    gamma_l = gamma_k
                    gamma_k *= alpha
            else:
                gamma_r = gamma_k
                gamma_k *= beta
        x_k += s_k
        return gamma_k, func.calculate(x_k)


class WolfePowell:
    def __init__(self):
        pass

    def calculate(self, func: BasicFunction, x0: np.ndarray, d0: np.ndarray, p=0.1, tho=0.9, alpha=1.5, beta=0.5,
                  min_gamma=0,
                  max_gamma=10):
        assert min_gamma < max_gamma
        x_k = x0
        d_k = d0
        gamma_l = min_gamma
        gamma_r = max_gamma
        gamma_k = 1.
        g_x_k = func.gradient(x_k)
        f_x_k = func.calculate(x_k)
        while gamma_l < gamma_r:
            s_k = gamma_k * d_k
            f_x_ka1 = func.calculate(x_k + s_k)
            g_x_ka1 = func.gradient(x_k + s_k)
            if f_x_ka1 - f_x_k <= p * g_x_k.T @ s_k:
                if g_x_ka1.T @ s_k >= tho * g_x_k.T @ s_k:
                    break
                else:
                    gamma_l = gamma_k
                    gamma_k *= alpha
            else:
                gamma_r = gamma_k
                gamma_k *= beta
        x_k += s_k
        return gamma_k, func.calculate(x_k)


if __name__ == '__main__':
    x = np.array([-1, 1], float).reshape((2, 1))
    d = np.array([1, 1], float).reshape((2, 1))
    f = Func03()
    print("=======================Goldstein法=======================")
    goldstein = Goldstein()
    print(goldstein.calculate(f, x.copy(), d.copy(), p=0.1, alpha=1.5, beta=0.5))
    print("=======================Goldstein-Price法=======================")
    goldsteinprice = GoldsteinPrice()
    print(goldsteinprice.calculate(f, x.copy(), d.copy(), p=0.1, tho=0.5, alpha=1.5, beta=0.5))
    print("=======================Wolfe-Powell法=======================")
    wolfepowell = WolfePowell()
    print(wolfepowell.calculate(f, x.copy(), d.copy(), p=0.1, tho=0.5, alpha=1.5, beta=0.5))
