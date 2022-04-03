#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Description:
    Author:LYC
    Time:2022/4/2 16:22
    Development Tool:PyCharm
"""
import numpy as np
from test_function import BasicFunction
from problem07 import Func07


def Momentum(func: BasicFunction, x0: np.ndarray,
             beta=0.6, alpha=0.4, delta=0.01):
    xk = x0
    vk = 0
    k = 1
    gk = func.gradient(x0)
    while np.linalg.norm(gk) > delta:
        vk = beta * vk - alpha * gk
        print('k={},x_k=({},{}),g_k=({},{}),v_k=({},{})'.format(k, xk[0, 0], xk[1, 0], gk[0, 0], gk[1, 0], vk[0, 0],
                                                                vk[1, 0]), end=',')
        xk = xk + vk
        gk = func.gradient(xk)
        print('x_ka1=({},{})'.format(xk[0, 0], xk[1, 0]))
        k += 1

    return xk, func.calculate(xk)


def NesterovMomentum(func: BasicFunction, x0: np.ndarray,
                     beta=0.6, alpha=0.4, delta=0.01):
    xk = x0
    vk = 0
    k = 1
    gk = func.gradient(x0 + beta * vk)
    while np.linalg.norm(gk) > delta:
        vk = beta * vk - alpha * func.gradient(xk + beta * vk)
        print('k={},x_k=({},{}),g_k=({},{}),v_k=({},{})'.format(k, xk[0, 0], xk[1, 0], gk[0, 0], gk[1, 0], vk[0, 0],
                                                                vk[1, 0]), end=',')
        xk = xk + vk
        gk = func.gradient(xk)
        print('x_ka1=({},{})'.format(xk[0, 0], xk[1, 0]))
        k += 1
    return xk, func.calculate(xk)


def Adagrad(func: BasicFunction, x0: np.ndarray, alpha=0.1, delta=0.01):
    k = 1
    xk = x0
    sk = np.zeros_like(xk)
    gk = func.gradient(xk)
    while np.linalg.norm(gk) > delta:
        sk += gk ** 2
        print('k={},x_k=({},{}),g_k=({},{}),s_k=({},{})'.format(k, xk[0, 0], xk[1, 0], gk[0, 0], gk[1, 0], sk[0, 0],
                                                                sk[1, 0]), end=',')
        xk = xk - alpha * gk / (1e-8 + np.sqrt(sk))
        gk = func.gradient(xk)
        print('x_ka1=({},{})'.format(xk[0, 0], xk[1, 0]))
        k += 1
    return xk, func.calculate(xk)


def RMSProp(func: BasicFunction, x0: np.ndarray, alpha=0.9, r=0.9, delta=0.01):
    k = 1
    xk = x0
    sk = np.zeros_like(xk)
    gk = func.gradient(xk)
    while np.linalg.norm(gk) > delta:
        sk = r * sk + (1 - r) * gk ** 2
        print('k={},x_k=({},{}),g_k=({},{}),s_k=({},{})'.format(k, xk[0, 0], xk[1, 0], gk[0, 0], gk[1, 0], sk[0, 0],
                                                                sk[1, 0]), end=',')
        xk = xk - alpha * gk / (1e-8 + np.sqrt(sk))
        gk = func.gradient(xk)
        print('x_ka1=({},{})'.format(xk[0, 0], xk[1, 0]))
        k += 1
    return xk, func.calculate(xk)


def Adadelta(func: BasicFunction, x0: np.ndarray, r=0.9, delta=0.01):
    k = 1
    t = 1e-5
    xk = x0
    rms2_gk = np.zeros_like(xk)
    rms2_delta_x = np.zeros_like(xk)
    gk = func.gradient(xk)
    print('迭代次数过多，如需显示，请取消注释函数中的被注释部分')
    while np.linalg.norm(gk) > delta:
        rms2_gk = r * rms2_gk + (1 - r) * gk ** 2
        delta_x = - (np.sqrt(rms2_delta_x) + t) * gk / (t + np.sqrt(rms2_gk))
        rms2_delta_x = r * rms2_delta_x + (1 - r) * delta_x ** 2
        # print('k={},x_k=({},{}),g_k=({},{})'.format(k, xk[0, 0], xk[1, 0], gk[0, 0], gk[1, 0]), end=',')
        xk = xk + delta_x
        gk = func.gradient(xk)
        # print('x_ka1=({},{})'.format(xk[0, 0], xk[1, 0]))
        k += 1
    return xk, func.calculate(xk)


def Adam(func: BasicFunction, x0: np.ndarray, alpha=0.01, r_v=0.9, r_s=0.999, delta=0.01):
    k = 1
    t = 1e-8
    xk = x0
    vka1 = np.zeros_like(xk)
    ska1 = 0
    gk = func.gradient(xk)
    print('迭代次数过多，如需显示，请取消注释函数中的被注释部分')
    while np.linalg.norm(gk) > delta:
        # print('k={},x_k=({},{}),g_k=({},{})'.format(k, xk[0, 0], xk[1, 0], gk[0, 0], gk[1, 0]), end=',')
        vka1 = r_v * vka1 + (1 - r_v) * gk
        ska1 = r_s * ska1 + (1 - r_s) * gk ** 2
        vka1_check = vka1 / (1 - r_v ** k)
        ska1_check = ska1 / (1 - r_s ** k)
        xk = xk - alpha * vka1_check / (t + np.sqrt(ska1_check))
        # print('x_ka1=({},{})'.format(xk[0, 0], xk[1, 0]))
        gk = func.gradient(xk)
        k += 1
    return xk, func.calculate(xk)


if __name__ == '__main__':
    x1 = np.array((0, 0), float).reshape((2, 1))
    f = Func07()
    print('==========Momentum方法==========')
    min_x, min_f = Momentum(f, x1.copy(), beta=0.6, alpha=0.4, delta=1e-2)
    print('最优x=({},{}),最优解={}\n'.format(min_x[0, 0], min_x[1, 0], min_f[0]))
    print('==========NesterovMomentum方法==========')
    min_x, min_f = NesterovMomentum(f, x1.copy(), beta=0.6, alpha=0.4, delta=1e-2)
    print('最优x=({},{}),最优解={}\n'.format(min_x[0, 0], min_x[1, 0], min_f[0]))
    print('==========Adagrad方法==========')
    min_x, min_f = Adagrad(f, x1.copy(), alpha=5, delta=0.01)
    print('最优x=({},{}),最优解={}\n'.format(min_x[0, 0], min_x[1, 0], min_f[0]))
    print('==========RMSProp方法==========')
    min_x, min_f = RMSProp(f, x1.copy(), alpha=0.6, r=0.9, delta=0.01)
    print('最优x=({},{}),最优解={}\n'.format(min_x[0, 0], min_x[1, 0], min_f[0]))
    print('==========Adadelta方法==========')
    min_x, min_f = Adadelta(f, x1.copy(), r=0.1, delta=0.01)
    print('最优x=({},{}),最优解={}\n'.format(min_x[0, 0], min_x[1, 0], min_f[0]))
    print('==========Adam方法==========')
    min_x, min_f = Adam(f, x1.copy(), alpha=0.3, r_v=0.9, r_s=0.999, delta=0.01)
    print('最优x=({},{}),最优解={}\n'.format(min_x[0, 0], min_x[1, 0], min_f[0]))
