#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Description:
    Author:LYC
    Time:2022/3/28 16:07
    Development Tool:PyCharm
"""
from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt


class BasicFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def calculate(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass

class Func06(BasicFunction):
    def calculate(self, x: np.ndarray):
        return x[0] ** 2 + 4 * x[1] ** 2 - 4 * x[0] - 8 * x[1]

    def gradient(self, x):
        return np.stack((
            2 * x[0] - x[1] - 10,
            2 * x[1] - x[1] - 4
        ),axis=0)

    def __init__(self):
        super(Func06, self).__init__()

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


class Ackley(BasicFunction):
    def __init__(self, a=20., b=0.2, c=2 * np.pi):
        super(Ackley, self).__init__()
        self.a = a
        self.b = b
        self.c = c

    def calculate(self, x: np.ndarray):
        """

        :param x:(dim,1,N)
        :return:(1,N)
        """
        d = x.shape[0]
        x2 = np.power(x, 2)
        sum_x2_dimwise = np.sum(x2, axis=0)  # (1,N)
        cos_cx = np.cos(self.c * x)
        sum_cos_cs_dimwise = np.sum(cos_cx, axis=0)  # (1,N)
        return -self.a * np.exp(-self.b * np.sqrt(sum_x2_dimwise / d)) - np.exp(sum_cos_cs_dimwise / d) + self.a + np.e


class Booth(BasicFunction):
    def __init__(self):
        super(Booth, self).__init__()
        pass

    def calculate(self, x: np.ndarray):
        """

        :param x: (2,1,N)
        :return:(1,N)
        """
        return np.power(x[0] + 2 * x[1] - 7, 2) + np.power(2 * x[0] + x[1] - 5, 2)


class Branin(BasicFunction):
    def __init__(self, a=1., b=5.1 / (4 * np.pi ** 2), c=5 * np.pi, r=6., s=10., t=1 / (8 * np.pi)):
        super(Branin, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.r = r
        self.s = s
        self.t = t

    def calculate(self, x: np.ndarray):
        """

        :param x:(2,1,N)
        :return:(1,N)
        """
        return self.a * np.power(x[1] - self.b * np.power(x[0], 2) + self.c * x[0] - self.r, 2) + self.s * (
                1 - self.t) * np.cos(x[0]) + self.s


class Flower(BasicFunction):
    def __init__(self, a=1., b=1., c=4.):
        super(Flower, self).__init__()
        self.a = a
        self.b = b
        self.c = c

    def calculate(self, x: np.ndarray):
        """

        :param x: (2,1,N)
        :return: (1,N)
        """
        return self.a * np.linalg.norm(x, ord=2, axis=0) + self.b * np.sin(self.c * np.arctan(x[1] / x[0]))


class Michalewicz(BasicFunction):
    def __init__(self, m=10.):
        super(Michalewicz, self).__init__()
        self.m = m

    def calculate(self, x: np.ndarray):
        """

        :param x:(dim,1,N)
        :return:(1,N)
        """
        sins = 0
        for i in range(x.shape[0]):
            sins += (-np.sin(x[i]) * np.sin((i + 1) * x[i] ** 2 / np.pi) ** (2 * self.m))
        return sins


class RosenbrockBanana(BasicFunction):
    def __init__(self, a=1., b=5.):
        super(RosenbrockBanana, self).__init__()
        self.a = a
        self.b = b

    def calculate(self, x):
        """

        :param x: (2,1,N)
        :return: (1,N)
        """
        return (self.a - x[0]) ** 2 - self.b * (x[1] - x[0] ** 2) ** 2


class Wheeler:
    def __init__(self, a=1.5):
        self.a = a

    def calculate(self, x):
        """

        :param x: (2,1,N)
        :return: (1,N)
        """
        return -np.exp(
            -(x[0] * x[1] - self.a) ** 2 - (x[1] - self.a) ** 2
        )


if __name__ == '__main__':
    N = 3000
    start = -10
    end = 20
    x0 = np.linspace(start, end, N)
    x1 = np.linspace(start, end, N)
    x = np.meshgrid(x0, x1)
    x = np.stack(x).reshape((-1, 1, N, N))
    print(x.shape)
    A = Func07()
    res = A.calculate(x)

    cset = plt.contourf(x[0, 0], x[1, 0], res.squeeze(), 20, cmap=plt.cm.hot)
    contour = plt.contourf(x[0, 0], x[1, 0], res.squeeze(), 20, cmap=plt.cm.hot)
    plt.clabel(contour, colors='k')
    plt.colorbar(cset)
    plt.show()

    print(A.calculate(np.array([1, 3 / 2], dtype=float).reshape((2, 1))))
