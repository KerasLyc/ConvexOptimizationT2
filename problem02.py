#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Description:
    Author:LYC
    Time:2022/3/29 10:14
    Development Tool:PyCharm
"""
import numpy as np
import copy

from test_function import BasicFunction


class Func02a(BasicFunction):
    def __init__(self):
        super(Func02a, self).__init__()

    def calculate(self, x):
        """

        :param x:
        :return:
        """
        return 2 * x ** 2 - x - 1

    def gradient(self, x):
        return 4 * x - 1


class Func02b(BasicFunction):
    def calculate(self, x):
        """

        :param x:
        :return:
        """
        return 3 * x ** 2 - 21.6 * x - 1

    def gradient(self, x):
        return 6 * x - 21.6

    def __init__(self):
        super(Func02b, self).__init__()


class GoldenSection:
    """
    黄金分割法
    """

    def __init__(self):
        self.alpha = 0.618

    def calculate(self, f: BasicFunction, a0: float, b0: float, delta: float):
        a_k = a0
        b_k = b0
        k = 1
        while b_k - a_k > delta:
            gamma_k = a_k + (1 - self.alpha) * (b_k - a_k)
            u_k = a_k + self.alpha * (b_k - a_k)
            f_gamma_k = f.calculate(gamma_k)
            f_u_k = f.calculate(u_k)
            self.show_status(k, a_k, b_k, gamma_k, u_k, f_gamma_k, f_u_k)
            if f_gamma_k > f_u_k:
                a_k = gamma_k
            else:
                b_k = u_k
            k += 1
        return a_k, b_k

    def show_status(self, k, a_k, b_k, gamma_k, u_k, f_gamma_k, f_u_k):
        print('k={},a_k={},b_k={},gamma_k={},u_k={},f_gamma_k={},f_u_k={}'.format(
            k, a_k, b_k, gamma_k, u_k, f_gamma_k, f_u_k
        ))


class FibonacciNumbers:
    """
    斐波那契数列法
    """

    def __init__(self, max_fib_num: int):
        self.F = np.ones(max_fib_num, dtype=float)
        for i in range(3, self.F.shape[0]):
            self.F[i] = self.F[i - 1] + self.F[i - 2]

    def calculate(self, f: BasicFunction, a0: float, b0: float, delta: float):
        a_k = a0
        b_k = b0
        floor_Fn = (b0 - a0) / delta
        _, n = np.unique(np.greater_equal(self.F, floor_Fn), return_index=True)
        n = n[-1]
        k = 1
        while k < n:
            gamma_k = a_k + self.F[n - k - 1] / self.F[n - k + 1] * (b_k - a_k)
            u_k = a_k + self.F[n - k] / self.F[n - k + 1] * (b_k - a_k)
            f_gamma_k = f.calculate(gamma_k)
            f_u_k = f.calculate(u_k)
            self.show_status(k, a_k, b_k, gamma_k, u_k, f_gamma_k, f_u_k)
            if f_gamma_k > f_u_k:
                a_k = gamma_k
            else:
                b_k = u_k
            k += 1
        return a_k, b_k

    def show_status(self, k, a_k, b_k, gamma_k, u_k, f_gamma_k, f_u_k):
        print('k={},a_k={},b_k={},gamma_k={},u_k={},f_gamma_k={},f_u_k={}'.format(
            k, a_k, b_k, gamma_k, u_k, f_gamma_k, f_u_k
        ))


class MidpointMethod:
    """
    二分法
    """

    def __init__(self):
        pass

    def calculate(self, f: BasicFunction, a0: float, b0: float, delta: float):
        k = 1
        a_k = a0
        b_k = b0
        while b_k - a_k >= delta:
            gamma_k = (a_k + b_k) / 2
            gradient_gamma_k = f.gradient(gamma_k)
            self.show_status(k, a_k, b_k, gamma_k, gradient_gamma_k)
            if gradient_gamma_k > 0:
                b_k = gamma_k
            elif gradient_gamma_k < 0:
                a_k = gamma_k
            elif gradient_gamma_k == 0:
                a_k = gamma_k
                b_k = gamma_k
                break
            k += 1
        return a_k, b_k

    def show_status(self, k, a_k, b_k, gamma_k, gradient_gamma_k):
        print('k={},a_k={},b_k={},gamma_k={},gradient_gamma_k={}'.format(
            k, a_k, b_k, gamma_k, gradient_gamma_k
        ))


class Dichotomous:
    """
    二分法
    """

    def __init__(self):
        pass

    def calculate(self, f: BasicFunction, a0: float, b0: float, r: float, delta: float):
        k = 1
        a_k = a0
        b_k = b0
        while b_k - a_k >= delta:
            gamma_k = (a_k + b_k) / 2 - r
            u_k = (a_k + b_k) / 2 + r
            f_gamma_k = f.calculate(gamma_k)
            f_u_k = f.calculate(u_k)
            self.show_status(k, a_k, b_k, gamma_k, u_k, f_gamma_k, f_u_k)
            if f_gamma_k < f_u_k:
                b_k = u_k
            else:
                a_k = gamma_k
            k += 1
        return a_k, b_k

    def show_status(self, k, a_k, b_k, gamma_k, u_k, f_gamma_k, f_u_k):
        print('k={},a_k={},b_k={},gamma_k={},u_k={},f_gamma_k={},f_u_k={}'.format(
            k, a_k, b_k, gamma_k, u_k, f_gamma_k, f_u_k
        ))


class ShubertPiyavskii:
    """
    ShubertPiyavskii方法求全局极小点
    """

    def __init__(self):
        pass

    def calculate(self, func: BasicFunction, a0: float, b0: float, l: float, delta: float):
        # 初始化，在两端变为上锯齿点后，所有下锯齿点为奇数索引，上锯齿点为偶数索引
        x = [a0, (a0 + b0) / 2, b0]
        f = [func.calculate(i) for i in x]
        up_min_index = 1
        y = [f[up_min_index] - l * (x[1] - x[0]), f[up_min_index], f[up_min_index] - l * (x[2] - x[1])]
        # 获取初始最小下锯齿点索引
        down_min_index = ShubertPiyavskii.get_min_index(y, 0, 2)
        while f[down_min_index] - y[down_min_index] > delta:
            delta_y = f[down_min_index] - y[down_min_index]
            delta_x = delta_y / (2 * l)
            # 计算新的两个下锯齿点x
            new_down_x1 = x[down_min_index] - delta_x
            new_down_x2 = x[down_min_index] + delta_x
            # 原最小下锯齿点变为上锯齿点
            y[down_min_index] = f[down_min_index]
            # 计算新的两个下锯齿点x
            new_down_y1 = y[down_min_index] - delta_y / 2
            new_down_y2 = y[down_min_index] - delta_y / 2
            # 更新列表
            if new_down_x2 <= b0:
                x.insert(down_min_index + 1, new_down_x2)
                f.insert(down_min_index + 1, func.calculate(new_down_x2))
                y.insert(down_min_index + 1, new_down_y2)
            if new_down_x1 >= a0:
                x.insert(down_min_index, new_down_x1)
                f.insert(down_min_index, func.calculate(new_down_x1))
                y.insert(down_min_index, new_down_y1)
            down_min_index = ShubertPiyavskii.get_min_index(y, 1, 2)
        return x[down_min_index - 1], x[down_min_index + 1]

    @staticmethod
    def get_min_index(x, start, step):
        min_value = x[0]
        k = 0
        for i in range(start, len(x), step):
            if x[i] < min_value:
                min_value = x[i]
                k = i
        return k


if __name__ == '__main__':
    af = Func02a()
    bf = Func02b()
    print("=======================黄金分割法=======================")
    golden_sec = GoldenSection()
    print('问题（a）的局部最小值在区间:({},{})\n'.format(*golden_sec.calculate(af, -1, 1, 0.06)))
    print('问题（b）的局部最小值在区间:({},{})\n'.format(*golden_sec.calculate(bf, 0, 25, 0.08)))

    print("=======================斐波那契数列法=======================")
    fib_sec = FibonacciNumbers(30)
    print('问题（a）的局部最小值在区间:({},{})\n'.format(*fib_sec.calculate(af, -1, 1, 0.06)))
    print('问题（b）的局部最小值在区间:({},{})\n'.format(*fib_sec.calculate(bf, 0, 25, 0.08)))
    print("=======================中点法=======================")
    mid_point = MidpointMethod()
    print('问题（a）的局部最小值在区间:({},{})\n'.format(*mid_point.calculate(af, -1, 1, 0.06)))
    print('问题（b）的局部最小值在区间:({},{})\n'.format(*mid_point.calculate(bf, 0, 25, 0.08)))
    print("=======================Dichotomous法=======================")
    dicho = Dichotomous()
    print('问题（a）的局部最小值在区间:({},{})\n'.format(*dicho.calculate(af, -1, 1, 0.01, 0.06)))
    print('问题（b）的局部最小值在区间:({},{})\n'.format(*dicho.calculate(bf, 0, 25, 0.01, 0.08)))
    print("=======================ShubertPiyavskii法=======================")
    shuber = ShubertPiyavskii()
    print('问题（a）的局部最小值在区间:({},{})\n'.format(*shuber.calculate(af, -1, 1, 200, 0.06)))
    print('问题（b）的局部最小值在区间:({},{})\n'.format(*shuber.calculate(bf, 0, 25, 200, 0.08)))
