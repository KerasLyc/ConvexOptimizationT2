import numpy as np

from test_function import BasicFunction


class Func01(BasicFunction):
    def __init__(self, c: np.ndarray, H: np.ndarray):
        """

        :param c:(2,1)
        :param H: (2,2)
        """
        super(Func01, self).__init__()
        # self.c = np.array([0, -12], dtype=float).reshape((2, 1))
        # self.H = np.array([[8, 4], [4, 8]], dtype=float).reshape((2, 2))
        self.c = c
        self.H = H
        # self.c = np.array([1, -1], dtype=float).reshape((2, 1))
        # self.H = np.array([[4, 2], [2, 2]], dtype=float).reshape((2, 2))

    def calculate(self, x: np.ndarray):
        """
        计算函数值
        :param x:(2,1)
        :return:(1,1)
        """
        assert x.shape == (2, 1)
        # return x[0] - x[1] + 2 * x[0] ** 2 + 2 * x[0] * x[1] + x[1] ** 2
        return self.c.T @ x + 1 / 2 * x.T @ self.H @ x

    def gradient(self, x: np.ndarray):
        """
        计算梯度
        :param x:(2,1)
        :return:(2,1)
        """
        assert x.shape == (2, 1)
        return self.c + self.H @ x


class ConjugateGradientMethodForQuadraticFunction:
    """
    用于计算二次函数的(局部)最小值的共轭梯度法
    计算二次函数：f(x)=c^T@x+x^T@H@x/2，的（局部）最小值
    """

    def __init__(self, func: Func01, x1: np.ndarray, d1: np.ndarray):
        self.func = func
        self.x1 = x1
        self.d1 = d1

    def calc_min(self):
        """
        算法计算过程
        :return:(x_local_min,f_local_min):((2,1),(1,1))
        """
        dk = self.d1
        xk = self.x1
        k = 1
        for i in range(self.x1.shape[0]):
            gamma_k = self.clac_gamma_k(dk)

            self.show_status(k, xk, dk, gamma_k)
            k += 1
            xk = xk + gamma_k * dk
            dk = self.calc_d_ka1(xk, dk)
        local_min = self.func.calculate(xk)
        return xk, local_min

    def show_status(self, k, xk, dk, gamma_k):
        print('k={},x_k=({},{})^T,f(x_k)={},d_k=({},{})^T,gamma_k={}'.format(
            k,
            xk[0, 0], xk[1, 0],
            self.func.calculate(xk)[0, 0],
            dk[0, 0], dk[1, 0],
            gamma_k[0, 0]))

    def clac_gamma_k(self, d_k: np.ndarray):
        """
        计算步长
        :param d_k:
        :return:(1,1)
        """
        return -(self.func.c.T @ d_k + self.x1.T @ self.func.H @ d_k) / (d_k.T @ self.func.H @ d_k)

    def calc_beta_k(self, x_ka1: np.ndarray, d_k: np.ndarray):
        """
        d_{k+1}=-g_k+beta_k*d_k
        根据关于H共轭计算beta_k
        :param x_ka1:
        :param d_k:
        :return:(1,1)
        """
        g_ka1 = self.func.gradient(x_ka1)
        return (g_ka1.T @ self.func.H @ d_k) / (d_k.T @ self.func.H @ d_k)

    def calc_d_ka1(self, x_ka1: np.ndarray, d_k: np.ndarray):
        """
        计算下一个共轭向量d_{k+1}=-g_k+beta_k*d_k
        :param x_ka1:
        :param d_k:
        :return:(2,1)
        """
        beta_k = self.calc_beta_k(x_ka1, d_k)
        return -self.func.gradient(x_ka1) + beta_k * d_k


if __name__ == '__main__':
    c = np.array([1, -1], dtype=float).reshape((2, 1))
    H = np.array([[4, 2], [2, 2]], dtype=float).reshape((2, 2))
    f = Func01(c=c, H=H)

    x1 = np.array([0, 0], dtype=float).reshape((2, 1))
    d1 = np.array([1, 0], dtype=float).reshape((2, 1))
    CGM = ConjugateGradientMethodForQuadraticFunction(f, x1, d1)

    x_lmin, f_lmin = CGM.calc_min()
    print('x*=({},{})^T,f(x*)={}'.format(x_lmin[0, 0], x_lmin[1, 0], f_lmin[0, 0]))
