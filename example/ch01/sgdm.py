import numpy as np
import math
import matplotlib.pyplot as plt


def SGD_Momentum(lad, x):
    y = 0
    while x > 0:
        y += math.pow(lad, x - 1)
        x -= 1

    return y


def _curve_sgd_m(x_array):
    y = []

    for x in x_array:
        y.append(SGD_Momentum(0.25, x))
    return y


# 动量 v 关于 迭代次数 的函数曲线
if __name__ == '__main__':
    x = np.arange(1, 10, 1)
    y = _curve_sgd_m(x)

    plt.plot(x, y, label="sgdm")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.title("momentum(epoch)")
    plt.legend()
    plt.show()
