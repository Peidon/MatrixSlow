import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# 设置Matplotlib图形的全局默认大小
# plt.rc('figure', figsize=(10, 5))

##高斯分布函数
import math

if __name__ == '__main__':
    mu = 0
    sigma = 1
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 51)  # 51个数
    y = np.exp(-(x - mu) ** 2) / (2 * sigma ** 2) / math.sqrt(2 * math.pi * sigma ** 2)
    plt.figure(facecolor='w')  # 背景白色
    plt.plot(x, y, 'r-', alpha=0.5, linewidth=2, markeredgecolor='k')
    plt.xlabel('X', fontsize=15)  # 坐标轴
    plt.ylabel('Y', fontsize=15)  # 坐标轴
    plt.title('Gaussian distribution', fontsize=18)  # 标题
    # plt.fill_between(x, 0, y, color='b', alpha=0.5)  ##0是函数着色范围
    plt.grid(True)  # 方框
    plt.show()