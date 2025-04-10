import numpy as np
import matplotlib.pyplot as plt
#sin & cos曲线
if __name__ == '__main__':
    x = np.arange(0, 6, 0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)
    plt.plot(x, y1, label="sin")
    plt.plot(x, y2, label="cos", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('sin & cos')
    plt.legend()  # 打上标签
    plt.show()