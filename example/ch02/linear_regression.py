import torch


class SimpleLinear:
    def __init__(self, regula=0.0):
        self.regula = regula
        self.W = None

    def fit(self, X, y, lr=0.01, num_iter=1000):
        """
        y = W * X + b
          = [W,b] * [x,1]T
        """

        if len(X) == 0 or len(y) != len(X):
            raise ValueError("X and y must have same length and cannot be empty. ")

        X = torch.hstack([torch.ones(len(X), 1), X])
        self.W = torch.randn(X.shape[1])

        for i in range(num_iter):
            yp = X @ self.W

            loss = torch.sum((yp - y) ** 2) + torch.sum(self.W ** 2) * self.regula

            gradients = 2 * ((torch.transpose(X, 0, 1) @ (yp - y)) + self.W * self.regula)

            self.W = self.W - gradients * lr

            if ((i+1)%1000) == 0: print(loss)

    def predict(self, X):
        X = torch.hstack([torch.ones(len(X), 1), X])
        p = X @ self.W
        return p


if __name__ == '__main__':
    y = torch.tensor([2, 4, 5, 4, 5])
    X = torch.t(torch.tensor([[1, 2, 3, 4, 5]]))
    lr = SimpleLinear(regula=0.1)
    lr.fit(X, y, lr=0.01, num_iter=10000)
    print(lr.W)
    y_predict = lr.predict(X)
    print(y_predict)
