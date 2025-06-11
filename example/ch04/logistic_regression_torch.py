import torch


class SimpleLogisticRegression(object):

    def __init__(self, lr=0.01, n_iter=10000, regula=0.0):
        self.regula = regula
        self.n_iter = n_iter
        self.lr = lr
        self.W = None
        self.bias = 0.0

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        n_samples, n_dim = X.shape
        if len(X) == 0 or len(y) != len(X):
            raise ValueError("X and y must have same length and cannot be empty. ")

        self.W = torch.randn(n_dim)

        for i in range(self.n_iter):

            z = X @ self.W + self.bias
            yp = _sigmoid(z)

            # loss = torch.sum((z - y) ** 2) + (torch.sum(self.W ** 2) + self.bias) * self.regula
            # dw = (torch.transpose(X, 0, 1) @ (z - y) + self.regula * self.W) * 2
            # db = 2 * torch.sum(z - y) + self.regula
            loss = (-1/n_samples) * (torch.log(yp) * y + (1-y) * torch.log(1-yp))
            dw = (1/n_samples) * (torch.transpose(X, 0, 1) @ (yp - y))
            db = (1/n_samples) * torch.sum(yp-y)

            if ((i + 1) % 1000) == 0: print(loss)

            self.W -= dw * self.lr
            self.bias -= db * self.lr

    def predict(self, X):
        return X @ self.W + self.bias


def _sigmoid(z):
    return 1 / 1 + torch.exp(-z)


if __name__ == '__main__':
    # create sample dataset
    X = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=torch.float)
    y = torch.tensor([0, 0, 1, 1, 1], dtype=torch.float)

    # initialize logistic regression model
    lr = SimpleLogisticRegression()

    # train model on sample dataset
    lr.fit(X, y)

    # make predictions on new data
    X_new = torch.tensor([[6, 7], [7, 8]], dtype=torch.float)
    y_p = lr.predict(X_new)

    print(y_p)
