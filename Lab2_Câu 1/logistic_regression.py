import numpy as np
from tqdm import tqdm

class LogisticsRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr
        self.w = None
        self.b = 0.0
        self.losses = []

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        y = y.reshape(-1, 1)
        y_hat = y_hat.reshape(-1, 1)
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        l = (1 - y) * np.log(1 - y_hat) + y * np.log(y_hat)
        return float(-l.mean())

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape
        y = y.reshape(-1, 1) 

        self.w = np.zeros((d, 1), dtype=np.float64)
        self.b = 0.0
        self.losses = []

        with tqdm(range(self.epoch), desc="Training") as pbar:
            for _ in pbar:
                y_hat = self.predict(X)
                delta_y = y_hat - y  

                l = self.loss_fn(y, y_hat)
                self.losses.append(l)

                gradient = delta_y.T @ X 
                self.w = self.w - (self.lr / N) * gradient.T
                self.b = self.b - self.lr * delta_y.mean()

                pbar.set_postfix({"loss": l})

        return self

    def sigmoid(self, z: np.ndarray):
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0
        out[pos] = 1 / (1 + np.exp(-z[pos]))

        ez = np.exp(z[~pos])
        out[~pos] = ez / (1 + ez)

        return out

    def predict(self, X: np.ndarray):
        return self.sigmoid(X @ self.w + self.b) 

    def evaluate(self, y: np.ndarray, y_hat: np.ndarray, threshold: float = 0.5) -> dict:
        y = y.reshape(-1)
        y_hat = (y_hat.reshape(-1) >= threshold).astype(int)

        tp = np.sum((y == 1) & (y_hat == 1))
        fp = np.sum((y == 0) & (y_hat == 1))
        fn = np.sum((y == 1) & (y_hat == 0))

        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}
