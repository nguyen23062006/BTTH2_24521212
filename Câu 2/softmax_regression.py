import numpy as np
from tqdm import tqdm

class SoftmaxRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr
        self.w = None
        self.b = None
        self.losses = []

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        y_hat = np.clip(y_hat, 1e-15, 1.0)
        return float(-(y * np.log(y_hat)).sum(axis=1).mean())

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape
        _, k = y.shape

        self.w = np.zeros((d, k), dtype=np.float64)
        self.b = np.zeros(k, dtype=np.float64)
        self.losses = []

        with tqdm(range(self.epoch), desc="Training") as pbar:
            for _ in pbar:
                y_hat = self.predict(X)

                l = self.loss_fn(y, y_hat)
                self.losses.append(l)

                delta_y = y_hat - y  
                gradient = delta_y.T @ X  
                self.w = self.w - (self.lr / N) * gradient.T
                self.b = self.b - self.lr * delta_y.mean(axis=0)

                pbar.set_postfix({"loss": l})

        return self

    def softmax(self, z: np.ndarray):
        z = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)

        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray):
        return self.softmax(X @ self.w + self.b) 

    def evaluate(self, y: np.ndarray, y_hat: np.ndarray) -> dict:
        y = y.reshape(-1)
        k = y_hat.shape[1]
        predicted_labels = np.argmax(y_hat, axis=1)
        precisions, recalls, f1_scores = [], [], []

        for label in range(k):
            tp = np.sum((y == label) & (predicted_labels == label))
            fp = np.sum((y != label) & (predicted_labels == label))
            fn = np.sum((y == label) & (predicted_labels != label))

            precision = float(tp / (tp + fp)) if tp + fp else 0.0
            recall = float(tp / (tp + fn)) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        return {"precision": float(np.mean(precisions)), "recall": float(np.mean(recalls)), "f1": float(np.mean(f1_scores))}
