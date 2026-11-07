from pathlib import Path
import idx2numpy
import numpy as np
from matplotlib import pyplot as plt
from softmax_regression import SoftmaxRegression

def convert_to_onehot_vector(labels: np.ndarray):
    N = labels.shape[0]
    oh_labels = np.zeros((N, 10), dtype=np.float64)
    oh_labels[np.arange(N), labels] = 1.0

    return oh_labels

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent/"data"

    train_images = idx2numpy.convert_from_file(str(data_dir/"train-images-idx3-ubyte"))
    train_labels = idx2numpy.convert_from_file(str(data_dir/"train-labels-idx1-ubyte"))

    test_images = idx2numpy.convert_from_file(str(data_dir/"t10k-images-idx3-ubyte"))
    test_labels = idx2numpy.convert_from_file(str(data_dir/"t10k-labels-idx1-ubyte"))

    encoded_train_labels = convert_to_onehot_vector(train_labels)

    N, _, _ = train_images.shape
    train_images = train_images.reshape(N, -1).astype(np.float64)

    N, _, _ = test_images.shape
    test_images = test_images.reshape(N, -1).astype(np.float64)

    mean = train_images.mean()
    std = train_images.std()
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std

    model = SoftmaxRegression(epoch=500, lr=0.01)
    model.fit(train_images, encoded_train_labels)

    predicted_probabilities = model.predict(test_images)
    metrics = model.evaluate(test_labels, predicted_probabilities)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    plt.plot(range(1, len(model.losses) + 1), model.losses)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy loss")
    plt.title("Training loss - MNIST 0-9")
    plt.show()
