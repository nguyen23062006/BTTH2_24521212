from pathlib import Path
import idx2numpy
import numpy as np
from matplotlib import pyplot as plt
from logistic_regression import LogisticsRegression

def filter_data(data, condition):
    images, labels = data

    new_images = images[labels == condition]
    new_labels = labels[labels == condition]

    return new_images, new_labels

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent/"data"

    train_images = idx2numpy.convert_from_file(str(data_dir/"train-images-idx3-ubyte"))
    train_labels = idx2numpy.convert_from_file(str(data_dir/"train-labels-idx1-ubyte"))
    train_data = (train_images, train_labels)

    test_images = idx2numpy.convert_from_file(str(data_dir/"t10k-images-idx3-ubyte"))
    test_labels = idx2numpy.convert_from_file(str(data_dir/"t10k-labels-idx1-ubyte"))
    test_data = (test_images, test_labels)

    train_images_0, train_labels_0 = filter_data(train_data, 0)
    train_images_1, train_labels_1 = filter_data(train_data, 1)
    train_images = np.concatenate([train_images_0, train_images_1], axis=0)
    train_labels = np.concatenate([train_labels_0, train_labels_1], axis=0)

    test_images_0, test_labels_0 = filter_data(test_data, 0)
    test_images_1, test_labels_1 = filter_data(test_data, 1)
    test_images = np.concatenate([test_images_0, test_images_1], axis=0)
    test_labels = np.concatenate([test_labels_0, test_labels_1], axis=0)

    N, _, _ = train_images.shape
    train_images = train_images.reshape(N, -1) / 255.0

    N, _, _ = test_images.shape
    test_images = test_images.reshape(N, -1) / 255.0

    model = LogisticsRegression(epoch=100, lr=0.1)
    model.fit(train_images, train_labels)

    predicted_probabilities = model.predict(test_images)
    metrics = model.evaluate(test_labels, predicted_probabilities)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    plt.plot(range(1, len(model.losses) + 1), model.losses)
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy loss")
    plt.title("Training loss - MNIST 0 vs 1")
    plt.show()
