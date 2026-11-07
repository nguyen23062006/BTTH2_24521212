from pathlib import Path
import idx2numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent/"data"
    train_images = idx2numpy.convert_from_file(str(data_dir/"train-images-idx3-ubyte"))
    train_labels = idx2numpy.convert_from_file(str(data_dir/"train-labels-idx1-ubyte"))
    test_images = idx2numpy.convert_from_file(str(data_dir/"t10k-images-idx3-ubyte"))
    test_labels = idx2numpy.convert_from_file(str(data_dir/"t10k-labels-idx1-ubyte"))

    X_train = train_images.reshape(len(train_images), -1) / 255.0
    X_test = test_images.reshape(len(test_images), -1) / 255.0

    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X_train, train_labels)
    predicted_labels = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predicted_labels, average="macro", zero_division=0)
    
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"f1: {f1:.4f}")
