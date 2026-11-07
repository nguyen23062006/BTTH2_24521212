import csv
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from softmax_regression import SoftmaxRegression

def load_weather(path):
    feature_columns = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "sunshine_duration", "wind_speed_10m_max"]

    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    X = np.array([[float(row[name]) for name in feature_columns] for row in rows])
    precipitation = np.array([float(row["precipitation_sum"]) for row in rows])

    y = np.zeros(len(rows), dtype=int)
    y[precipitation > 0] = 1
    y[precipitation >= 10] = 2

    dates = np.array([row["time"] for row in rows])
    cities = np.array([row["city"] for row in rows])

    return X, y, dates, cities

def prepare_features(X, dates, cities, train_mask):
    mean = X[train_mask].mean(axis=0)
    std = X[train_mask].std(axis=0)
    std[std == 0] = 1.0
    numeric = (X - mean) / std

    months = np.array([int(date[5:7]) for date in dates])
    angles = 2 * np.pi * (months - 1) / 12
    month_features = np.column_stack([np.sin(angles), np.cos(angles)])

    city_names = np.unique(cities[train_mask])
    city_features = (cities[:, None] == city_names[None, 1:]).astype(float)
    city_month_sin = city_features * month_features[:, 0:1]
    city_month_cos = city_features * month_features[:, 1:2]

    return np.column_stack([numeric, month_features, city_features, city_month_sin, city_month_cos])

def convert_to_onehot_vector(labels: np.ndarray):
    N = labels.shape[0]
    oh_labels = np.zeros((N, 3), dtype=np.float64)
    oh_labels[np.arange(N), labels] = 1.0

    return oh_labels

if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent/"data"/"weather_daily.csv"
    X, y, dates, cities = load_weather(data_path)

    train_mask = (dates >= "2021-01-01") & (dates < "2024-01-01")
    val_mask = (dates >= "2024-01-01") & (dates < "2025-01-01")
    test_mask = (dates >= "2025-01-01") & (dates < "2026-01-01")

    X = prepare_features(X, dates, cities, train_mask)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    encoded_train_labels = convert_to_onehot_vector(y_train)

    epoch_candidates = [100, 500, 1000]
    best_f1 = -1.0

    for epoch in epoch_candidates:
        model = SoftmaxRegression(epoch=epoch, lr=0.1)
        model.fit(X_train, encoded_train_labels)

        val_probabilities = model.predict(X_val)
        val_metrics = model.evaluate(y_val, val_probabilities)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_model = model

    test_probabilities = best_model.predict(X_test)
    test_metrics = best_model.evaluate(y_test, test_probabilities) 

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    plt.plot(range(1, len(best_model.losses) + 1), best_model.losses)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy loss")
    plt.title("Training loss - Weather (3 classes)")
    plt.show()
