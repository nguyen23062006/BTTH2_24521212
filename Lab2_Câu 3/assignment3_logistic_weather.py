import csv
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

def load_weather(path):
    feature_columns = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "sunshine_duration", "wind_speed_10m_max"]
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    X = np.array([[float(row[name]) for name in feature_columns] for row in rows])
    y = np.array([int(float(row["precipitation_sum"]) > 0) for row in rows])
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


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent/"data"/"weather_daily.csv"
    X, y, dates, cities = load_weather(data_path)

    train_mask = (dates >= "2021-01-01") & (dates < "2025-01-01")
    test_mask = (dates >= "2025-01-01") & (dates < "2026-01-01")
    X = prepare_features(X, dates, cities, train_mask)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)
    predicted_labels = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted_labels, average="binary", zero_division=0)
    
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"f1: {f1:.4f}")
