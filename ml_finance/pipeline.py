"""Market microstructure modelling pipeline for synthetic crypto data."""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class Trade:
    timestamp: datetime
    trade_price: float
    side: str
    trade_qty: float
    bid_qty: float
    ask_qty: float


@dataclass
class TimeBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    side_mean: float
    bid_qty: float
    ask_qty: float


@dataclass
class DatasetSplit:
    feature_names: List[str]
    train_features: List[List[float]]
    train_targets: List[int]
    test_features: List[List[float]]
    test_targets: List[int]
    validation_features: List[List[float]]
    validation_targets: List[int]


@dataclass
class StandardScaler:
    means: List[float]
    stds: List[float]

    def transform(self, rows: List[List[float]]) -> List[List[float]]:
        transformed: List[List[float]] = []
        for row in rows:
            transformed.append(
                [
                    0.0 if self.stds[i] == 0 else (row[i] - self.means[i]) / self.stds[i]
                    for i in range(len(row))
                ]
            )
        return transformed


@dataclass
class PCAResult:
    components: List[List[float]]

    def transform(self, rows: List[List[float]]) -> List[List[float]]:
        transformed: List[List[float]] = []
        for row in rows:
            transformed.append([
                sum(value * component[idx] for idx, value in enumerate(row))
                for component in self.components
            ])
        return transformed


@dataclass
class LogisticModel:
    weights: List[float]
    bias: float

    def predict_proba(self, rows: List[List[float]]) -> List[float]:
        probabilities: List[float] = []
        for row in rows:
            z = self.bias
            for weight, value in zip(self.weights, row):
                z += weight * value
            probabilities.append(1.0 / (1.0 + math.exp(-z)))
        return probabilities

    def predict(self, rows: List[List[float]]) -> List[int]:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(rows)]


def load_market_data(path: Path) -> List[Trade]:
    trades: List[Trade] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                timestamp = datetime.fromisoformat(row["timestamp"])
                trade_price = float(row["trade_price"])
                side = row["side"].strip()
                trade_qty = float(row["trade_qty"])
                bid_qty = float(row["bid_qty"])
                ask_qty = float(row["ask_qty"])
            except (ValueError, KeyError, TypeError):
                continue
            trades.append(Trade(timestamp, trade_price, side, trade_qty, bid_qty, ask_qty))
    return trades


def clean_market_data(trades: List[Trade]) -> List[Trade]:
    seen = set()
    cleaned: List[Trade] = []
    for trade in trades:
        key = (trade.timestamp, trade.trade_price, trade.trade_qty, trade.side)
        if key in seen:
            continue
        seen.add(key)
        if trade.trade_price <= 0 or trade.trade_qty <= 0:
            continue
        if trade.side not in {"BUY", "SELL"}:
            continue
        cleaned.append(trade)
    cleaned.sort(key=lambda t: t.timestamp)
    return cleaned


def _floor_to_second(ts: datetime) -> datetime:
    return ts.replace(microsecond=0)


def resample_to_seconds(trades: List[Trade]) -> List[TimeBar]:
    buckets: Dict[datetime, List[Trade]] = defaultdict(list)
    for trade in trades:
        buckets[_floor_to_second(trade.timestamp)].append(trade)

    bars: List[TimeBar] = []
    for ts in sorted(buckets.keys()):
        bucket = buckets[ts]
        prices = [t.trade_price for t in bucket]
        volume = sum(t.trade_qty for t in bucket)
        side_mean = sum(1 if t.side == "BUY" else -1 for t in bucket) / len(bucket)
        bid_qty = bucket[-1].bid_qty
        ask_qty = bucket[-1].ask_qty
        bars.append(
            TimeBar(
                timestamp=ts,
                open=prices[0],
                high=max(prices),
                low=min(prices),
                close=prices[-1],
                volume=volume,
                side_mean=side_mean,
                bid_qty=bid_qty,
                ask_qty=ask_qty,
            )
        )
    return bars


def _simple_moving_average(values: Sequence[float], window: int) -> List[float | None]:
    result: List[float | None] = []
    total = 0.0
    q: deque[float] = deque()
    for value in values:
        q.append(value)
        total += value
        if len(q) > window:
            total -= q.popleft()
        if len(q) == window:
            result.append(total / window)
        else:
            result.append(None)
    return result


def _rolling_std(values: Sequence[float], window: int) -> List[float | None]:
    result: List[float | None] = []
    q: deque[float] = deque()
    for value in values:
        q.append(value)
        if len(q) > window:
            q.popleft()
        if len(q) == window:
            mean = sum(q) / window
            variance = sum((x - mean) ** 2 for x in q) / window
            result.append(math.sqrt(variance))
        else:
            result.append(None)
    return result


def _compute_rsi(values: Sequence[float], window: int) -> List[float | None]:
    rsi: List[float | None] = [None] * len(values)
    gains: List[float] = []
    losses: List[float] = []
    avg_gain = avg_loss = None
    for idx in range(1, len(values)):
        delta = values[idx] - values[idx - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        gains.append(gain)
        losses.append(loss)
        if idx < window:
            continue
        if idx == window:
            avg_gain = sum(gains[-window:]) / window
            avg_loss = sum(losses[-window:]) / window
        else:
            assert avg_gain is not None and avg_loss is not None
            avg_gain = ((avg_gain * (window - 1)) + gain) / window
            avg_loss = ((avg_loss * (window - 1)) + loss) / window
        if avg_loss == 0:
            rsi[idx] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[idx] = 100 - (100 / (1 + rs))
    return rsi


def engineer_alphas(bars: List[TimeBar]) -> Tuple[List[str], List[Tuple[datetime, List[float]]]]:
    closes = [bar.close for bar in bars]
    returns: List[float] = [0.0]
    log_returns: List[float] = [0.0]
    for idx in range(1, len(closes)):
        prev = closes[idx - 1]
        curr = closes[idx]
        returns.append((curr - prev) / prev if prev != 0 else 0.0)
        log_returns.append(math.log(curr / prev) if prev != 0 else 0.0)

    momentum = _simple_moving_average(closes, 20)
    mean_short = _simple_moving_average(closes, 5)
    mean_long = _simple_moving_average(closes, 30)
    volatility = _rolling_std(returns, 60)
    order_imbalance = []
    for bar in bars:
        denom = bar.bid_qty + bar.ask_qty
        order_imbalance.append((bar.bid_qty - bar.ask_qty) / denom if denom != 0 else 0.0)
    order_imbalance_ma = _simple_moving_average(order_imbalance, 1)
    trade_flow = _simple_moving_average([bar.side_mean for bar in bars], 15)
    rsi = _compute_rsi(closes, 14)

    feature_names = [
        "alpha_momentum",
        "alpha_mean_reversion",
        "alpha_volatility",
        "alpha_order_imbalance",
        "alpha_trade_flow",
        "alpha_rsi",
    ]

    rows: List[Tuple[datetime, List[float]]] = []
    for idx, bar in enumerate(bars):
        if (
            momentum[idx] is None
            or mean_short[idx] is None
            or mean_long[idx] is None
            or volatility[idx] is None
            or trade_flow[idx] is None
            or rsi[idx] is None
        ):
            continue
        feature_vector = [
            bar.close / momentum[idx] - 1.0,
            (mean_short[idx] - mean_long[idx]),
            volatility[idx],
            order_imbalance_ma[idx] if order_imbalance_ma[idx] is not None else 0.0,
            trade_flow[idx],
            rsi[idx],
        ]
        rows.append((bar.timestamp, feature_vector))
    return feature_names, rows


def create_target(bars: List[TimeBar], horizon_seconds: int, threshold: float = 0.0005) -> Dict[datetime, int]:
    target: Dict[datetime, int] = {}
    for idx in range(len(bars) - horizon_seconds):
        future_price = bars[idx + horizon_seconds].close
        current_price = bars[idx].close
        if current_price == 0:
            continue
        future_return = (future_price - current_price) / current_price
        label = 1 if future_return > threshold else 0
        target[bars[idx].timestamp] = label
    return target


def _align(features: List[Tuple[datetime, List[float]]], targets: Dict[datetime, int]) -> Tuple[List[List[float]], List[int]]:
    X: List[List[float]] = []
    y: List[int] = []
    for timestamp, vector in features:
        if timestamp in targets:
            X.append(vector)
            y.append(targets[timestamp])
    return X, y


def _standardize(X: List[List[float]]) -> Tuple[StandardScaler, List[List[float]]]:
    n_features = len(X[0])
    means = [0.0] * n_features
    stds = [0.0] * n_features
    for j in range(n_features):
        column = [row[j] for row in X]
        mean = sum(column) / len(column)
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        means[j] = mean
        stds[j] = math.sqrt(variance)
    scaler = StandardScaler(means, stds)
    return scaler, scaler.transform(X)


def _matvec(matrix: List[List[float]], vector: List[float]) -> List[float]:
    return [sum(row[i] * vector[i] for i in range(len(vector))) for row in matrix]


def _normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [v / norm for v in vector]


def _power_iteration(matrix: List[List[float]], iterations: int = 100) -> Tuple[float, List[float]]:
    n = len(matrix)
    vector = [random.random() for _ in range(n)]
    vector = _normalize(vector)
    for _ in range(iterations):
        vector = _matvec(matrix, vector)
        vector = _normalize(vector)
    mv = _matvec(matrix, vector)
    eigenvalue = sum(vector[i] * mv[i] for i in range(n))
    return eigenvalue, vector


def _covariance_matrix(X: List[List[float]]) -> List[List[float]]:
    n_samples = len(X)
    n_features = len(X[0])
    matrix = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    for row in X:
        for i in range(n_features):
            for j in range(i, n_features):
                matrix[i][j] += row[i] * row[j]
    for i in range(n_features):
        for j in range(i, n_features):
            value = matrix[i][j] / (n_samples - 1 if n_samples > 1 else 1)
            matrix[i][j] = value
            matrix[j][i] = value
    return matrix


def _pca(X: List[List[float]], n_components: int) -> Tuple[PCAResult, List[List[float]]]:
    covariance = _covariance_matrix(X)
    components: List[List[float]] = []
    matrix = [row[:] for row in covariance]
    for _ in range(n_components):
        eigenvalue, eigenvector = _power_iteration(matrix)
        components.append(eigenvector)
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j]
    pca = PCAResult(components)
    return pca, pca.transform(X)


def _train_test_validation_split(X: List[List[float]], y: List[int]) -> DatasetSplit:
    total = len(X)
    if total < 10:
        raise ValueError("Not enough samples to split the dataset")
    train_end = int(total * 0.6)
    test_end = int(total * 0.8)

    return DatasetSplit(
        feature_names=[],
        train_features=X[:train_end],
        train_targets=y[:train_end],
        test_features=X[train_end:test_end],
        test_targets=y[train_end:test_end],
        validation_features=X[test_end:],
        validation_targets=y[test_end:],
    )


def _logistic_regression_train(X: List[List[float]], y: List[int], learning_rate: float = 0.1, epochs: int = 200) -> LogisticModel:
    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0
    for _ in range(epochs):
        grad_w = [0.0] * n_features
        grad_b = 0.0
        for features, target in zip(X, y):
            z = bias
            for weight, value in zip(weights, features):
                z += weight * value
            prediction = 1.0 / (1.0 + math.exp(-z))
            error = prediction - target
            for idx in range(n_features):
                grad_w[idx] += error * features[idx]
            grad_b += error
        for idx in range(n_features):
            weights[idx] -= learning_rate * grad_w[idx] / len(X)
        bias -= learning_rate * grad_b / len(X)
    return LogisticModel(weights, bias)


def _classification_metrics(model: LogisticModel, X: List[List[float]], y: List[int]) -> Dict[str, object]:
    if not X:
        return {}
    predictions = model.predict(X)
    accuracy = sum(1 for pred, target in zip(predictions, y) if pred == target) / len(y)
    confusion = [[0, 0], [0, 0]]
    for pred, target in zip(predictions, y):
        confusion[target][pred] += 1

    def precision_recall_f1(label: int) -> Dict[str, float]:
        tp = confusion[label][label]
        fp = confusion[1 - label][label]
        fn = confusion[label][1 - label]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    report = {str(label): precision_recall_f1(label) for label in (0, 1)}
    return {"accuracy": accuracy, "confusion_matrix": confusion, "report": report}


def run_pipeline(data_path: Path, horizon_seconds: int = 60) -> Dict[str, object]:
    trades = load_market_data(data_path)
    cleaned = clean_market_data(trades)
    bars = resample_to_seconds(cleaned)
    feature_names, feature_rows = engineer_alphas(bars)
    targets = create_target(bars, horizon_seconds=horizon_seconds)
    X_raw, y = _align(feature_rows, targets)
    if not X_raw:
        raise ValueError("No features available after alignment")
    scaler, X_scaled = _standardize(X_raw)
    n_components = min(5, len(feature_names))
    pca, X_pca = _pca(X_scaled, n_components)
    split = _train_test_validation_split(X_pca, y)
    split.feature_names = feature_names
    model = _logistic_regression_train(split.train_features, split.train_targets)
    metrics = {
        "train": _classification_metrics(model, split.train_features, split.train_targets),
        "test": _classification_metrics(model, split.test_features, split.test_targets),
        "validation": _classification_metrics(model, split.validation_features, split.validation_targets),
    }
    return {
        "scaler": scaler,
        "pca": pca,
        "model": model,
        "metrics": metrics,
        "feature_names": feature_names,
    }


def main(argv: Iterable[str] | None = None) -> Dict[str, object]:
    parser = argparse.ArgumentParser(description="Run the market prediction pipeline.")
    parser.add_argument("--data", type=Path, default=Path("data/mock_market_data.csv"), help="Path to input CSV")
    parser.add_argument("--horizon", type=int, default=60, help="Prediction horizon in seconds")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_pipeline(args.data, horizon_seconds=args.horizon)


if __name__ == "__main__":
    results = main()
    for split_name, metrics in results["metrics"].items():
        if not metrics:
            continue
        print(f"=== {split_name.upper()} ===")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])
