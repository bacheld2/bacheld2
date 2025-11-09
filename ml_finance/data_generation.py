"""Synthetic market data generator for high-volatility crypto-like assets."""
from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List

SECONDS_PER_YEAR = 365 * 24 * 60 * 60


@dataclass
class MarketConfig:
    """Configuration used to generate synthetic trade and order book data."""

    start_price: float = 20_000.0
    n_rows: int = 20_000
    seed: int = 13
    target_annual_vol: float = 0.80

    def __post_init__(self) -> None:
        if self.start_price <= 0:
            raise ValueError("start_price must be positive")
        if self.n_rows <= 1:
            raise ValueError("n_rows must be greater than 1")


def _student_t(df: int, rng: random.Random) -> float:
    z = rng.gauss(0.0, 1.0)
    g = rng.gammavariate(df / 2.0, 2.0)
    return z / math.sqrt(g / df)


def _sample_returns(config: MarketConfig, rng: random.Random) -> List[float]:
    dt = 1.0 / SECONDS_PER_YEAR
    scale = config.target_annual_vol * math.sqrt(dt)
    returns: List[float] = []
    for _ in range(config.n_rows - 1):
        r = _student_t(df=3, rng=rng) * scale
        vol_shock = rng.gammavariate(2.5, 0.6)
        returns.append(r * (0.5 + 0.5 * vol_shock))
    return returns


def _generate_order_book_noise(config: MarketConfig, rng: random.Random) -> List[tuple[float, float]]:
    depth: List[tuple[float, float]] = []
    for _ in range(config.n_rows):
        base_qty = rng.lognormvariate(3.0, 0.7)
        imbalance = rng.gauss(0.0, 0.35)
        bid_qty = max(base_qty * (1 + imbalance), 1.0)
        ask_qty = max(base_qty * (1 - imbalance), 1.0)
        depth.append((bid_qty, ask_qty))
    return depth


def build_market_data(config: MarketConfig) -> List[dict[str, object]]:
    rng = random.Random(config.seed)
    log_returns = _sample_returns(config, rng)
    prices: List[float] = [config.start_price]
    for r in log_returns:
        prices.append(prices[-1] * math.exp(r))

    base_time = datetime.now(tz=timezone.utc).replace(microsecond=0)
    timestamps = [base_time]
    for _ in range(config.n_rows - 1):
        delta_ms = rng.randint(1, 50)
        timestamps.append(timestamps[-1] + timedelta(milliseconds=delta_ms))

    sides = ["BUY" if rng.random() < 0.52 else "SELL" for _ in range(config.n_rows)]
    qty = [rng.lognormvariate(2.0, 0.8) for _ in range(config.n_rows)]
    depth = _generate_order_book_noise(config, rng)

    rows: List[dict[str, object]] = []
    for idx in range(config.n_rows):
        bid_qty, ask_qty = depth[idx]
        rows.append(
            {
                "timestamp": timestamps[idx].isoformat(),
                "trade_price": round(prices[idx], 8),
                "side": sides[idx],
                "trade_qty": round(qty[idx], 6),
                "bid_qty": round(bid_qty, 6),
                "ask_qty": round(ask_qty, 6),
            }
        )

    error_indices = rng.sample(range(config.n_rows), k=max(3, config.n_rows // 500))
    rows[error_indices[0]]["trade_price"] = -abs(float(rows[error_indices[0]]["trade_price"]))
    rows[error_indices[1]]["trade_qty"] = ""
    rows[error_indices[2]]["side"] = ""
    return rows


def write_market_csv(data: List[dict[str, object]], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "trade_price", "side", "trade_qty", "bid_qty", "ask_qty"]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    return output


def main(argv: Iterable[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Generate synthetic crypto-like market data.")
    parser.add_argument("--rows", type=int, default=20_000, help="Number of rows to generate.")
    parser.add_argument("--output", type=Path, default=Path("data/mock_market_data.csv"), help="Path to the CSV file.")
    parser.add_argument("--seed", type=int, default=13, help="RNG seed for reproducibility.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = MarketConfig(n_rows=args.rows, seed=args.seed)
    data = build_market_data(config)
    write_market_csv(data, args.output)
    return args.output


if __name__ == "__main__":
    main()
