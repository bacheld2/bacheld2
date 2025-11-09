# Synthetic Crypto Market ML Pipeline

This project demonstrates an end-to-end machine learning workflow for high-frequency, crypto-like market data using only the Python standard library. It includes:

- A data generator that produces synthetic trades with millisecond timestamps, order book depth, and injected data quality issues.
- A cleaning and feature engineering pipeline that constructs a basket of technical analysis inspired alphas.
- A dimensionality reduction and classification model that predicts the direction of the next one-minute price move using handcrafted PCA and logistic regression implementations.

## Getting Started

1. **Generate synthetic market data**

   ```bash
   python -m ml_finance.data_generation --rows 20000 --output data/mock_market_data.csv
   ```

2. **Run the modelling pipeline**

   ```bash
   python -m ml_finance.pipeline --data data/mock_market_data.csv --horizon 60
   ```

The pipeline splits the engineered alpha factors into training, testing, and validation sets (60%/20%/20%), performs PCA to combine the most predictive alphas, and fits a logistic regression model. The resulting accuracy and confusion matrices for each split are printed to the console.

## Project Structure

```
ml_finance/
├── __init__.py
├── data_generation.py   # Synthetic data generator
└── pipeline.py          # Cleaning, feature engineering, PCA, and modelling
```

Generated data files are stored under `data/`.
