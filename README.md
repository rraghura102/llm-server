# Ticker New Prediction

Ticker News AI is a Proof of Concept (POC) of a multi-input neural network built with Keras 3 and powered by the JAX backend, designed to model stock movement by combining structured OHLC (Open, High, Low, Close, Volume) price data with unstructured news headlines. It demonstrates how financial time-series and natural language inputs can be fused for predictive modeling in a clean, functional Keras architecture.

⚠️ NOTE
This code is not production-ready and is intended solely for proof-of-concept (PoC) and demonstration purposes. It lacks production-grade features such as authentication, request limits, error handling, hardening, and full model lifecycle management.

# Features

- Dual-input model: numeric OHLC + tokenized news headlines
- Built using the Keras Functional API with JAX acceleration
- Supports training, saving, and zero-shot inference
- Clean preprocessing pipeline with `scikit-learn` and `Tokenizer`
- Easily extendable to real financial targets and news sentiment

# Inputs 

- `train_ohlc.csv`: OHLC data with `ticker,date,open,low,high,close,volume`
- `train_news.csv`: News with `id,ticker,exchange,headline,source`
- `test_ohlc.csv` and `test_news.csv`: Optional zero-shot evaluation inputs

# Quick Install

```
pip install -r requirements.txt
python3 ticker_news_ai.py
```
### License

© 2025 Rayan Raghuram. All rights reserved.
This project is licensed under the MIT License.
