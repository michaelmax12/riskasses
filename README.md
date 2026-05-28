# Risk Management Dashboard

Streamlit app for portfolio risk analysis: Monte Carlo simulation, efficient frontier, Black–Litterman, hierarchical risk parity (HRP), custom PPP optimization, and Indonesian market data via Yahoo Finance.

## Requirements

- Python 3.10 or newer (3.11+ recommended)
- See [requirements.txt](requirements.txt) for Python packages

## Installation

```bash
git clone https://github.com/noprague/Risk_Management.git
cd Risk_Management

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Run the app

From the project root:

```bash
streamlit run main.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Project structure

```
Risk_Management/
├── main.py              # Streamlit UI
├── requirements.txt
├── src/
│   ├── Helper.py        # Data download, EF, BL, HRP, Monte Carlo
│   └── Custom.py        # PPP / custom optimization (PyMC)
└── README.md
```

## Usage notes

- Enter tickers in the sidebar (comma or newline separated). Include `^JKSE` for Jakarta Composite Index when using Black–Litterman or stock analysis vs. the market.
- Historical period examples: `1y`, `2y`, `5y`, `max`, etc. (yfinance periods).
- PPP and fundamental-style features rely on yfinance; for production Bloomberg-style data, swap in your own data source in `src/Custom.py`.

## License

This project is licensed under the [MIT License](LICENSE) — Copyright (c) 2026 Wira.
