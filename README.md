# GHOST ASBE

Streamlit research dashboard for the DPAM event study.

## Deploy with Streamlit Community Cloud
1. Push this repository to GitHub.
2. In Streamlit Community Cloud, select **Create app**.
3. Repository: `swasthik-s/GHOST`
4. Branch: `main`
5. Main file path: `app.py`
6. Deploy. Future commits to the selected branch trigger redeployment.

## Run locally
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Data
Upload an MT5 OHLC CSV containing Open, High, Low and Close. Timestamp is recommended. Separate Date and Time columns are supported.

## Research limitations
Research only. No MT5 connection or order execution. Outputs are raw price-unit returns, not account-currency P/L or R. Spread, commission, slippage, swaps, execution latency and FTMO risk constraints are not modelled.
