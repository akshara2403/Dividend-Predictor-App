import streamlit as st
import yfinance as yf
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

st.title("📈 Predict Dividend Change using yFinance Data")

ticker_input = st.text_input("Enter a Ticker Symbol (e.g., AAPL, MSFT, GE):", value="AAPL")
industry = st.selectbox("Select Industry", ["Consumer", "Financials", "Energy", "Other"])
model_option = st.selectbox("Choose a Model:", ["CatBoost", "XGBoost", "LightGBM"])

def safe_get(df, key):
    df = df.T
    return df[key].iloc[0] if key in df.columns else 0

def safe_div(a, b):
    try:
        return float(a) / float(b) if a and b and b != 0 else 0
    except:
        return 0

if st.button("🔍 Fetch & Predict"):
    try:
        stock = yf.Ticker(ticker_input)

        # Use older-style quarterly attributes
        income_raw = stock.get_income_stmt(freq="quarterly")
        balance_raw = stock.get_balance_sheet(freq="quarterly")
        cashflow_raw = stock.get_cash_flow(freq="quarterly")

        
        # Show raw results for debugging
        st.write("🧾 Income Statement Shape:", income_raw.shape)
        st.write("🧾 Balance Sheet Shape:", balance_raw.shape)
        st.write("🧾 Cash Flow Shape:", cashflow_raw.shape)

        if income_raw.empty or balance_raw.empty or cashflow_raw.empty:
            st.error("❌ One or more financial statements are unavailable. Try a different ticker.")
            st.stop()

        short_debt = safe_get(balance_raw, "Short Long Term Debt")
        long_debt = safe_get(balance_raw, "Long Term Debt")
        total_debt = short_debt + long_debt

        ratios = {
            'dpr': safe_div(safe_get(cashflow_raw, "Cash Dividends Paid"), safe_get(income_raw, "Net Income")),
            'roe': safe_div(safe_get(income_raw, "Net Income"), safe_get(balance_raw, "Stockholders Equity")),
            'roa': safe_div(safe_get(income_raw, "Net Income"), safe_get(balance_raw, "Total Assets")),
            'GProf': safe_div(safe_get(income_raw, "Gross Profit"), safe_get(income_raw, "Total Revenue")),
            'npm': safe_div(safe_get(income_raw, "Net Income"), safe_get(income_raw, "Total Revenue")),
            'fcf_ocf': safe_div(safe_get(cashflow_raw, "Free Cash Flow"), safe_get(cashflow_raw, "Operating Cash Flow")),
            'cash_debt': safe_div(safe_get(balance_raw, "Cash And Cash Equivalents"), total_debt),
            'cash_lt': safe_div(safe_get(balance_raw, "Cash And Cash Equivalents"), long_debt),
            'ocf_lct': safe_div(safe_get(cashflow_raw, "Operating Cash Flow"), total_debt),
            'totdebt_invcap': safe_div(total_debt, safe_get(balance_raw, "Invested Capital")),
            'de_ratio': safe_div(total_debt, safe_get(balance_raw, "Stockholders Equity")),
            'debt_ebitda': safe_div(total_debt, safe_get(income_raw, "EBITDA")),
            'intcov_ratio': safe_div(safe_get(income_raw, "EBIT"), safe_get(income_raw, "Interest Expense")),
            'curr_ratio': safe_div(safe_get(balance_raw, "Current Assets"), safe_get(balance_raw, "Current Liabilities")),
            'cash_ratio': safe_div(safe_get(balance_raw, "Cash And Cash Equivalents"), safe_get(balance_raw, "Current Liabilities")),
            'quick_ratio': safe_div(safe_get(balance_raw, "Current Assets") - safe_get(balance_raw, "Inventory"), safe_get(balance_raw, "Current Liabilities")),
        }

        input_df = pd.DataFrame([ratios])
        input_df.fillna(0, inplace=True)
        if input_df.isnull().values.any():
            st.warning("⚠️ Warning: NaN detected in input. Filling with 0.")

        st.subheader("📋 Computed Financial Ratios")
        st.dataframe(input_df.T)

        sector_key = industry.lower()
        model_paths = {
            "CatBoost": f"models/catboost_model_{sector_key}.pkl",
            "XGBoost": f"models/xgboost_model_{sector_key}.pkl",
            "LightGBM": f"models/lightgbm_model_{sector_key}.pkl",
        }

        model_path = model_paths[model_option]
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        y_pred = model.predict(input_df)
        y_proba = model.predict_proba(input_df)[0]

        label_map = {-1: "📉 Decrease", 0: "➖ No Change", 1: "📈 Increase"}
        pred = int(y_pred.flatten()[0]) if hasattr(y_pred, 'flatten') else int(y_pred[0])
        st.success(f"📊 Predicted Dividend Change: *{label_map[pred]}*")

        proba_map = {-1: y_proba[-1], 0: y_proba[0], 1: y_proba[1]}
        proba_df = pd.DataFrame.from_dict({label_map[k]: [v] for k, v in proba_map.items()}, orient='columns')
        st.subheader("🔢 Prediction Probabilities")
        st.bar_chart(proba_df.T.rename(columns={0: "Probability"}))

        st.session_state.prediction_label = label_map[pred]
        st.session_state.input_df = input_df
        st.session_state.ticker = ticker_input
        st.session_state.industry = industry

        # Simple GPT-style explanation
        st.markdown("---")
        st.title("GPT-Style Analysis of Dividend Prediction")
        st.markdown(f"### Ticker: `{ticker_input}` | Industry: `{industry}`")
        st.markdown(f"### 📊 Predicted Dividend Change: **{label_map[pred]}**")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")


import streamlit as st
import yfinance as yf
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

st.title("📈 Predict Dividend Change using yFinance Data")

# --- User Inputs ---
ticker_input = st.text_input("Enter a Ticker Symbol (e.g., AAPL, MSFT, GE):", value="AAPL")
industry = st.selectbox("Select Industry", ["Consumer", "Financials", "Energy", "Other"])
model_option = st.selectbox("Choose a Model:", ["CatBoost", "XGBoost", "LightGBM"])

# --- Safe Functions ---
def safe_div(a, b):
    try:
        return float(a) / float(b) if a and b and b != 0 else 0
    except:
        return 0

def safe_get(series, key):
    return series.get(key, 0) or 0

if st.button("🔍 Fetch & Predict"):
    try:
        stock = yf.Ticker(ticker_input)

        income_raw = stock.quarterly_income_stmt
        balance_raw = stock.quarterly_balance_sheet
        cashflow_raw = stock.quarterly_cashflow

        st.write("🧾 Income Statement Shape:", income_raw.shape)
        st.write("🧾 Balance Sheet Shape:", balance_raw.shape)
        st.write("🧾 Cash Flow Shape:", cashflow_raw.shape)

        if income_raw.empty or balance_raw.empty or cashflow_raw.empty:
            st.error("❌ One or more financial statements are unavailable. Try a different ticker.")
            st.stop()

        # Take most recent quarter
        income_stmt = income_raw.iloc[:, 0]
        balance_sheet = balance_raw.iloc[:, 0]
        cashflow_stmt = cashflow_raw.iloc[:, 0]

        short_debt = safe_get(balance_sheet, "Short Long Term Debt")
        long_debt = safe_get(balance_sheet, "Long Term Debt")
        total_debt = short_debt + long_debt

        # --- Compute Ratios ---
        ratios = {
            'dpr': safe_div(safe_get(cashflow_stmt, "Cash Dividends Paid"), safe_get(income_stmt, "Net Income")),
            'roe': safe_div(safe_get(income_stmt, "Net Income"), safe_get(balance_sheet, "Stockholders Equity")),
            'roa': safe_div(safe_get(income_stmt, "Net Income"), safe_get(balance_sheet, "Total Assets")),
            'GProf': safe_div(safe_get(income_stmt, "Gross Profit"), safe_get(income_stmt, "Total Revenue")),
            'npm': safe_div(safe_get(income_stmt, "Net Income"), safe_get(income_stmt, "Total Revenue")),
            'fcf_ocf': safe_div(safe_get(cashflow_stmt, "Free Cash Flow"), safe_get(cashflow_stmt, "Operating Cash Flow")),
            'cash_debt': safe_div(safe_get(balance_sheet, "Cash And Cash Equivalents"), total_debt),
            'cash_lt': safe_div(safe_get(balance_sheet, "Cash And Cash Equivalents"), long_debt),
            'ocf_lct': safe_div(safe_get(cashflow_stmt, "Operating Cash Flow"), total_debt),
            'totdebt_invcap': safe_div(total_debt, safe_get(balance_sheet, "Invested Capital")),
            'de_ratio': safe_div(total_debt, safe_get(balance_sheet, "Stockholders Equity")),
            'debt_ebitda': safe_div(total_debt, safe_get(income_stmt, "EBITDA")),
            'intcov_ratio': safe_div(safe_get(income_stmt, "EBIT"), safe_get(income_stmt, "Interest Expense")),
            'curr_ratio': safe_div(safe_get(balance_sheet, "Current Assets"), safe_get(balance_sheet, "Current Liabilities")),
            'cash_ratio': safe_div(safe_get(balance_sheet, "Cash And Cash Equivalents"), safe_get(balance_sheet, "Current Liabilities")),
            'quick_ratio': safe_div(safe_get(balance_sheet, "Current Assets") - safe_get(balance_sheet, "Inventory"), safe_get(balance_sheet, "Current Liabilities")),
        }

        # --- Format Input ---
        input_df = pd.DataFrame([ratios])
        input_df.fillna(0, inplace=True)
        if input_df.isnull().values.any():
            st.warning("⚠️ NaNs detected. Filled with zeros.")

        st.subheader("📋 Computed Financial Ratios")
        st.dataframe(input_df.T)

        # --- Load Model ---
        sector_key = industry.lower()
        model_paths = {
            "CatBoost": f"models/catboost_model_{sector_key}.pkl",
            "XGBoost": f"models/xgboost_model_{sector_key}.pkl",
            "LightGBM": f"models/lightgbm_model_{sector_key}.pkl",
        }
        model_path = model_paths[model_option]
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # --- Predict ---
        y_pred = model.predict(input_df)
        y_proba = model.predict_proba(input_df)[0]
        label_map = {-1: "📉 Decrease", 0: "➖ No Change", 1: "📈 Increase"}
        pred = int(y_pred.flatten()[0]) if hasattr(y_pred, 'flatten') else int(y_pred[0])
        st.success(f"📊 Predicted Dividend Change: **{label_map[pred]}**")

        # --- Probabilities ---
        proba_map = {-1: y_proba[-1], 0: y_proba[0], 1: y_proba[1]}
        proba_df = pd.DataFrame.from_dict({label_map[k]: [v] for k, v in proba_map.items()}, orient="columns")
        st.subheader("🔢 Prediction Probabilities")
        st.bar_chart(proba_df.T.rename(columns={0: "Probability"}))

        # --- Cache for GPT ---
        st.session_state.prediction_label = label_map[pred]
        st.session_state.input_df = input_df
        st.session_state.ticker = ticker_input
        st.session_state.industry = industry

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

