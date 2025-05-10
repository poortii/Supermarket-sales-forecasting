import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from forecast_utils import (
    load_data,
    forecast_with_prophet,
    forecast_with_holtwinters,
    forecast_with_lstm,
    forecast_with_xgboost,
    plot_forecasts,
    compute_metrics   # ✅ add this
)


st.set_page_config(page_title="Category Sales Forecast", layout="wide")
st.title("📈 Category-wise Sales Forecasting Dashboard")
st.markdown("Select a category to explore and compare forecasts using different time series models.")

# Load and preprocess data
data = load_data("Superstore.xls")
categories = data['Category'].unique().tolist()

# Sidebar for category selection
selected_category = st.sidebar.selectbox("Select a Category", categories)
st.sidebar.markdown("---")

# Filter data for the selected category
category_df = data[data['Category'] == selected_category][['Order Date', 'Sales']]
category_df = category_df.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
category_df['ds'] = pd.to_datetime(category_df['ds'])
category_df = category_df.groupby('ds')['y'].sum().reset_index()
category_df = category_df.sort_values(by='ds')

# Show preview of filtered data
st.subheader(f"📊 Preview: {selected_category} Sales Data")
st.dataframe(category_df.tail(10))

# Forecast horizon
n_months = st.sidebar.slider("Forecast Months", min_value=6, max_value=36, value=24, step=6)

# Run forecasts
st.subheader(f"🔮 Forecast Results for {selected_category} (Next {n_months} Months)")

with st.spinner("Training models and generating forecasts..."):
    prophet_df = forecast_with_prophet(category_df, n_months)
    hw_df = forecast_with_holtwinters(category_df, n_months)
    lstm_df = forecast_with_lstm(category_df, n_months)
    xgb_df = forecast_with_xgboost(category_df, n_months)

# Accuracy metrics section
with st.expander("📉 View Forecast Accuracy Metrics"):
    true_future = category_df.tail(n_months)

    metrics = []

    def safe_compute(true_df, forecast_df, model_name):
        try:
            return compute_metrics(true_df, forecast_df, model_name)
        except ValueError:
            return {"Model": model_name, "MAE": "N/A", "RMSE": "N/A", "MAPE (%)": "N/A"}

    metrics.append(safe_compute(true_future, prophet_df, "Prophet"))
    metrics.append(safe_compute(true_future, hw_df, "Holt-Winters"))
    metrics.append(safe_compute(true_future, lstm_df, "LSTM"))
    metrics.append(safe_compute(true_future, xgb_df, "XGBoost"))

    st.dataframe(pd.DataFrame(metrics))


# Display combined plot
# Compute metrics
st.subheader("📐 Forecast Accuracy Metrics")

st.write("📏 Forecast lengths:")
st.write(f"Prophet: {len(prophet_df)}, HW: {len(hw_df)}, LSTM: {len(lstm_df)}, XGB: {len(xgb_df)}") 

# Compare predictions with actuals
true_future = category_df.tail(n_months)

# Debug: check dates
st.subheader("🧪 Debug: Date overlap check")
st.write("✅ Actual (true_future):")
st.dataframe(true_future.tail(10))

st.write("🔮 Holt-Winters forecast:")
st.dataframe(hw_df.tail(10))

# Compute metrics only if 'ds' matches
metrics = []

metrics.append(compute_metrics(true_future, prophet_df, "Prophet"))
metrics.append(compute_metrics(true_future, hw_df, "Holt-Winters"))
metrics.append(compute_metrics(true_future, lstm_df, "LSTM"))
metrics.append(compute_metrics(true_future, xgb_df, "XGBoost"))

# Show metrics in table
metrics_df = pd.DataFrame(metrics)
st.dataframe(metrics_df)

plot_forecasts(category_df, prophet_df, hw_df, lstm_df, xgb_df, selected_category)

st.success("Forecasting completed!")


