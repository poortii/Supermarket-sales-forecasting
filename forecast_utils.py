import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

def load_data(filepath):
    df = pd.read_excel(filepath)
    return df

def train_prophet(df):
    df = df.rename(columns={"Order Date": "ds", "Sales": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=12, freq="MS")
    forecast = model.predict(future)
    return model, forecast


def train_holtwinters(y):
    model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(12)
    return fitted_model, forecast


def train_lstm(y):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(y.values.reshape(-1, 1))
    X, y_train = [], []
    for i in range(12, len(scaled_data)):
        X.append(scaled_data[i - 12:i, 0])
        y_train.append(scaled_data[i, 0])
    X, y_train = np.array(X), np.array(y_train)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y_train, epochs=10, batch_size=1, verbose=0)

    inputs = scaled_data[-12:]
    X_test = np.array([inputs[:, 0]])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    preds = []
    for _ in range(12):
        pred = model.predict(X_test)
        preds.append(pred[0, 0])
        X_test = np.append(X_test[:, 1:, :], [[pred]], axis=1)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    return model, preds.flatten()


def train_xgboost(y):
    df = pd.DataFrame({"y": y.values})
    df["lag1"] = df["y"].shift(1)
    df["lag2"] = df["y"].shift(2)
    df.dropna(inplace=True)
    X = df[["lag1", "lag2"]]
    y_train = df["y"]

    model = XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X, y_train)

    last1, last2 = y.iloc[-1], y.iloc[-2]
    preds = []
    for _ in range(12):
        pred = model.predict(np.array([[last1, last2]]))[0]
        preds.append(pred)
        last2, last1 = last1, pred
    return model, np.array(preds)

from prophet import Prophet

def forecast_with_prophet(df, n_months):
    df = df.copy()
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=n_months, freq='MS')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

def forecast_with_lstm(df, n_months):
    df = df.copy()
    df.set_index('ds', inplace=True)
    df = df.asfreq('MS')
    
    values = df['y'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    def create_dataset(data, look_back=12):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i + look_back])
            y.append(data[i + look_back])
        return np.array(X), np.array(y)

    look_back = 12
    X, y = create_dataset(scaled_values, look_back)

    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=0)

    forecast = []
    input_seq = scaled_values[-look_back:]

    for _ in range(n_months):
        inp = input_seq.reshape((1, look_back, 1))
        pred = model.predict(inp, verbose=0)[0][0]
        forecast.append(pred)
        input_seq = np.append(input_seq[1:], [[pred]], axis=0)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=n_months, freq='MS')
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast})

    return forecast_df

from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

def forecast_with_xgboost(df, n_months):
    df = df.copy()
    df.set_index('ds', inplace=True)
    df = df.asfreq('MS')
    df['y'] = df['y'].fillna(method='ffill')

    # Feature engineering: add month and lag features
    df['month'] = df.index.month
    df['lag1'] = df['y'].shift(1)
    df['lag2'] = df['y'].shift(2)
    df['lag3'] = df['y'].shift(3)
    df = df.dropna()

    X = df[['month', 'lag1', 'lag2', 'lag3']]
    y = df['y']
    
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)

    last_known = df.iloc[-3:].copy()
    future_dates = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=n_months, freq='MS')

    preds = []
    for date in future_dates:
        month = date.month
        lags = last_known['y'].values[-3:]
        input_data = pd.DataFrame([[month, lags[-1], lags[-2], lags[-3]]], columns=['month', 'lag1', 'lag2', 'lag3'])
        pred = model.predict(input_data)[0]
        preds.append(pred)

        # Update last_known
        new_row = pd.DataFrame({'y': [pred]}, index=[date])
        last_known = pd.concat([last_known, new_row])
        last_known = last_known.iloc[-3:]

    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': preds})
    return forecast_df

def forecast_with_holtwinters(df, n_months):
    df = df.copy()
    df = df.set_index('ds')
    df = df.asfreq('MS')  # monthly start

    # Drop NaNs
    df = df.dropna()

    if len(df) < 24:
        return pd.DataFrame(columns=['ds', 'yhat'])

    model = ExponentialSmoothing(df['y'], trend='add', seasonal='add', seasonal_periods=12)
    fit = model.fit()

    forecast_index = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=n_months, freq='MS')
    forecast = fit.forecast(n_months)

    return pd.DataFrame({'ds': forecast_index, 'yhat': forecast.values})

def plot_forecasts(original_df, prophet_df, hw_df, lstm_df, xgb_df, category_name):
    import matplotlib.pyplot as plt
    import streamlit as st

    plt.figure(figsize=(12, 6))
    plt.plot(original_df['ds'], original_df['y'], label='Actual', color='black')
    if not prophet_df.empty:
        plt.plot(prophet_df['ds'], prophet_df['yhat'], label='Prophet Forecast')
    if not hw_df.empty:
        plt.plot(hw_df['ds'], hw_df['yhat'], label='Holt-Winters Forecast')
    if not lstm_df.empty:
        plt.plot(lstm_df['ds'], lstm_df['yhat'], label='LSTM Forecast')
    if not xgb_df.empty:
        plt.plot(xgb_df['ds'], xgb_df['yhat'], label='XGBoost Forecast')

    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title(f'Forecast Comparison for {category_name}')
    plt.legend()
    st.pyplot(plt)

from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(true_df, forecast_df, model_name):
    merged = pd.merge(true_df, forecast_df, on='ds', how='inner')
    if merged.empty:
        print(f"⚠️ No overlapping dates between actual and forecast for {model_name}.")
        return {
            "Model": model_name,
            "MAE": "N/A",
            "RMSE": "N/A",
            "MAPE (%)": "N/A"
        }

    y_true = merged['y']
    y_pred = merged['yhat']
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2)
    }
