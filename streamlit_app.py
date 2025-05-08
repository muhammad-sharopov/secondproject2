import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

# Загрузка данных
def get_data_from_EIA_local():
    df = pd.read_csv("brent-daily.csv", header=0)
    df = df[['Date', 'Price']] 
    df['Date'] = pd.to_datetime(df['Date'])
    df['Price'] = df['Price'].astype(float) 
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

# Сохранение модели
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Загрузка модели
def load_model(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
    return None

# Прогнозирование с ARIMA
def arima_forecast(df):
    model_filename = 'arima_model.pkl'
    model = load_model(model_filename)
    if model is None:
        model = ARIMA(df['Price'], order=(5,1,0)).fit()
        save_model(model, model_filename)
    y_pred_arima = model.forecast(steps=len(df))
    return y_pred_arima

# Прогнозирование с SARIMAX
def sarimax_forecast(df):
    model_filename = 'sarimax_model.pkl'
    model = load_model(model_filename)
    if model is None:
        train_exog = df.index.month.values.reshape(-1, 1)
        model = SARIMAX(df['Price'], exog=train_exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
        save_model(model, model_filename)
    y_pred_sarimax = model.forecast(steps=len(df), exog=train_exog)
    return y_pred_sarimax

# Прогнозирование с LSTM
def lstm_forecast(df):
    model_filename = 'lstm_model.pkl'
    model = load_model(model_filename)
    if model is None:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Price']])

        sequence_length = 50
        X = []
        y = []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        train_size = int(len(X) * 0.7)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

        save_model(model, model_filename)

    predicted = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted.reshape(-1, 1))
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    return real_prices, predicted_prices

# Прогнозирование с Prophet
def prophet_forecast(df):
    model_filename = 'prophet_model.pkl'
    model = load_model(model_filename)
    if model is None:
        df_prophet = df.reset_index().rename(columns={"Date": "ds", "Price": "y"})
        model = Prophet()
        model.fit(df_prophet)
        save_model(model, model_filename)
    
    future = df.reset_index()[['Date']].rename(columns={'Date': 'ds'})
    forecast = model.predict(future)
    return df['Price'], forecast['yhat']

# Визуализация графиков
def plot_results(df, model_predictions, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Price'], label='Historical Prices', color='blue', linewidth=2)
    plt.plot(df.index, model_predictions, label=f'{model_name} Predictions', color='red', linestyle='--', linewidth=2)
    plt.title(f'{model_name} Forecast vs Historical Data', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    plt.grid(True)
    st.pyplot()

# Интерфейс Streamlit
st.title('Interactive Time Series Forecasting')

# Загрузка данных
df = get_data_from_EIA_local()

# Выбор модели
model_choice = st.selectbox('Select a model to display predictions', 
                            ['ARIMA', 'SARIMAX', 'LSTM', 'Prophet'])

# Выбор отображения графиков
if model_choice == 'ARIMA':
    st.subheader('ARIMA Forecast')
    y_pred_arima = arima_forecast(df)
    plot_results(df, y_pred_arima, 'ARIMA')

elif model_choice == 'SARIMAX':
    st.subheader('SARIMAX Forecast')
    y_pred_sarimax = sarimax_forecast(df)
    plot_results(df, y_pred_sarimax, 'SARIMAX')

elif model_choice == 'LSTM':
    st.subheader('LSTM Forecast')
    real_prices, predicted_prices = lstm_forecast(df)
    plot_results(df, predicted_prices, 'LSTM')

elif model_choice == 'Prophet':
    st.subheader('Prophet Forecast')
    historical, forecast = prophet_forecast(df)
    plot_results(df, forecast, 'Prophet')
