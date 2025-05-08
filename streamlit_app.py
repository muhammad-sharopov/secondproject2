import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# Загрузка данных
def get_data_from_EIA_local():
    df = pd.read_csv("brent-daily.csv", header=0)
    df = df[['Date', 'Price']] 
    df['Date'] = pd.to_datetime(df['Date'])
    df['Price'] = df['Price'].astype(float) 
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

# Функция для построения графика
def plot_results(actual, predicted, title):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Prices')
    plt.plot(predicted, label='Predicted Prices')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot()

# Оценка моделей
def evaluate_model(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'model': label, 'rmse': rmse, 'mae': mae, 'r2': r2}

# Функция для прогноза с использованием Prophet
def prophet_forecast(df):
    df_prophet = df.reset_index().rename(columns={"Date": "ds", "Price": "y"})
    train_size = int(len(df_prophet) * 0.7)
    train = df_prophet.iloc[:train_size]
    test = df_prophet.iloc[train_size:]
    
    model = Prophet()
    model.fit(train)
    future = test[['ds']].copy()
    forecast = model.predict(future)
    
    return forecast, test

# Старт Streamlit
st.title('Time Series Forecasting with Multiple Models')

# Загрузка данных
df = get_data_from_EIA_local()

# Отображаем данные
st.subheader('Data')
st.write(df.head())

# Прогнозирование с Prophet
st.subheader('Prophet Forecasting')
forecast, test = prophet_forecast(df)
plot_results(test['y'], forecast['yhat'], 'Prophet Forecast vs Actual')

# Прогнозирование с ARIMA
st.subheader('ARIMA Forecasting')
arima_model = ARIMA(df['Price'], order=(5,1,0)).fit()
y_pred_arima = arima_model.forecast(steps=len(df))
plot_results(df['Price'], y_pred_arima, 'ARIMA Forecast vs Actual')

# Прогнозирование с SARIMAX
st.subheader('SARIMAX Forecasting')
train_exog = df.index.month.values.reshape(-1, 1)
sarimax_model = SARIMAX(df['Price'], exog=train_exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
y_pred_sarimax = sarimax_model.forecast(steps=len(df), exog=train_exog)
plot_results(df['Price'], y_pred_sarimax, 'SARIMAX Forecast vs Actual')

# Прогнозирование с LSTM
st.subheader('LSTM Forecasting')
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

predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted.reshape(-1, 1))
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

plot_results(real_prices, predicted_prices, 'LSTM Forecast vs Actual')

# Прогнозирование с CatBoost
st.subheader('CatBoost Forecasting')
catboost_model = CatBoostRegressor(iterations=100, depth=5, learning_rate=0.1, random_seed=42, verbose=0)
catboost_model.fit(X_train.reshape(-1, X_train.shape[1]), y_train)
y_pred_catboost = catboost_model.predict(X_test.reshape(-1, X_test.shape[1]))
plot_results(real_prices, y_pred_catboost, 'CatBoost Forecast vs Actual')

# Оценка моделей
metrics = []
metrics.append(evaluate_model(df['Price'], y_pred_arima, 'ARIMA'))
metrics.append(evaluate_model(df['Price'], y_pred_sarimax, 'SARIMAX'))
metrics.append(evaluate_model(df['Price'], predicted_prices, 'LSTM'))
metrics.append(evaluate_model(df['Price'], y_pred_catboost, 'CatBoost'))

metrics_df = pd.DataFrame(metrics)
st.subheader('Model Evaluation')
st.write(metrics_df)

# Отображаем исходный график всех данных
st.subheader('Historical Data')

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Price'], label='Actual Prices', color='blue')
plt.title('Historical Prices of Brent Oil')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot()  # Показываем график в Streamlit

# Добавляем интерактивный график для прогноза
st.subheader('Interactive Model Comparison')

# Кнопка для отображения моделей
model_names = ['Prophet', 'ARIMA', 'SARIMAX', 'LSTM', 'CatBoost']
selected_model = st.selectbox("Choose a model to display predictions", model_names)

# Визуализация выбранной модели
if selected_model == 'Prophet':
    forecast, test = prophet_forecast(df)
    plot_results(test['y'], forecast['yhat'], 'Prophet Forecast vs Actual')
    st.pyplot()  # Показываем график в Streamlit

elif selected_model == 'ARIMA':
    arima_model = ARIMA(df['Price'], order=(5,1,0)).fit()
    y_pred_arima = arima_model.forecast(steps=len(df))
    plot_results(df['Price'], y_pred_arima, 'ARIMA Forecast vs Actual')
    st.pyplot()  # Показываем график в Streamlit

elif selected_model == 'SARIMAX':
    train_exog = df.index.month.values.reshape(-1, 1)
    sarimax_model = SARIMAX(df['Price'], exog=train_exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
    y_pred_sarimax = sarimax_model.forecast(steps=len(df), exog=train_exog)
    plot_results(df['Price'], y_pred_sarimax, 'SARIMAX Forecast vs Actual')
    st.pyplot()  # Показываем график в Streamlit

elif selected_model == 'LSTM':
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

    predicted = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted.reshape(-1, 1))
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    plot_results(real_prices, predicted_prices, 'LSTM Forecast vs Actual')
    st.pyplot()  # Показываем график в Streamlit

elif selected_model == 'CatBoost':
    catboost_model = CatBoostRegressor(iterations=100, depth=5, learning_rate=0.1, random_seed=42, verbose=0)
    catboost_model.fit(X_train.reshape(-1, X_train.shape[1]), y_train)
    y_pred_catboost = catboost_model.predict(X_test.reshape(-1, X_test.shape[1]))
    plot_results(real_prices, y_pred_catboost, 'CatBoost Forecast vs Actual')
    st.pyplot()  # Показываем график в Streamlit
