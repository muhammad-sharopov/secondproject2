import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go

st.title("Brent Oil Data — Feature Engineering Demo")

@st.cache_data
def get_data_from_EIA_local():
    df = pd.read_csv("brent-daily.csv", header=0)
    df = df[['Date', 'Price']]
    df['Date'] = pd.to_datetime(df['Date'])
    df['Price'] = df['Price'].astype(float)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

def add_features(df):
    df['Price_MA7'] = df['Price'].rolling(window=7).mean()
    df['Price_MA30'] = df['Price'].rolling(window=30).mean()
    df['Price_STD7'] = df['Price'].rolling(window=7).std()
    df['Price_diff1'] = df['Price'].diff()
    df['Return'] = df['Price'].pct_change()
    df['Price_lag1'] = df['Price'].shift(1)
    df['Price_lag7'] = df['Price'].shift(7)
    df['Month'] = df.index.month
    df['DayOfWeek'] = df.index.dayofweek
    return df

# Загрузка данных
df = get_data_from_EIA_local()
df_feat = add_features(df.copy())

# Радио-переключатель
view_option = st.radio(
    "Выберите отображение данных:",
    ("До обработки", "После обработки")
)

# Отображение данных
if view_option == "До обработки":
    st.subheader("Исходные данные")
    st.dataframe(df.head(10))
else:
    st.subheader("Данные с признаками")
    st.dataframe(df_feat.head(10))

st.subheader("График цены Brent с выбором диапазона по годам")

# Получение минимального и максимального годов
min_year = df_feat.index.year.min()
max_year = df_feat.index.year.max()

# Слайдер для выбора диапазона
start_year, end_year = st.slider(
    "Выберите диапазон годов:",
    min_value=int(min_year),
    max_value=int(max_year),
    value=(int(min_year), int(max_year)),
    step=1
)

# Фильтрация по выбранному диапазону
df_range = df_feat[(df_feat.index.year >= start_year) & (df_feat.index.year <= end_year)]

# Построение графика
fig = px.line(
    df_range.reset_index(),
    x="Date",
    y="Price",
    title=f"Цена Brent с {start_year} по {end_year}",
    labels={"Price": "Цена ($)", "Date": "Дата"}
)
fig.update_layout(xaxis_title="Дата", yaxis_title="Цена ($)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 Модели прогнозирования")

# Исходные данные
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Price'], name="Исходные данные", line=dict(color='black')))

# Общие переменные
train_size = int(len(df) * 0.7)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Checkboxes
show_lr = st.checkbox("Linear Regression")
show_rf = st.checkbox("Random Forest")
show_cb = st.checkbox("CatBoost")
show_lstm = st.checkbox("LSTM")
show_prophet = st.checkbox("Prophet")
show_arima = st.checkbox("ARIMA")
show_sarima = st.checkbox("SARIMA")

# Прогнозы для моделей
# Linear Regression
if show_lr:
    lr_model = LinearRegression()
    lr_model.fit(train[['Price_lag1', 'Price_MA7', 'Price_STD7', 'Return', 'DayOfWeek', 'Month']], train['Price'])
    y_pred_lr = lr_model.predict(test[['Price_lag1', 'Price_MA7', 'Price_STD7', 'Return', 'DayOfWeek', 'Month']])
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_lr, name="Linear Regression"))

# Random Forest
if show_rf:
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(train[['Price_lag1', 'Price_MA7', 'Price_STD7', 'Return', 'DayOfWeek', 'Month']], train['Price'])
    y_pred_rf = rf_model.predict(test[['Price_lag1', 'Price_MA7', 'Price_STD7', 'Return', 'DayOfWeek', 'Month']])
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_rf, name="Random Forest"))

# CatBoost
if show_cb:
    cb_model = CatBoostRegressor(verbose=0, random_state=42)
    cb_model.fit(train[['Price_lag1', 'Price_MA7', 'Price_STD7', 'Return', 'DayOfWeek', 'Month']], train['Price'])
    y_pred_cb = cb_model.predict(test[['Price_lag1', 'Price_MA7', 'Price_STD7', 'Return', 'DayOfWeek', 'Month']])
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_cb, name="CatBoost"))

# LSTM
if show_lstm:
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
    lstm_index = df.index[-len(predicted_prices):]
    fig.add_trace(go.Scatter(x=lstm_index, y=predicted_prices.flatten(), name="LSTM"))

# Prophet
if show_prophet:
    prophet_model = Prophet()
    prophet_data = df.reset_index()[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'})
    prophet_model.fit(prophet_data)
    future = prophet_model.make_future_dataframe(prophet_data, periods=len(test))
    forecast = prophet_model.predict(future)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Prophet"))

# ARIMA
if show_arima:
    arima_model = ARIMA(train['Price'], order=(5, 1, 0)).fit()
    arima_pred = arima_model.forecast(steps=len(test))
    fig.add_trace(go.Scatter(x=test.index, y=arima_pred, name="ARIMA"))

# SARIMA
if show_sarima:
    sarima_model = SARIMAX(train['Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
    sarima_pred = sarima_model.forecast(steps=len(test))
    fig.add_trace(go.Scatter(x=test.index, y=sarima_pred, name="SARIMA"))

fig.update_layout(title="Прогнозы моделей", xaxis_title="Дата", yaxis_title="Цена", legend_title="Модели")
st.plotly_chart(fig, use_container_width=True)
