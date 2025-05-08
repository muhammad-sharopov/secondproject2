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

train_size = int(len(df_feat) * 0.7)  
train, test = df_feat.iloc[:train_size], df_feat.iloc[train_size:]

# Define X_train and X_test with the feature columns
X_train = train[['Price_lag1', 'Price_MA7', 'Price_STD7', 'Return', 'DayOfWeek', 'Month']]
X_test = test[['Price_lag1', 'Price_MA7', 'Price_STD7', 'Return', 'DayOfWeek', 'Month']]

# Исходные данные
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['Price'], name="Исходные данные", line=dict(color='black')))

if show_lr:
    # Ensure you're using the df_feat DataFrame with the added features
    lr_model = LinearRegression()
    X_train = train[['Price_lag1', 'Price_MA7', 'Price_STD7', 'Return', 'DayOfWeek', 'Month']]
    X_test = test[['Price_lag1', 'Price_MA7', 'Price_STD7', 'Return', 'DayOfWeek', 'Month']]
    lr_model.fit(X_train, train['Price'])
    y_pred_lr = lr_model.predict(X_test)
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_lr, name="Linear Regression"))

if show_rf:
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train, train['Price'])
    y_pred_rf = rf_model.predict(X_test)
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_rf, name="Random Forest"))

if show_cb:
    cb_model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, cat_features=['Month', 'DayOfWeek'], verbose=0)
    cb_model.fit(X_train, train['Price'])
    y_pred_cb = cb_model.predict(X_test)
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_cb, name="CatBoost"))

if show_lstm:
    # Assume lstm_model is trained here (same structure as previous code)
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    lstm_model.add(LSTM(units=50, return_sequences=False))
    lstm_model.add(Dense(units=1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    # Assuming reshaped X_train for LSTM
    X_train_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    lstm_model.fit(X_train_lstm, train['Price'], epochs=10, batch_size=32)
    predicted_prices = lstm_model.predict(X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1)))
    fig.add_trace(go.Scatter(x=test.index, y=predicted_prices.flatten(), name="LSTM"))

if show_prophet:
    df_prophet = df_feat.reset_index()[['Date', 'Price']]
    df_prophet.columns = ['ds', 'y']
    prophet_model = Prophet()
    prophet_model.fit(df_prophet)
    forecast = prophet_model.predict(df_prophet)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Prophet"))

if show_arima:
    arima_model = ARIMA(train['Price'], order=(5, 1, 0))
    arima_model_fit = arima_model.fit()
    arima_pred = arima_model_fit.forecast(steps=len(test))
    fig.add_trace(go.Scatter(x=test.index, y=arima_pred, name="ARIMA"))

if show_sarima:
    sarima_model = SARIMAX(train['Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_model_fit = sarima_model.fit(disp=False)
    sarima_pred = sarima_model_fit.forecast(steps=len(test))
    fig.add_trace(go.Scatter(x=test.index, y=sarima_pred, name="SARIMA"))

fig.update_layout(title="Прогнозы моделей", xaxis_title="Дата", yaxis_title="Цена", legend_title="Модели")
st.plotly_chart(fig, use_container_width=True)
