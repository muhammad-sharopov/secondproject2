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
# Fill missing values with the median of the column
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Кэширование результатов моделей
@st.cache_data
def train_and_predict_lr(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)

@st.cache_data
def train_and_predict_rf(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model.predict(X_test)

@st.cache_data
def train_and_predict_cb(X_train, y_train, X_test):
    model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, cat_features=['Month', 'DayOfWeek'], verbose=0)
    model.fit(X_train, y_train)
    return model.predict(X_test)

@st.cache_data
def train_and_predict_lstm(X_train, y_train, X_test):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    X_train_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    model.fit(X_train_lstm, y_train, epochs=10, batch_size=32)
    predicted_prices = model.predict(X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1)))
    return predicted_prices.flatten()

@st.cache_data
def train_and_predict_prophet(df):
    df_prophet = df.reset_index()[['Date', 'Price']]
    df_prophet.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df_prophet)
    forecast = model.predict(df_prophet)
    return forecast['yhat']

@st.cache_data
def train_and_predict_arima(train):
    model = ARIMA(train['Price'], order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit.forecast(steps=len(train))

@st.cache_data
def train_and_predict_sarima(train):
    model = SARIMAX(train['Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    return model_fit.forecast(steps=len(train))

# Прогнозы
if show_lr:
    y_pred_lr = train_and_predict_lr(X_train, train['Price'], X_test)
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_lr, name="Linear Regression"))

if show_rf:
    y_pred_rf = train_and_predict_rf(X_train, train['Price'], X_test)
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_rf, name="Random Forest"))

if show_cb:
    y_pred_cb = train_and_predict_cb(X_train, train['Price'], X_test)
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_cb, name="CatBoost"))

if show_lstm:
    y_pred_lstm = train_and_predict_lstm(X_train, train['Price'], X_test)
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_lstm, name="LSTM"))

if show_prophet:
    y_pred_prophet = train_and_predict_prophet(df_feat)
    fig.add_trace(go.Scatter(x=df_feat.index, y=y_pred_prophet, name="Prophet"))

if show_arima:
    y_pred_arima = train_and_predict_arima(train)
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_arima, name="ARIMA"))

if show_sarima:
    y_pred_sarima = train_and_predict_sarima(train)
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_sarima, name="SARIMA"))

fig.update_layout(title="Прогнозы моделей", xaxis_title="Дата", yaxis_title="Цена", legend_title="Модели")
st.plotly_chart(fig, use_container_width=True)





# Выбираем модель с минимальной MSE
errors = {}

# Прогнозы для каждой модели и их ошибки
if show_lr:
    y_pred_lr = train_and_predict_lr(X_train, train['Price'], X_test)
    mse_lr = mean_squared_error(test['Price'], y_pred_lr)
    mae_lr = mean_absolute_error(test['Price'], y_pred_lr)
    errors["Linear Regression"] = {"mse": mse_lr, "mae": mae_lr}
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_lr, name="Linear Regression"))

if show_rf:
    y_pred_rf = train_and_predict_rf(X_train, train['Price'], X_test)
    mse_rf = mean_squared_error(test['Price'], y_pred_rf)
    mae_rf = mean_absolute_error(test['Price'], y_pred_rf)
    errors["Random Forest"] = {"mse": mse_rf, "mae": mae_rf}
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_rf, name="Random Forest"))

if show_cb:
    y_pred_cb = train_and_predict_cb(X_train, train['Price'], X_test)
    mse_cb = mean_squared_error(test['Price'], y_pred_cb)
    mae_cb = mean_absolute_error(test['Price'], y_pred_cb)
    errors["CatBoost"] = {"mse": mse_cb, "mae": mae_cb}
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_cb, name="CatBoost"))

if show_lstm:
    y_pred_lstm = train_and_predict_lstm(X_train, train['Price'], X_test)
    mse_lstm = mean_squared_error(test['Price'], y_pred_lstm)
    mae_lstm = mean_absolute_error(test['Price'], y_pred_lstm)
    errors["LSTM"] = {"mse": mse_lstm, "mae": mae_lstm}
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_lstm, name="LSTM"))

if show_prophet:
    y_pred_prophet = train_and_predict_prophet(df_feat)
    mse_prophet = mean_squared_error(df_feat['Price'], y_pred_prophet)
    mae_prophet = mean_absolute_error(df_feat['Price'], y_pred_prophet)
    errors["Prophet"] = {"mse": mse_prophet, "mae": mae_prophet}
    fig.add_trace(go.Scatter(x=df_feat.index, y=y_pred_prophet, name="Prophet"))

if show_arima:
    y_pred_arima = train_and_predict_arima(train)
    mse_arima = mean_squared_error(test['Price'], y_pred_arima)
    mae_arima = mean_absolute_error(test['Price'], y_pred_arima)
    errors["ARIMA"] = {"mse": mse_arima, "mae": mae_arima}
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_arima, name="ARIMA"))

if show_sarima:
    y_pred_sarima = train_and_predict_sarima(train)
    mse_sarima = mean_squared_error(test['Price'], y_pred_sarima)
    mae_sarima = mean_absolute_error(test['Price'], y_pred_sarima)
    errors["SARIMA"] = {"mse": mse_sarima, "mae": mae_sarima}
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_sarima, name="SARIMA"))
