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





# Функция для вычисления метрик модели
def compute_metrics(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError(f"Размерности y_true и y_pred не совпадают: {len(y_true)} != {len(y_pred)}")
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

# Прогнозирование и вычисление метрик для каждой модели
metrics = {}

if show_lr:
    y_pred_lr = train_and_predict_lr(X_train, train['Price'], X_test)
    mse, rmse, mae, r2 = compute_metrics(test['Price'], y_pred_lr)
    metrics['Linear Regression'] = (mse, rmse, mae, r2)

if show_rf:
    y_pred_rf = train_and_predict_rf(X_train, train['Price'], X_test)
    mse, rmse, mae, r2 = compute_metrics(test['Price'], y_pred_rf)
    metrics['Random Forest'] = (mse, rmse, mae, r2)

if show_cb:
    y_pred_cb = train_and_predict_cb(X_train, train['Price'], X_test)
    mse, rmse, mae, r2 = compute_metrics(test['Price'], y_pred_cb)
    metrics['CatBoost'] = (mse, rmse, mae, r2)

if show_lstm:
    y_pred_lstm = train_and_predict_lstm(X_train, train['Price'], X_test)
    mse, rmse, mae, r2 = compute_metrics(test['Price'], y_pred_lstm)
    metrics['LSTM'] = (mse, rmse, mae, r2)

if show_prophet:
    y_pred_prophet = train_and_predict_prophet(df_feat)
    mse, rmse, mae, r2 = compute_metrics(test['Price'], y_pred_prophet)
    metrics['Prophet'] = (mse, rmse, mae, r2)

if show_arima:
    y_pred_arima = train_and_predict_arima(train)
    # Убедитесь, что длина y_pred_arima совпадает с тестовыми данными
    if len(y_pred_arima) != len(test['Price']):
        # Приводим y_pred_arima к той же длине, что и test['Price']
        y_pred_arima = y_pred_arima[:len(test['Price'])]
    mse, rmse, mae, r2 = compute_metrics(test['Price'], y_pred_arima)
    metrics['ARIMA'] = (mse, rmse, mae, r2)

if show_sarima:
    y_pred_sarima = train_and_predict_sarima(train)
    mse, rmse, mae, r2 = compute_metrics(test['Price'], y_pred_sarima)
    metrics['SARIMA'] = (mse, rmse, mae, r2)

# Выбираем модель с наименьшей ошибкой (по MSE)
best_model_name = min(metrics, key=lambda model: metrics[model][0])  # Наименьший MSE

# Печать лучшей модели и ее метрик
st.write(f"Лучшая модель: {best_model_name}")
st.write(f"Метрики для лучшей модели:")
st.write(f"MSE: {metrics[best_model_name][0]:.2f}")
st.write(f"RMSE: {metrics[best_model_name][1]:.2f}")
st.write(f"MAE: {metrics[best_model_name][2]:.2f}")
st.write(f"R2: {metrics[best_model_name][3]:.2f}")

# Прогнозирование с лучшей моделью
if best_model_name == "Linear Regression":
    y_pred_best = train_and_predict_lr(X_train, train['Price'], X_test)
elif best_model_name == "Random Forest":
    y_pred_best = train_and_predict_rf(X_train, train['Price'], X_test)
elif best_model_name == "CatBoost":
    y_pred_best = train_and_predict_cb(X_train, train['Price'], X_test)
elif best_model_name == "LSTM":
    y_pred_best = train_and_predict_lstm(X_train, train['Price'], X_test)
elif best_model_name == "Prophet":
    y_pred_best = train_and_predict_prophet(df_feat)
elif best_model_name == "ARIMA":
    y_pred_best = train_and_predict_arima(train)
elif best_model_name == "SARIMA":
    y_pred_best = train_and_predict_sarima(train)

# Слайдер для предсказания будущей цены
future_days = st.slider(
    "Выберите количество дней для предсказания:",
    min_value=1,
    max_value=365,
    value=30,
    step=1
)

# Прогнозируем будущую цену
if best_model_name in ["Linear Regression", "Random Forest", "CatBoost", "LSTM"]:
    future_X = X_test.tail(future_days)  # Берем последние данные для прогноза
    future_price = y_pred_best[-future_days:]  # Прогноз на будущее
    st.write(f"Предсказанная цена на {future_days} дней вперёд: {future_price[-1]:.2f} USD")

elif best_model_name == "Prophet":
    # Для Prophet используем future data frame
    future = model.make_future_dataframe(df_prophet, periods=future_days)
    forecast = model.predict(future)
    future_price = forecast['yhat'].tail(future_days)
    st.write(f"Предсказанная цена на {future_days} дней вперёд: {future_price.iloc[-1]:.2f} USD")

elif best_model_name in ["ARIMA", "SARIMA"]:
    # Для ARIMA и SARIMA
    future_price = y_pred_best[-future_days:]
    st.write(f"Предсказанная цена на {future_days} дней вперёд: {future_price[-1]:.2f} USD")
