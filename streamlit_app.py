import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# Function to get data
@st.cache_data
def get_data_from_EIA_local():
    df = pd.read_csv("brent-daily.csv", header=0)
    df = df[['Date', 'Price']]
    df['Date'] = pd.to_datetime(df['Date'])
    df['Price'] = df['Price'].astype(float)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

# Function to make future dates
def make_future_dates(last_date, period):
    prediction_dates = pd.date_range(last_date, periods=period + 1, freq='B')
    return prediction_dates[1:]

# Function to evaluate model
def evaluate(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

def evaluate_model(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'label': label, 'rmse': rmse, 'mae': mae}

st.title("Анализ и прогнозирование цен на нефть Brent")

df = get_data_from_EIA_local()

st.subheader("Исходные данные")
st.dataframe(df.head())

data = df['2019-07-06':'2020-07-06'].copy()
data.Price["2020-04-20"] = (data.Price["2020-04-17"] + data.Price["2020-04-21"]) / 2

st.subheader("График цены на нефть (2019-07-06 по 2020-07-06)")
fig_data, ax_data = plt.subplots(figsize=(10, 5))
ax_data.plot(data['Price'])
ax_data.set_xlabel('Дата')
ax_data.set_ylabel('Цена')
ax_data.grid(True)
st.pyplot(fig_data)

st.subheader("График цены на нефть (полный период)")
fig_full, ax_full = plt.subplots(figsize=(14, 6))
ax_full.plot(df.index, df['Price'], label='USD', color='blue')
ax_full.set_title('Цена нефти в долларах', fontsize=16)
ax_full.set_xlabel('Дата')
ax_full.set_ylabel('Цена нефти')
ax_full.grid(True)
ax_full.legend()
fig_full.tight_layout()
st.pyplot(fig_full)

st.subheader("Сезонная декомпозиция")
res = seasonal_decompose(df['Price'], model='additive', period=365)
fig_decomp, axes_decomp = plt.subplots(4, 1, sharex=True, figsize=(20, 10))
res.observed.plot(ax=axes_decomp[0], legend=False, color='r')
axes_decomp[0].set_ylabel('Observed')
res.trend.plot(ax=axes_decomp[1], legend=False, color='g')
axes_decomp[1].set_ylabel('Trend')
res.seasonal.plot(ax=axes_decomp[2], legend=False)
axes_decomp[2].set_ylabel('Seasonal')
res.resid.plot(ax=axes_decomp[3], legend=False, color='k')
axes_decomp[3].set_ylabel('Residual')
fig_decomp.tight_layout()
st.pyplot(fig_decomp)

df['Price_MA7'] = df['Price'].rolling(window=7).mean()
df['Price_MA30'] = df['Price'].rolling(window=30).mean()
df['Price_STD7'] = df['Price'].rolling(window=7).std()
df['Price_diff1'] = df['Price'].diff()
df['Return'] = df['Price'].pct_change()
df['Price_lag1'] = df['Price'].shift(1)
df['Price_lag7'] = df['Price'].shift(7)
df['Month'] = df.index.month
df['DayOfWeek'] = df.index.dayofweek
df = df.dropna()

train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

features = ['Price_lag1', 'Price_MA7', 'Price_STD7', 'Return', 'DayOfWeek', 'Month']
X_train = train[features]
X_test = test[features]
y_train = train['Price']
y_test = test['Price']

st.subheader("Прогнозирование с помощью классических моделей")

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
metrics_lr = evaluate(y_test, y_pred_lr)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
metrics_rf = evaluate(y_test, y_pred_rf)

# CatBoost
cb_model = CatBoostRegressor(verbose=0, random_state=42)
cb_model.fit(X_train, y_train)
y_pred_cb = cb_model.predict(X_test)
metrics_cb = evaluate(y_test, y_pred_cb)

fig_models, ax_models = plt.subplots(figsize=(15, 6))
ax_models.plot(y_test.index, y_test, label='Actual Price', color='black')
ax_models.plot(y_test.index, y_pred_lr, label='Linear Regression')
ax_models.plot(y_test.index, y_pred_rf, label='Random Forest')
ax_models.plot(y_test.index, y_pred_cb, label='CatBoost')
ax_models.set_title('Сравнение прогнозов')
ax_models.set_xlabel('Дата')
ax_models.set_ylabel('Цена')
ax_models.legend()
ax_models.grid(True)
st.pyplot(fig_models)

metrics = {
    'Linear Regression': metrics_lr,
    'Random Forest': metrics_rf,
    'CatBoost': metrics_cb
}

metrics_df = pd.DataFrame(metrics).T
st.subheader("Метрики классических моделей")
st.dataframe(metrics_df)

st.subheader("Анализ стационарности ряда")
price_diff = df['Price'].diff().dropna()
fig_acf_pacf, axes_acf_pacf = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(price_diff, lags=40, ax=axes_acf_pacf[0])
axes_acf_pacf[0].set_title('ACF (Автокорреляция)')
plot_pacf(price_diff, lags=40, ax=axes_acf_pacf[1], method='ywm')
axes_acf_pacf[1].set_title('PACF (Частичная автокорреляция)')
fig_acf_pacf.tight_layout()
st.pyplot(fig_acf_pacf)

series = df['Price'].dropna()
adf_result = adfuller(series)
st.write(f'ADF Statistic: {adf_result[0]:.4f}')
st.write(f'p-value: {adf_result[1]:.4f}')
if adf_result[1] < 0.05:
    st.write("✅ Ряд стационарен — можно использовать d=0")
    d_param = 0
else:
    st.write("❌ Ряд нестационарен — используем d=1")
    d_param = 1

st.subheader("Прогнозирование с помощью ARIMA и SARIMAX")
# ARIMA
try:
    arima_model = ARIMA(y_train, order=(5, d_param, 5)).fit()
    y_pred_arima = arima_model.forecast(steps=len(y_test))
    arima_metrics = evaluate_model(y_test, y_pred_arima, 'ARIMA')
except Exception as e:
    st.error(f"Ошибка при обучении ARIMA: {e}")
    arima_metrics = None

# SARIMAX
try:
    train_exog = train.index.month.values.reshape(-1, 1)
    test_exog = test.index.month.values.reshape(-1, 1)
    sarimax_model = SARIMAX(y_train, exog=train_exog, order=(1, d_param, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    y_pred_sarimax = sarimax_model.forecast(steps=len(y_test), exog=test_exog)
    sarimax_metrics = evaluate_model(y_test, y_pred_sarimax, 'SARIMAX')
except Exception as e:
    st.error(f"Ошибка при обучении SARIMAX: {e}")
    sarimax_metrics = None

fig_ts, ax_ts = plt.subplots(figsize=(20, 8))
ax_ts.plot(df.index, df['Price'], label='Actual', color='black')
if arima_metrics:
    ax_ts.plot(test.index, y_pred_arima, label=f'{arima_metrics["label"]} (RMSE: {arima_metrics["rmse"]:.2f}, MAE: {arima_metrics["mae"]:.2f})', color='green')
if sarimax_metrics:
    ax_ts.plot(test.index, y_pred_sarimax, label=f'{sarimax_metrics["label"]} (RMSE: {sarimax_metrics["rmse"]:.2f}, MAE: {sarimax_metrics["mae"]:.2f})', color='purple')
ax_ts.set_title("Сравнение прогнозов временных рядов")
ax_ts.set_xlabel("Дата")
ax_ts.set_ylabel("Цена")
ax_ts.legend()
ax_ts.grid(True)
fig_ts.tight_layout()
st.pyplot(fig_ts)

results_ts = []
if arima_metrics:
    results_ts.append(arima_metrics)
if sarimax_metrics:
    results_ts.append(sarimax_metrics)

if results_ts:
    results_df_ts = pd.DataFrame(results_ts)
    st.subheader("Метрики ARIMA и SARIMAX")
    st.dataframe(results_df_ts)

st.subheader("Прогнозирование с помощью LSTM")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Price']])
sequence_length = 50
X = []
y = []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
train_size_lstm = int(len(X) * 0.7)
X_train_lstm, X_test_lstm = X[:train_size_lstm], X[train_size_lstm:]
y_train_lstm, y_test_lstm = y[:train_size_lstm], y[train_size_lstm:]

lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train_lstm.shape[1], 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=64, verbose=0)

predicted_lstm = lstm_model.predict(X_test_lstm)
predicted_prices_lstm = scaler.inverse_transform(predicted_lstm.reshape(-1, 1))
real_prices_lstm = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))

rmse_lstm = np.sqrt(mean_squared_error(real_prices_lstm, predicted_prices_lstm))
mae_lstm = mean_absolute_error(real_prices_lstm, predicted_prices_lstm)
r2_lstm = r2_score(real_prices_lstm, predicted_prices_lstm)

st.write(f"LSTM - RMSE: {rmse_lstm:.4f}")
st.write(f"LSTM - MAE : {mae_lstm:.4f}")
st.write(f"LSTM - R²  : {r2_lstm:.4f}")

fig_lstm, ax_lstm = plt.subplots(figsize=(12, 6))
ax_lstm.plot(df.iloc[len(train_size_lstm) + sequence_length:].index, real_prices_lstm, label='Фактические цены')
ax_lstm.plot(df.iloc[len(train_size_lstm) + sequence_length:].index, predicted_prices_lstm, label='LSTM прогноз')
ax_lstm.set_title('LSTM — прогноз цен')
ax_lstm.set_xlabel('Время')
ax_lstm.set_ylabel('Цена')
ax_lstm.legend()
st.pyplot(fig_lstm)

st.subheader("Прогнозирование с помощью Prophet")
df_prophet = df.reset_index().rename(columns={"Date": "ds", "Price": "y"})
train_size_prophet = int(len(df_prophet) * 0.7)
train_prophet = df_prophet.iloc[:train_size_prophet]
test_prophet = df_prophet.iloc[train_size_prophet:]

prophet_model = Prophet()
prophet_model.fit(train_prophet)
future_prophet = test_prophet[['ds']].copy()
forecast_prophet = prophet_model.predict(future_prophet)

fig_prophet = plt.figure(figsize=(12, 6))
plt.plot(test_prophet['ds'], test_prophet['y'], label='Фактические цены')
plt.plot(forecast_prophet['ds'], forecast_prophet['yhat'], label='Прогноз (Prophet)')
plt.fill_between(forecast_prophet['ds'], forecast_prophet['yhat_lower'], forecast_prophet['yhat_upper'], alpha=0.3)
plt.title('Прогноз цен с помощью Prophet')
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.legend()
st.pyplot(fig_prophet)

rmse_prophet = np.sqrt(mean_squared_error(test_prophet['y'], forecast_prophet['yhat']))
mae_prophet = mean_absolute_error(test_prophet['y'], forecast_prophet['yhat'])
r2_prophet = r2_score(test_prophet['y'], forecast_prophet['yhat'])

st.write(f"Prophet - RMSE: {rmse_prophet:.4f}")
st.write(f"Prophet - MAE : {mae_prophet:.4f}")
st.write(f"Prophet - R²  : {r2_prophet:.4f}")
