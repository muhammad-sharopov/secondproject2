import streamlit as st
import pandas as pd

st.title("Brent Oil Data - Feature Engineering Demo")

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
df_raw = get_data_from_EIA_local()

# Интерфейс
col1, col2 = st.columns(2)

with col1:
    if st.button("Показать данные ДО обработки"):
        st.subheader("Исходные данные")
        st.dataframe(df_raw.head(10))

with col2:
    if st.button("Показать данные ПОСЛЕ обработки"):
        st.subheader("Данные с признаками")
        df_feat = add_features(df_raw.copy())
        st.dataframe(df_feat.head(10))
