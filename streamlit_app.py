import streamlit as st
import pandas as pd
import plotly.express as px

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




if "2020-04-20" in df.index:
    df.loc["2020-04-20", "Price"] = (df.loc["2020-04-17", "Price"] + df.loc["2020-04-21", "Price"]) / 2

# Выбор периода
min_date = df.index.min().date()
max_date = df.index.max().date()

start_date, end_date = st.date_input(
    "Выберите диапазон дат",
    value=(datetime(2019, 7, 6), datetime(2020, 7, 6)),
    min_value=min_date,
    max_value=max_date
)

# Фильтрация данных
filtered_df = df[start_date:end_date]

# Отрисовка интерактивного графика
fig = px.line(
    filtered_df,
    x=filtered_df.index,
    y="Price",
    labels={"x": "Дата", "Price": "Цена нефти (USD)"},
    title="График цены нефти за выбранный период"
)
fig.update_layout(xaxis_title="Дата", yaxis_title="Цена", hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)
