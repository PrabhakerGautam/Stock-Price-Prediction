import streamlit as st
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.gofplots import qqplot
from typing import Union
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf
#import  p-value, q_value from 2_Modelling
import time  
import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline

# Disable warning about passing a figure to st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)
# Set page title and icon
st.set_page_config(page_title="Predict", page_icon="./images/object.png")



# Define the stock symbols
stock_symbols = { 'Google':'GOOGL', 'Apple':'AAPL',
                  'Microsoft': 'MSFT', 
                 'Amazon': 'AMZN', 
                 'Facebook': 'FB', 
                 'Tesla': 'TSLA', 
                 'Alphabet': 'GOOG', 
                 'Netflix': 'NFLX', 
                 'Nvidia': 'NVDA', 
                 'Adobe': 'ADBE', 
                 'Intel': 'INTC', 
                 'PayPal': 'PYPL',
                'Johnson & Johnson': 'JNJ',
                 'Visa': 'V',
                 'JPMorgan Chase': 'JPM',
                 'Walmart': 'WMT',
                 'Procter & Gamble': 'PG',
                 'Bank of America': 'BAC',
                 'Mastercard': 'MA',
                 'Verizon': 'VZ',
                 'Coca-Cola': 'KO',
                 'Disney': 'DIS'}


# Function to download stock data
def download_stock_data(stock_symbol, start_date, end_date):
    df = yf.download(stock_symbol, start_date, end=end_date)
    df.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
    return df

import streamlit as st
from tqdm import tqdm
def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int, window: int,p_value,q_value, method: str) -> list:

    total_len = train_len + horizon

    if method == 'mean':
        pred_mean = []

        for i in range(train_len, total_len, window):
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))

        return pred_mean

    elif method == 'last':
        pred_last_value = []

        for i in range(train_len, total_len, window):
            last_value = df[:i].iloc[-1].values[0]
            pred_last_value.extend(last_value for _ in range(window))

        return pred_last_value

    elif method == 'ARIMA':
        pred_ARIMA = []

        for i in range(train_len, total_len, window):
            model = SARIMAX(df[:i], order=(p_value,1,q_value))
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window - 1)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_ARIMA.extend(oos_pred)

        return pred_ARIMA


def get_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None        



# Main function for the app
size = 1

def main():
    st.title('Model Prediction')
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_stock = st.selectbox('Select stock symbol:', list(stock_symbols.items()), format_func=lambda x: x[0])
    stock_symbol = selected_stock[1]
    
    with col2:
        start_date = st.date_input('Select start date:')
        
    with col3:
        end_date = st.date_input('Select end date:')

    df = download_stock_data(stock_symbol, start_date, end_date)
    
    #size = st.text_input('Enter the number of days to predicted')
    # Input for number of days to predict
    size = st.number_input('Enter the number of days to predict', min_value=1, value=7, step=1, format='%d')

        
    p_value = st.text_input('Enter the P')
    q_value = st.text_input('Enter the q')

# Convert input values to integers if they are not empty
    if p_value and q_value and size:
        p_value = int(p_value)
        q_value = int(q_value)
        size =  int(size)
        
    #st.write(size)
    train = df[:int(-size)]
    test = df[int(-size):] 
    
    TRAIN_LEN = len(train)
    HORIZON = len(test)
    WINDOW = 1


#pred_ARIMA = rolling_forecast(df["Close"], TRAIN_LEN, HORIZON, WINDOW,p_value,q_value, 'ARIMA')



   
    if st.button('Predict'):
        with st.spinner('Fitting model...'):
            pred_ARIMA = rolling_forecast(df["Close"], TRAIN_LEN, HORIZON, WINDOW,p_value,q_value, 'ARIMA')

        
            test.loc[:, 'pred_ARIMA'] = pred_ARIMA
            
            #st.write(test.head())
            st.success('Model fitted successfully!')
        
            st.write(test.tail())
            key_ = get_key(stock_symbols, stock_symbol)
            # Plot original data and forecasted values
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['Close'][TRAIN_LEN:], label='Original Data')
            ax.plot(test['pred_ARIMA'], 'k--', label='Pedicted')
            ax.set_title(f'Prediction for {key_}')
            #ax.set_title(f'Forecast for {stock_symbol}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USD)')
            plt.xticks(rotation=30) 
            ax.legend()
            st.pyplot(fig)

    
   

   
        

        
            
if __name__ == "__main__":
    main()
