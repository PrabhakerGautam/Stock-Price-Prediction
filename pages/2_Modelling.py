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

import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline

# Disable warning about passing a figure to st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)
#set page icon
st.set_page_config(page_icon="./images/object.png")
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

# Function to optimize ARIMA model
def optimize_ARIMA(endog: Union[pd.Series, list], order_list: list, d: int) -> pd.DataFrame:
    results = []
    progress_bar = st.progress(0)
    for i, order in enumerate(order_list):
    
        try:
            model = SARIMAX(endog, order=(order[0], d, order[1]), simple_differencing=False).fit(disp=False)
        except:
            continue
        aic = model.aic
        results.append([order, aic])
        progress_bar.progress((i+1) / len(order_list))

    result_df = pd.DataFrame(results, columns=['(p,q)', 'AIC'])
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return result_df

# Function to perform ADF test for stationarity
def adf_test(series):
    result = sm.tsa.adfuller(series)
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])
    #st.write('Critical Values:')
    #for key, value in result[4].items():
    #    st.write(f'{key}: {value}')
        
    if result[1] < 0.05:
        st.write("All set data is stationary and we have P and Q for forecasting")
        
    else:
        st.write("Not stationary")
        



# Main function for the app


def main():
    st.title('Model Optimization')
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_stock = st.selectbox('Select stock symbol:', list(stock_symbols.items()), format_func=lambda x: x[0])
    stock_symbol = selected_stock[1]
    
    with col2:
        start_date = st.date_input('Select start date:')
        
    with col3:
        end_date = st.date_input('Select end date:')

    df = download_stock_data(stock_symbol, start_date, end_date)
    size = -7
    train = df[:size]
    test = df[size:]  
    p_value, q_value  = 0 , 0
    #with st.spinner:
    if st.button('Optimize'):
        
        ps = range(0, 4, 1)
        qs = range(0, 4, 1)
        d = 2
        order_list = list(product(ps, qs))
        result_df = optimize_ARIMA(train, order_list, d)
        st.write(result_df.head(5))
        # Extracting p and q values from the top row of result_df
        
        p_value = result_df.iloc[0]['(p,q)'][0]
        q_value = result_df.iloc[0]['(p,q)'][1]

        # Printing the extracted values
        st.write(f'Optimized values for P and Q for ARIMA Model are: P={p_value}, Q={q_value}')

        # Perform ADF test for stationarity
        st.subheader('Augmented Dickey-Fuller (ADF) Test for Stationarity')
        df_diff = np.diff(train['Close'], n=1)
        adf_test(df_diff)
         # Set flag to True to indicate optimization is completed
        #optimization_completed = True
        
        
        
    
    if st.button('Fit the model and Plot diagnostics'):
        with st.spinner('Fitting model...'):
            model = SARIMAX(train, order=(p_value,1,q_value), simple_differencing=False)
            model_fit = model.fit(disp=False)
            st.pyplot(model_fit.plot_diagnostics(figsize=(10,8)))
            
            
if __name__ == "__main__":
    main()
