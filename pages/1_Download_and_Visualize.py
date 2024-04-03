
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.gofplots import qqplot


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline

# Disable warning about passing a figure to st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)
# Set page icon
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
@st.cache_data
def load_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# Function to visualize data
def visualize_data(data,key_):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Close'], label='Close Price')
    ax.set_title(f'{key_} Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    plt.grid(True)
    ax.legend()
    st.pyplot(fig)
    
    from statsmodels.tsa.seasonal import STL
    decomposition = STL(data['Close'], period=12).fit()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True,  figsize=(10,8))
    ax1.plot(decomposition.observed)
    ax1.set_ylabel('Observed')
    ax2.plot(decomposition.trend)
    ax2.set_ylabel('Trend')
    ax3.plot(decomposition.seasonal)
    ax3.set_ylabel('Seasonal')
    ax4.plot(decomposition.resid)
    ax4.set_ylabel('Residuals')
    #plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
    fig.autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig)




# Function for forecasting
def get_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # If the value is not found

# Main function to create the app
def main():
    st.title('Download and Visualize')
    
    
    col1,col2,col3 = st.columns(3)

   
    #st.title('Download and Visualize Data')
    with col1 :
        selected_stock = st.selectbox('Select stock symbol:', list(stock_symbols.items()), format_func=lambda x: x[0])
    stock_symbol = selected_stock[1]
    key_ = get_key(stock_symbols, stock_symbol)
    #stock_symbol = st.sidebar.text_input('Enter stock symbol (e.g., AAPL for Apple):')
    with col2:
        start_date = st.date_input('Select start date:')
    with col3:
        end_date = st.date_input('Select end date:')
        
    if st.button('Download Data'):
        data = load_data(stock_symbol, start_date, end_date)
        visualize_data(data,key_)

    
        
        
            

    
        
    
        
if __name__ == "__main__":
    main()
