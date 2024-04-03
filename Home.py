import streamlit as st 

st.set_page_config(page_title="Stock Price Prediction",
                   layout='wide',
                   page_icon='./images/home.png')

st.title("Stock Price Prediction App")
st.caption('This web application predicts stock prices based on historical data.')

# Content
st.markdown("""
### Welcome to the Stock Price Prediction App!

This app predicts the future stock prices of a selected company using historical data.
You can select a company from the dropdown menu and adjust the parameters accordingly.

Below are some of the companies available in the dropdown menu:

1. Google
2. Apple
3. Microsoft
4. Amazon
5. Facebook
6. Tesla
7. Alphabet
8. Netflix
9. Nvidia
10. Adobe
11. Intel
12. PayPal
13. Johnson & Johnson
14. Visa
15. JPMorgan Chase
16. Walmart
17. Procter & Gamble
18. Bank of America
19. Mastercard
20. Verizon
21. Coca-Cola
22. Disney

Feel free to explore and predict stock prices for different companies!
""")
