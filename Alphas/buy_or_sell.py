from sklearn.preprocessing import StandardScaler, MinMaxScaler
import all_alphas
from normalize_alphas import normalize_alphas, standardize_alphas
import yfinance as yf
import pandas as pd

#input is a normalized alpha of stock, num data pts, criteria for buy or -1*sell
def buy_or_sell(x, v, criteria):
    length = 0

    if len(x) < v:
        length = len(x)
    else:
        length = v

    x_compare = x.tail(length)

    avg = x_compare.mean() #maybe want to switch to median?

    print(x_compare)
    print('****************')
    print(avg)
    

    if(avg > criteria):
        return 'BUY'
    elif(avg < -1*criteria):
        return 'SELL'
    else:
        return 'HOLD'


def main():
    ticker = yf.Ticker("^GSPC")
    data = ticker.history(period="2mo") #pricing data for the S&P 500

    low = data['Low']
    volume = data['Volume'] #THIS REFERS TO VOLUME NOT PRICE VOLUME- should it be changed????  
    close = data['Close']
    high = data['High']
    open = data['Open']
    
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    # Calculate the volume-weighted average price
    vwap = (typical_price * data['Volume']).sum() / data['Volume']

    alpha = all_alphas.alpha54(low, close, open, high)

    alpha = normalize_alphas(alpha)
    alpha = standardize_alphas(alpha)
    
    print(buy_or_sell(alpha, 4, 0.5))


if __name__ == "__main__":
    main()
