from sklearn.preprocessing import StandardScaler, MinMaxScaler
import all_alphas
import yfinance as yf
import pandas as pd

def normalize_alphas(x):
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit_transform(x.to_frame())
    return x


def standardize_alphas(x):
    scaler = StandardScaler()
    scaler.fit_transform(x.to_frame())
    return x

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
    print(alpha)


if __name__ == "__main__":
    main()
