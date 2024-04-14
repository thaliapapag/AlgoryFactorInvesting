import pandas as pd
import yfinance as yf
import normalize_alphas

def save_data(tickerSymbol, start_date, end_date, file_name):
    tickerData = yf.Ticker(tickerSymbol)
    tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)
    tickerDf.to_csv(file_name) #save data to csv file

def csv_to_dataframe(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

#create y dataframe for neural network
def create_y(data, alphas_df, days_ago=2):
    earliest_date = alphas_df.index.min()
    latest_date = alphas_df.index.max()
    data_trimmed = data.loc[earliest_date:latest_date]
    buy_date = data_trimmed['Close'].iloc[-1]
    returns_df = pd.DataFrame({
        'returns_x_days_ago': (data_trimmed['Close'] - data_trimmed['Close'].shift(days_ago)) / data_trimmed['Close'].shift(days_ago)
    }, index=data_trimmed.index)
    returns_df.dropna(subset=['returns_x_days_ago'], inplace=True)
    
    return returns_df

def main():
    file = 'SPY_data_2022_2024.csv'
    data = csv_to_dataframe(file)
    
if __name__ == "__main__":
    main()