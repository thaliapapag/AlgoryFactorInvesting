import pandas as pd
import yfinance as yf

def save_data(tickerSymbol, start_date, end_date, file_name):
    tickerData = yf.Ticker(tickerSymbol)
    tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)
    tickerDf.to_csv(file_name) #save data to csv file

def csv_to_dataframe(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df



#create y dataframe 
def create_y(data, days_ago=2):
    buy_date = data['Close'].iloc[-1]
    returns_df = pd.DataFrame({
        'returns_x_days_ago': (data['Close'] - data['Close'].shift(days_ago)) / data['Close'].shift(days_ago)
    }, index=data.index)
    returns_df.dropna(subset=['returns_x_days_ago'], inplace=True)
    return returns_df


def main():
    file = 'SPY_data_2022_2024.csv'
    data = csv_to_dataframe(file)
    y = create_y(data)
    print(data)
    print(y)

if __name__ == "__main__":
    main()