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

def save_dataframe_to_csv(df, file_name):
    df.to_csv(file_name)

#create y dataframe for neural network
#holding period is going to be 8 weeks 
#weekends account for 8 of those days and assuming 1 holiday day 
#56 - 16 - 1 = 39 trading days
def create_y(data):
    closed_shifted = data['Close'].shift(-39) #shifted up 39 days to get the closing price 39 days from now
    closed_shifted = closed_shifted[:-39]
    closed = data['Close'][:-39] #making sure that both columns of data are the same length 
    returns = closed_shifted / closed 
    returns.index = closed.index 
    # min_date = returns.index.min()
    # max_date = returns.index.max()
    # data_trimmed = data.loc[min_date:max_date]
    returns.fillna(method='ffill', inplace=True)
    returns.fillna(0, inplace=True)

    return returns

    

def main():
    file = 'SPY_data_2022_2024.csv'
    data = csv_to_dataframe(file)

    alphas_df = csv_to_dataframe('alphas_SPY_2022_2024.csv')

    y = create_y(data, alphas_df)
    print(y)
    #save_dataframe_to_csv(y, 'y_SPY_2022_2024.csv')
    
if __name__ == "__main__":
    main()