import yfinance as yf
from pandas_datareader import data as pdr
import pandas as pd
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
import multiprocessing as mp
from tqdm import tqdm
import time
import os
import json
import datetime

root = "Alphas/Data"

# https://stackoverflow.com/questions/33061302/dictionary-of-pandas-dataframe-to-json


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # default is the function applied to objects not already json serializable
        if hasattr(obj, "to_json"):
            return obj.to_json(orient="columns")
        return json.JSONEncoder.default(self, obj)


# For S&P 500, we can pull GICS from Wikipedia


def get_spy_close(start_date="2018-1-1", end_date="2023-11-1", csv=True):
    class CachedLimiterSession(
        CacheMixin, LimiterMixin, Session
    ):  # inherits three classes
        pass

    session = CachedLimiterSession(
        limiter=Limiter(
            RequestRate(2, Duration.SECOND * 5)
        ),  # max 2 requests per 5 seconds. Yahoo Finance API seems to rate limit to 2000 requests per hour
        bucket_class=MemoryQueueBucket,
        backend=SQLiteCache(os.path.join(root, "yfinance.cache")),
    )

    tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[
        0
    ]
    print(tickers.head())

    spy_data = yf.download(
        tickers.Symbol.to_list(),
        start=start_date,
        end=end_date,
        period="1d",
        auto_adjust=True,
        session=session,
    )["Close"]
    # Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo Intraday data cannot extend last 60 days
    spy_data.dropna(axis=1, how="any", inplace=True)
    print(start_date, end_date)
    print(spy_data.head())
    print(spy_data.tail())

    if csv:
        spy_data.to_csv(os.path.join(root, "spy.csv"), index=True)

    return spy_data


def stockHistory(ticker: str, start, end, to_json=True):
    stock = yf.Ticker(ticker).history(start=start, end=end)
    # stock.name = ticker
    # print(stock)
    # stock.reset_index(inplace=True)
    # return stock.to_json(orient="columns")
    return stock


def dictdf_to_dict(df):
    """
    Converts dictionary of dataframes into complete dictionary for json
    """
    data_dict = {key: df[key].to_dict(orient="records") for key in df.keys()}

    return data_dict


# https://stackoverflow.com/questions/20776189/concurrent-futures-vs-multiprocessing-in-python-3
def get_spy_data(
    start_date="2018-1-1",
    end_date="2023-11-1",
    tickers: list = [],
    processes=10,
    seconds=10,
) -> dict:
    """
    Elaborate collection of data, returned as a dictionary of pandas DataFrames
    """

    print(f"You have {mp.cpu_count()} cores.")

    tickers = get_spy_tickers()[:10] if not tickers else tickers
    ticker_list = []  # tickers to save to spy_tickers.txt
    # we can parse spy_tickers.txt to pick out the specific stocks we want to test on for alpha calculation

    for ticker in tqdm(tickers):
        stock_info = stockHistory(ticker, start=start_date, end=end_date, to_json=True)

        if stock_info.empty:
            print(f"Ticker {ticker} encountered error. Cutting ticker from stock info.")
            continue

        # dictdf_to_dict(stock_info)
        save_path = f"Stock_History/{ticker}_info.json"
        ticker_list.append(ticker)

        print(f"List of tickers downloaded: {ticker_list}")

        with open(os.path.join(root, save_path), "w") as f:
            json.dump(stock_info, f, indent=4, cls=JSONEncoder)

    with open(os.path.join(root, "spy_tickers.txt"), "w") as f:
        for ticker in ticker_list:
            f.write(ticker + "\n")

    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html
    # I can properly serialize and deserialize but this is very low priority
    # for ticker in tqdm(tickers):
    #     with mp.Pool(processes=processes) as pool:
    #         results = [
    #             pool.apply_async(stockHistory, args=(ticker, start_date, end_date))
    #             for ticker in tickers
    #         ]

    # for result in tqdm(results):
    #     try:
    #         ticker, stock = result.get(seconds)
    #         stock_info[ticker] = stock
    #     except TimeoutError:
    #         print(f"Timeout Error: {ticker}")
    #     except Exception as e:
    #         print(f"Other error: {e}.")

    # for stock in results:
    #     stock_info[stock.name] = stock


def get_spy_tkr_data(start_date, end_date, ticker):
    try:
        stock_info = stockHistory(ticker, start=start_date, end=end_date, to_json=True)

        if stock_info.empty:
            print(f"Ticker {ticker} encountered error. Cutting ticker from stock info.")
            raise Exception("TKR ERROR: yfinance failed to find stock info.")

        # dictdf_to_dict(stock_info)
        save_path = f"Stock_History/{ticker}_info.json"

        with open(os.path.join(root, save_path), "w") as f:
            json.dump(stock_info, f, indent=4, cls=JSONEncoder)

    except Exception as e:
        print("ERROR OCCURED IN GET TKR DATA: ", e)


def get_spy_tickers() -> list:
    tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[
        0
    ]
    return tickers.Symbol.tolist()


def int_to_datetime(datetime_int):
    date = datetime.datetime.fromtimestamp(datetime_int / 1e3)
    return date


def get_spy_index_data(start_date="2018-1-1", end_date="2023-11-1", csv=True):
    spy_data = yf.download(
        "SPY",
        start=start_date,
        end=end_date,
        period="1d",
        auto_adjust=True,
    )["Close"]
    # Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo Intraday data cannot extend last 60 days
    print(start_date, end_date)
    print(spy_data.head())
    print(spy_data.tail())

    if csv:
        spy_data.to_csv(os.path.join(root, "spy_index.csv"), index=True)

    return spy_data


if __name__ == "__main__":
    start_time = time.time()

    # get_spy_close()
    get_spy_data()

    print(f"Took {time.time()-start_time} seconds.")
