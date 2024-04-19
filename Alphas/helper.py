import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from datetime import datetime
import random
from tqdm import tqdm


def adv(hst, d):
    # average daily dollar volume for the past d days
    print(hst)
    vol = hst["Volume"]
    rolling_mean = vol.rolling(window=d).mean()
    hst["adv"] = rolling_mean
    return hst["adv"]


def vwap(hst):  # yf.Ticker.history object
    # daily volume-weighted average price
    vol = hst["Volume"]
    close = hst["Close"]

    sum_vol = vol.sum()

    hst["vwap"] = (vol * close).div(sum_vol)

    return hst["vwap"]


def ts_rank():
    pass


def decay_linear(hst, d):
    weights = np.array([i for i in range(d, 0, -1)])
    sum_weights = np.sum(weights)
    # print(hst, hst.shape)

    # This is some spaghetti
    weighted_ma = hst.rolling(window=d, center=True, min_periods=1).apply(
        lambda x: np.divide(np.sum(weights[: len(x)] * x), sum_weights), raw=False
    )

    # Discard inaccurate values
    weighted_ma.shift(d)

    print(weighted_ma, "INDEX", weighted_ma.index, "VALUES", weighted_ma.values)

    return weighted_ma


def correlation_days(x, y, d):
    # returns a scalar
    df = pd.DataFrame
    x_new = x.tail(d).values()
    y_new = y.tail(d).values()
    corr = np.corrcoef(x_new, y_new)

    return corr[0][1]  # return value from correlation matrix


def slice_database_by_dates(database, start_date, end_date=None):
    # Does not have error handling. Must provide valid business days for both start_date and end_date
    if start_date not in database.index:
        raise ValueError(
            f"Could not find start date. {start_date}. Database index range: {database.index[0],database.index[-1]}"
        )

    if not end_date:
        return database.iloc[start_index:]

    if end_date not in database.index:
        raise ValueError(
            f"Could not find end date. {end_date}. Database index range: {database.index[0],database.index[-1]}"
        )

    start_index = np.where(database.index == start_date)[0][0]
    end_index = np.where(database.index == end_date)[0][0]

    database = database.iloc[start_index : end_index + 1]
    return database


def get_market_start_date(days_ago=50, end_date=datetime.now(), return_type="str"):
    # assumes that each stock is listed on either NYSE or Nasdaq, which follow the same schedule
    nyse = mcal.get_calendar("NYSE")
    date = pd.to_datetime(
        end_date.strftime("%Y-%m-%d")
    ) - pd.tseries.offsets.CustomBusinessDay(
        days_ago, holidays=nyse.holidays().holidays
    )
    if return_type == "str":
        date = date.strftime("%Y-%m-%d")
        # "2002-01-01"
        # Format: year-month-day
    return date


def get_market_end_date(end_date: str, change_days=1, return_type="str"):
    # assumes that each stock is listed on either NYSE or Nasdaq, which follow the same schedule
    nyse = mcal.get_calendar("NYSE")
    date = pd.to_datetime(end_date) + pd.tseries.offsets.CustomBusinessDay(
        change_days, holidays=nyse.holidays().holidays
    )
    if return_type == "str":
        date = date.strftime("%Y-%m-%d")
        # "2002-01-01"
        # Format: year-month-day
    return date


def generate_random_instructions(n: int = 1000) -> list:
    # generating random instructions to benchmark backtest efficiency
    res = []
    symbols = ["AAPL", "MSFT", "SAVE", "TER", "GME", "AMGN"]
    order_types = ["BUY", "HOLD", "SELL"]
    quantity = lambda x: random.randrange(5, 100)
    for _ in tqdm(range(n)):
        instruction = []
        """
        order_type, symbol, quantity
        order_type: BUY, SELL, HOLD
        """
        instruction.append(random.choice(symbols))
        instruction.append(random.choice(order_types))
        instruction.append(quantity(None))

        res.append(instruction)

    return res

def func_string(func,*args):
    
    pass


if __name__ == "__main__":
    a = generate_random_instructions(100)
    print(a)
