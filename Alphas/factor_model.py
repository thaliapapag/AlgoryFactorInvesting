import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import inspect
import json

"""
To-Do:
Slice-dates (generate orders in range of specific dates, preferably by dates->index)
Order generation
Sync with alpha functions (alpha parameters need to be standardized for this to work cleanly)

"""

print(os.getcwd())

### CONFIG ###
THRESHOLD_BUY = 0.3
THRESHOLD_SELL = 0.1
ORDER_SIZE = 10  # can also do floor div of $ amt by Price/Share
root = "Alphas"

### CONFIG ###

start_time = time.time()
# data = pd.read_csv(os.path.join(root, "Data/spy.csv"))
# if we want to backtest on all tickers we can push data.columns
tickers = np.loadtxt(os.path.join(root, "Data/spy_tickers.txt"), dtype="str")
stock_data = []
weights = {}  # key,value : alpha (function name),weight. Ex: alpha1 : 0.58
day_range = 100  # tmp spaghetti
orders = [] * day_range


for ticker in tqdm(tickers):
    with open(os.path.join(root, f"Data/Stock_History/{ticker}_info.json")) as f:
        data = json.loads(json.load(f))

    data = pd.DataFrame.from_dict(data)
    stock_data.append(data.iloc[-100:])

# sliced_data = data.iloc[-50:].to_numpy()

# use np.vectorize for efficient element-wise operations
combine = (
    lambda func, data, w: np.vectorize(func)(data) * w
)  # assumes that func only has data as arg (or as *args)


# order_type, symbol, quantity = order
def convert_to_order(num, ticker):
    if num > THRESHOLD_BUY:
        return [ticker, ORDER_SIZE, "BUY"]
    elif num < THRESHOLD_SELL:
        return [ticker, ORDER_SIZE, "SELL"]

    return [ticker, ORDER_SIZE, "HOLD"]


for ticker, data in tqdm(zip(tickers, stock_data)):
    res = np.sum(combine(func, data, w) for func, w in weights.items())
    for i, j in enumerate(res):
        orders[i].append(convert_to_order(j))


# https://python-course.eu/numerical-programming/linear-combinations-in-python.php


# Generate orders
