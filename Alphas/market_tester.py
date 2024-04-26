import cProfile

import time
import yfinance as yf
import csv
import datetime
import strategy
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from helper import (
    slice_database_by_dates,
    get_market_end_date,
)
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from data_download import get_spy_data, get_spy_index_data
from config import cfg

# from helper import get_market_start_date

settings = cfg.broker_opt
root = "Alphas"

# Linear list of trades made
trades_made = []

# Current positions organized by ticker
positions = {}

# Tracks days held, updates based on sales, and auto sells at certain date
daytracker = defaultdict(list)

current_capital = cfg.INITIAL_CAPITAL

day_number = 0

orders_index = []
orders = pd.DataFrame()

database = pd.read_csv(os.path.join(root, "Data/spy.csv"), index_col="Date")
database.index = database.index.map(lambda x: x.split(" ")[0])

portfolio_history = pd.Series()


class Trade:
    def __init__(self, trade_id, symbol, quantity, price, timestamp, type):
        self.trade_id = trade_id
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp
        self.type = type


def calculate_commission(trade):
    if settings.get("C_TYPE") == "PERCENT":
        return trade.price * trade.quantity * settings["C_PERCENT"]
    elif settings.get("C_TYPE") == "FLAT":
        return settings["C_FLAT"]
    elif settings.get("C_TYPE") == "NONE":
        return 0
    else:
        raise TypeError("No valid commision")


# calculates value for positions TO NOW
def portfolio_value(latest=True, date: str = ""):
    total_value = current_capital
    if latest:
        for symbol, position in positions.items():
            stock = yf.Ticker(symbol)
            current_price = stock.history(period="1d")["Close"].iloc[-1]

            # Calculate the current value of the position
            position_value = position["quantity"] * current_price
            total_value += position_value
    else:
        for symbol, position in positions.items():
            current_price = database.loc[date][symbol]

            position_value = position["quantity"] * current_price
            total_value += position_value

    return total_value


def buy_stock(symbol, quantity, price, timestamp: str):  # 2018-06-20
    global current_capital, day_number  # Declare global to update the outer variable
    new_trade = Trade(
        len(trades_made),
        symbol=symbol,
        quantity=quantity,
        price=price,
        timestamp=timestamp,
        type="buy",
    )

    if quantity == 0:
        # since we are portion-sizing based on USD, there is a chance we get quantity of zero
        # avoid division by zero
        return

    if current_capital >= (quantity * price + calculate_commission(new_trade)):
        trades_made.append(new_trade)

        # Calculate the total cost of the purchase, including commission
        total_cost = (quantity * price) + calculate_commission(new_trade)

        if symbol in positions:
            positions[symbol]["quantity"] += quantity
            positions[symbol]["avg_price"] = (
                positions[symbol]["avg_price"] * positions[symbol]["quantity"]
                + quantity * price
            ) / (positions[symbol]["quantity"] + quantity)
        else:
            positions[symbol] = {"quantity": quantity, "avg_price": price}

        # daytracker = add_to_daytracker(daytracker, quantity, symbol, day_number, price)

        current_capital -= total_cost
    else:
        print("Not enough capital to buy", quantity, "shares of", symbol)


def sell_stock(symbol, quantity, price, timestamp):
    new_trade = Trade(
        len(trades_made),
        symbol=symbol,
        quantity=quantity,
        price=price,
        timestamp=timestamp,
        type="sell",
    )
    global current_capital  # Declare global to update the outer variable

    if (
        new_trade.symbol in positions
        and positions[new_trade.symbol]["quantity"] >= quantity
    ):
        # Calculate the total revenue from the sale, after deducting commission
        total_revenue = (quantity * new_trade.price) - calculate_commission(new_trade)

        # Update position
        positions[new_trade.symbol]["quantity"] -= quantity
        # daytracker = remove_from_daytracker(symbol, quantity, daytracker)

        # Check if all shares are sold for this position
        if positions[new_trade.symbol]["quantity"] == 0:
            del positions[new_trade.symbol]

        # Update current capital
        current_capital += total_revenue

    else:
        print("Not enough shares to sell or invalid trade.", new_trade)


# STRUCTURE SHOULD BE [[BUY, symbol, quantity], [SELL, symbol, quantity]]


def run_daily_instructions(current_day: str, instructions=list[list]):
    # print(instructions, type(instructions))
    # TODO: Add concurrency
    try:
        for order in instructions:
            order_type, symbol, quantity = order

            price = float(database.loc[current_day][symbol])

            if order_type == "BUY":
                buy_stock(symbol, quantity, price, current_day)
            elif order_type == "SELL":
                sell_stock(symbol, quantity, price, current_day)
            else:
                print(f"Invalid order{order}")
                return

        # print(f"{order_type: <4} {symbol: <4} on {current_day}: {float(price):.2f}")
    except Exception as e:
        print(e)
        return


def run_timeline(orders: pd.DataFrame, start_date: str, end_date: str):
    """
    Runs all instructions from a dataframe
    """
    format = cfg.fmt
    start_date = datetime.datetime.strptime(start_date, format)
    end_date = datetime.datetime.strptime(end_date, format)
    to_sell = []
    global day_number, daytracker, portfolio_history

    day_number = 0

    # Set the 'Date' column as the index
    # database.set_index("Date", inplace=True)
    # database.sort_index(inplace=True)

    # TODO: add concurrency
    for current_date in tqdm(orders.index):  # fix
        # Check if the date exists in the index
        # try:
        day_number += 1
        data_rows = orders.loc[current_date]
        instructions = data_rows

        run_daily_instructions(current_date, instructions.values.tolist())

        # update daytracker. Will move into function if this works

        portfolio_history = save_portfolio_value(portfolio_history, current_date)

    return None


def save_portfolio_value(series: pd.Series, current_date: str, latest=False):
    # global portfolio_history
    series.at[len(series)] = portfolio_value(latest, current_date)
    return series


def print_results():
    print(f"Current Capital: {float(current_capital):.2f}")
    print(f"Current Portfolio Value: {float(portfolio_value()):.2f}")
    print("Positions:", positions)


# does not account for stock splits
# database currently contains stocks from current s&p 500, if stocks leave/rejoin it gets weird
if __name__ == "__main__":
    portfolio_history = pd.Series()
    start_time = time.time()
    start_date = "2018-06-12"
    end_date = "2022-01-03"

    get_spy_data(
        start_date=start_date, end_date=get_market_end_date(end_date=end_date)
    )  # We need to call market_end_date since yfinance doesn't download the last day
    # https://github.com/ranaroussi/yfinance/issues/1445

    root = "StatArb"
    csv_path = "spy.csv"
    database = pd.read_csv(os.path.join(root, csv_path))
    database_index = list(map(lambda x: x.split(" ")[0], database["Date"].tolist()))
    database["Date"] = database_index
    database.set_index("Date", inplace=True)

    database = slice_database_by_dates(
        database=database, start_date=start_date, end_date=end_date
    )

    # print(database, database.head(), database.tail(), len(database))
    # database.sort_index(inplace=True)

    run_timeline(orders, orders_index[0], orders_index[-1])
    print(portfolio_history)

    print(f"Current Capital: {float(current_capital):.2f}")
    print(f"Current Portfolio Value: {float(portfolio_value()):.2f}")
    print("Positions:", positions)
    # print(f"Day tracker: ", daytracker)

    time_diff = time.time() - start_time

    print(f"Done. Took {time_diff:.2f} seconds.")
