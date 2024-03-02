import pandas as pd
import json
import os
import time
from tqdm import tqdm
import time
from datetime import datetime
from numpy import loadtxt

# https://stackoverflow.com/questions/33061302/dictionary-of-pandas-dataframe-to-json


# read from disk
print(os.getcwd())

cwd_folder = os.getcwd().split("\\")[-1]
root = "Alphas/Data" if cwd_folder == "AlgoryFactorInvesting" else "Data"
print(root)


def load_tickers(root=root, max_tickers: int = 500):
    with open(os.path.join(root, "spy_tickers.txt"), "r") as f:
        tickers = f.read().splitlines()

    tickers = tickers[: min(len(tickers), max_tickers)]
    return tickers


def load_stock_history_data(root=root, tickers: list = None, max_tickers: int = 500):
    """
    loads dictionary of dataframes representing stock history of all the stocks
    Applies to stock_info and alpha main

    @tickers (list): Specify a specific list of tickers
    @max_tickers (int): optional argumet to cap number of tickers being retrieved
    """
    if not tickers:
        with open(os.path.join(root, "spy_tickers.txt"), "r") as f:
            tickers = f.read().splitlines()

    print(tickers)
    tickers = tickers[: min(len(tickers), max_tickers)]
    data = {}
    for ticker in tqdm(tickers):
        try:
            with open(
                os.path.join(root, f"Stock_History/{ticker}_info.json"), "r"
            ) as fp:
                data_dict = json.load(fp)

            data[ticker] = pd.DataFrame(eval(data_dict))
            data[ticker].index = data[ticker].index.map(
                lambda x: datetime.fromtimestamp(int(x) / 1e3)
            )
        except Exception as e:
            print(f"Error occurred in fetching data for {ticker}: {e}.")

    return data


def load_alpha_helper_data(root=root, path="alpha_src.json"):
    """
    Loads dictionary of dictionary (helper alphas) of DataFrames (all Head: stock. Index: time)
    The loading time for this will be awful. Maybe can optimize with multiprocessing
    """
    data = {}
    with open(os.path.join(root, path), "r") as fp:
        data_dict = json.load(fp)

    for alpha_key in tqdm(data_dict.keys()):
        for key in data_dict[alpha_key]:
            data[alpha_key][key] = {pd.DataFrame(eval(data_dict[key]))}
            data[alpha_key][key].index = data[alpha_key][key].index.map(
                lambda x: datetime.fromtimestamp(int(x) / 1e3)
            )

    return data


if __name__ == "__main__":
    start_time = time.time()

    data = load_stock_history_data()
    print(data)

    print(f"{time.time()-start_time} seconds.")
