from market_tester import (
    run_timeline,
    print_results,
    portfolio_history,
    portfolio_value,
    blockPrint,
    enablePrint,
)
from factor_model import orders, cfg, process_xy
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
from typing import List, Union
from config import random_thresholds
from tqdm import tqdm
import csv


root = "Alphas"


def normalize_data(series: pd.Series) -> pd.Series:
    return (series / series.iloc[0] - 1) * 100


def plot_all(data: pd.Series = portfolio_history, *args: List[Union[pd.Series, str]]):
    """
    @port_hist: our strategy's portfolio history
    @*args (list[pd.Series,str]): Other strategy history, with strategy label
    """
    try:
        portfolio_history.name = "Factor Investing"
        portfolio_history.index = orders.index
        fig, ax = plt.subplots()
        data = normalize_data(data).to_frame()
        for series, label in args:
            if isinstance(series, pd.DataFrame):
                if len(series.columns) > 1:
                    raise InvalidPlotArgument
                series = series.iloc[:, 0]
            normalized_series = normalize_data(series)
            print("NORM SERIES:\n", normalized_series)
            series.name = label
            data = data.merge(
                series,
                how="left",
                left_index=True,
                right_index=True,
            )

            data[label] = normalize_data(data[label])

        print("\n\nDATA: \n", data)

        ax.set_xlabel("Time")
        ax.set_ylabel("Percentage Returns")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_title(f"Factor Investing: {start_date} - {end_date}")

        for col in data.columns:
            ax.plot(data[col], label=col)

        every_nth = 250
        for n, (label, tick, tickline) in enumerate(
            zip(
                ax.xaxis.get_ticklabels(),
                ax.xaxis.get_major_ticks(),
                ax.xaxis.get_ticklines(),
            )
        ):
            if n % every_nth != 0:
                label.set_visible(False)
                tick.set_visible(False)
                tickline.set_visible(False)

        ax.legend()
        fig.tight_layout()

        ax.tick_params(axis="x", labelrotation=10, labelsize=7)

        fig.savefig(
            os.path.join(root, "Backtests/Images", f"{start_date}_{end_date}.png")
        )

        fig.show()
    except InvalidPlotArgument as e:
        print("Error occured with Invalid Plot Argument:", e)
    except Exception as e:
        print("Generic exception occured in plotting: ", e)


class InvalidPlotArgument(Exception):
    def __init__(self):
        self.message = "Multiple column dataframe provided in plot_all."
        super().__init__(self.message)


def write_to_backtest_csv(result, t_buy, t_sell, csv_path="Backtests/log.txt"):
    line = {"t_buy": t_buy, "t_sell": t_sell, "RETURN": result}

    with open(os.path.join(root, csv_path), "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=line.keys())
        writer.writerow(line)


def sort_csv(csv_path="Backtests/log.txt"):
    df = pd.read_csv(os.path.join(root, csv_path))
    print(df, df.columns)
    df.sort_values(by=["RETURN"], ascending=False, inplace=True)

    df.to_csv(os.path.join(root, csv_path), index=False)


def random_trials(trials=100, blockPrints=True):
    if blockPrints:
        blockPrint()
    for _ in tqdm(range(trials)):
        t_sell, t_buy = random_thresholds()
        # Modify cfg object directly later instead of this spaghetti
        run_iteration(t_buy=t_buy, t_sell=t_sell)


def run_iteration(t_buy, t_sell):
    orders = process_xy(t_buy, t_sell)
    orders = orders[:500]
    start_date = orders.index[0]
    end_date = orders.index[-1]

    run_timeline(orders, start_date, end_date)
    print_results()
    write_to_backtest_csv(portfolio_value(False, orders.index[-1]), t_buy, t_sell)


if __name__ == "__main__":
    sort_csv()
    start_time = time.time()

    settings = cfg.exog
    spy_data = pd.read_csv(os.path.join(root, "Data/spy_index.csv")).set_index("Date")

    orders = process_xy(1.2814, 1.245)
    orders = orders[:500]
    start_date = orders.index[0]
    end_date = orders.index[-1]

    # random_trials(1000)
    # run_iteration(t_buy=1.2814, t_sell=1.245)

    # spy_data = pd.read_csv(os.path.join(root, "Data/spy_index.csv")).set_index("Date")

    # portfolio_history.index = orders.index
    sort_csv()
    plot_all(portfolio_history, [spy_data, "SPY"])

    print(f"Done. Took {time.time()-start_time:.2f} seconds.")
