from market_tester import run_timeline, print_results, portfolio_history
from factor_model import orders, cfg
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

root = "Alphas"


def normalize_data(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: (x / (series.values[0]) - 1) * 100)


def plot_all(data: pd.Series = portfolio_history, *args: pd.Series):
    """
    @port_hist: our strategy's portfolio history
    @*args (list[pd.Series,str]): Other strategy history, with strategy label
    """
    portfolio_history.name = "Factor Investing"

    fig, ax1 = plt.subplots()
    data = normalize_data(data).to_frame()
    for series in args:
        data = data.merge(
            normalize_data(series),
            how="left",
        )

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Percentage Returns")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_title(f"Factor Investing: {start_date} - {end_date}")

    for col in data.columns:
        ax1.plot(data[col], label=col)

    every_nth = 200
    for n, label in enumerate(ax1.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    ax1.legend()
    fig.tight_layout()

    fig.savefig(os.path.join(root, "Backtests/Images", f"{start_date}_{end_date}.png"))

    fig.show()


if __name__ == "__main__":
    start_time = time.time()

    settings = cfg.exog
    start_date = orders.index[0]
    end_date = orders.index[-1]

    print(orders, len(orders), orders.values.tolist(), len(orders.values.tolist()))
    run_timeline(orders, start_date, end_date)
    print_results()
    plot_all(portfolio_history)

    print(f"Done. Took {time.time()-start_time:.2f} seconds.")
