from market_tester import run_timeline, print_results, portfolio_history
from factor_model import orders, cfg
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
from typing import List, Union


root = "Alphas"


def normalize_data(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: (x / (series.values[0]) - 1) * 100)


def plot_all(data: pd.Series = portfolio_history, *args: List[Union[pd.Series, str]]):
    """
    @port_hist: our strategy's portfolio history
    @*args (list[pd.Series,str]): Other strategy history, with strategy label
    """
    try:
        portfolio_history.name = "Factor Investing"
        print(portfolio_history.index, portfolio_history)
        fig, ax = plt.subplots()
        data = normalize_data(data).to_frame()
        for series, label in args:
            if type(series) == pd.DataFrame:
                if len(series.columns) > 1:
                    raise InvalidPlotArgument
                series = series.iloc[:, 0]
            print(series)
            series.name = label
            data = data.merge(
                normalize_data(series), how="left", left_index=True, right_index=True
            )

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


if __name__ == "__main__":
    start_time = time.time()

    settings = cfg.exog
    start_date = orders.index[0]
    end_date = orders.index[-1]

    # print(orders, len(orders), orders.values.tolist(), len(orders.values.tolist()))
    print(orders.index)
    run_timeline(orders, start_date, end_date)
    print_results()

    spy_data = pd.read_csv(os.path.join(root, "Data/spy_index.csv")).set_index("Date")
    print(type(spy_data))
    print(spy_data)

    portfolio_history.index = orders.index
    plot_all(portfolio_history, [spy_data, "SPY"])

    print(f"Done. Took {time.time()-start_time:.2f} seconds.")
