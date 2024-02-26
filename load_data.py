import pandas as pd
import json
import os
import time
from tqdm import tqdm
import time
from datetime import datetime

# https://stackoverflow.com/questions/33061302/dictionary-of-pandas-dataframe-to-json


# read from disk
root = "Data"


def load_data(path="stock_info.json"):
    """
    loads dictionary of dataframes representing stock history of all the stocks
    """
    with open(os.path.join(root, path), "r") as fp:
        data_dict = json.load(fp)

    # convert dictionaries into dataframes
    data = {key: pd.DataFrame(eval(data_dict[key])) for key in tqdm(data_dict)}

    for key in tqdm(data_dict):
        data[key] = pd.DataFrame(eval(data_dict[key]))
        data[key].index = data[key].index.map(
            lambda x: datetime.fromtimestamp(int(x) / 1e3)
        )
        print(data[key].index)

    return data


if __name__ == "__main__":
    start_time = time.time()

    data = load_data()

    print(f"{time.time()-start_time} seconds.")
