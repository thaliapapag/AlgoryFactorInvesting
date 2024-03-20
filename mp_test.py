import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
import time
import random
from tqdm import tqdm


### Config ###
num_rows = 10
num_cols = 10
random.seed(10)
##############


def generate_df(seed=10, num_rows=10, num_cols=10):
    random.seed(seed)

    labels = [chr(ord("a") + i) for i in range(num_rows)]
    cols = [[random.random() for _ in range(num_rows)] for _ in range(num_cols)]

    data = {k: v for k, v in zip(labels, cols)}
    df = pd.DataFrame(data=data)

    # print(df)
    return df


def mystery(df):
    df.apply(lambda x: x**2)
    df.apply(lambda x: x**2)
    df.apply(lambda x: x**2)
    df.apply(lambda x: x**2)
    return df


f = mystery

if __name__ == "__main__":
    print(f"CPU COUNT: {mp.cpu_count()}")

    items = 1000
    objects = []
    for i in range(items):
        objects.append(generate_df(i + 5))

    # print(objects)

    start_time = time.time()

    # imap unordered is ridiculously fast compared to normal map.
    # We can store tuple and recombine into dictionary. I don't know if this is faster or not

    with Pool(20) as p:
        r = list(tqdm(p.imap(f, objects), total=items))

    # print(objects)
    p

    print(f"Finished multiprocessing. Took {time.time()-start_time} seconds.")

    objects = []
    for i in range(items):
        objects.append(generate_df(i + 5))

    # print(objects)
    start_time = time.time()

    for i, j in tqdm(enumerate(objects), total=items):
        objects[i] = f(j)

    # print(objects)

    print(f"Finished standard. Took {time.time()-start_time} seconds.")
