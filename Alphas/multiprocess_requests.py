import multiprocessing
import os
import functools

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_pickle.html
# https://pandas.pydata.org/docs/dev/reference/api/pandas.DataFrame.to_pickle.html
# https://stackoverflow.com/questions/75502567/how-to-use-multiprocessing-pool-for-pandas-apply-function
# https://pypi.org/project/pandarallel/#description
# https://stackoverflow.com/questions/74421948/return-dataframe-variable-on-multiprocessing


# https://stackoverflow.com/questions/72385535/how-to-generate-a-dataframe-from-the-results-of-a-concurrent-futures-processpool
# https://www.tjansson.dk/2018/04/parallel-processing-pandas-dataframes/


def concurrency(func, cores=multiprocessing.cpu_count()):
    @functools.wraps(func)
    # this just adds the documentation via .__doc__ but it makes me sound smart
    # so I'm keeping it
    def wrapper(*args, **kwargs):
        with multiprocessing.Pool(processes=cores) as pool:
            try:
                # map to an array of dfs?
                results = pool.map(func, *args, **kwargs)
            except Exception as e:
                print(f"Error: {e}")

        return results

    return wrapper


def concurrency(func, objects: list, cores=multiprocessing.cpu_count()):
    with multiprocessing.Pool(processes=cores) as pool:
        try:
            # map to an array of dfs?
            results = pool.map(func, objects)
        except Exception as e:
            print(f"Error: {e}")

    return results
