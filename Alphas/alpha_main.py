"""
Make 1 dataframe per alpha. Each column is represented by a stock, with rows representing normalized time.
    - For us to properly rank, we need to accept a parameter with list of stocks to map this function to and then rank

Store each dataframe/alpha in a hashmap. Key,value = alpha#, dataframe

Also, store key metrics for each companies in their own company-wide dataframes (we can also store and load from json)
"""

from data_load import load_data

stock_data_dict = load_data()

alpha_dict = load_data("alpha_src.json")
