"""MLT: Utility code."""

import os
import pandas as pd


def symbol_to_path(symbol, base_dir=os.path.join(".", "history/day/data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates, addSPY=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'sh000001' not in symbols:  # add SPY for reference, if absent
        symbols = ['sh000001'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol[2:]), index_col='date',
                parse_dates=True, usecols=['date', 'close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'close': symbol})
        df = df.join(df_temp)
        if symbol == 'sh000001':  # drop dates SPY did not trade
            df = df.dropna(subset=["sh000001"])
    return df

def get_stock(symbol, dates):
    """Read stock data (open, high, low, close, vol) for given symbols from CSV files.
       Index on all market days
       A value of NaN is assigned for the days when that stock does not trade
    """
    symbol_index = 'sh000001'
    # Get a DafaFrame for sh000001
    df = pd.DataFrame(index=dates)
    df_temp = pd.read_csv(symbol_to_path(symbol_index[2:]), index_col='date', parse_dates=True, usecols=['date', 'close'], na_values=['nan'])
    df = df.join(df_temp)
    df = df.rename(columns={'close': symbol_index})
    # Get a target DataFrame and join with sh000001
    df_temp = pd.read_csv(symbol_to_path(symbol[2:]), index_col='date', parse_dates=True, usecols=['date', 'open', 'high', 'low', 'close', 'volume'], na_values=['nan'])
    df = df_temp.join(df)
    df = df.dropna(subset=[symbol_index])
    return df

def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""
    df_data.fillna(method='ffill', inplace=True)
    df_data.fillna(method='bfill', inplace=True)

def compute_euclidean_distance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += (instance1[i] - instance2[i])**2
    return distance

# Test case for this python file
if __name__=="__main__":
    print("Command line execution not implemented")
