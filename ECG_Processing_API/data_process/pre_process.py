import pandas as pd

def series_to_string(ecg_series):
    return ecg_series.to_string(index=False)


def string_to_series(ecg_string):
    return pd.Series(ecg_string.split('\n'))
