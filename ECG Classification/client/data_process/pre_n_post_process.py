import pandas as pd
import json
import numpy as np

def series_to_string(ecg_series):
    return ecg_series.to_string(index=False)


def string_to_series(ecg_string):
    return pd.Series(ecg_string.split('\n'))

def beats_str_to_list(beats_json):
    
    # Convert beats_json string back into a list of lists
    beats_list = json.loads(beats_json)

    # Convert each inner list in beats_list into a numpy array
    new_beats = [np.array(beat) for beat in beats_list]

    return new_beats
