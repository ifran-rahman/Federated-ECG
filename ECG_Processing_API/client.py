## This script deals with posting and getting data from the API
import requests
from data_process.dataProcessor import signal_to_beats
import numpy as np
import pandas as pd 
from data_process.pre_n_post_process import beats_str_to_list

def load_signal():
  # read a sample signal 
  df = pd.read_csv('archive/100.csv')
  signal = df['\'MLII\''][0:600]
  
  return signal

# load signal
signal = load_signal()
signal_data = signal.values.tolist()

# POST ECG signal to ECG-Process API
url = "http://127.0.0.1:8000/signals/"

# The data to post
data = {
  "signal_data": signal_data,
   "is_verified": 0
}

# POST signal
API_response = requests.post(url, json = data)

# GET heartbeats
json_data = API_response.json()
beats_json = json_data['beats']

# GET beats and convert beats_json to list
beats = beats_str_to_list(beats_json)

# get annotation for the current signal infereing the existing model
# we assume the annotation for the current signal is 1 (Abnormal)
annotation = 1

# choose dataset path according to annotation
if annotation == 0:
    path = "E:/Work/TnR Lab/FED-MAIN/Federated-ECG/ECG Classification/datasets/ptbdb_normal.csv"
else: 
    path = "E:/Work/TnR Lab/FED-MAIN/Federated-ECG/ECG Classification/datasets/ptbdb_abnormal.csv"

# Create a DataFrame from the list of NumPy arrays
# remove the first and the last beat as they tend to remain incomplete
new_beats = pd.DataFrame(beats[1: (len(beats)-1)])

# Load the existing CSV file
existing_dataset = pd.read_csv(path, header=None)

# Add the DataFrame to the existing DataFrame
new_df = pd.concat([existing_dataset, new_beats], axis=0)

# Write the new DataFrame to the CSV file
new_df.to_csv(path, index=False)