## This script deals with posting and getting data from the API
import requests
import numpy as np
import pandas as pd 
from data_process.pre_n_post_process import beats_str_to_list
from client1 import *


# Dataset paths
NORMAL = 'datasets/ptbdb_normal.csv'
ABNORMAL = 'datasets/ptbdb_abnormal.csv'

def post_signal(signal_data, is_verified, annotation):
  # annotation needs to be added to the db later

  # POST ECG signal to ECG-Process API
  url = "http://127.0.0.1:8000/signals/"

  # The data to post
  data = {
    "signal_data": signal_data,
    "is_verified": is_verified
  }

  # POST signal
  API_response = requests.post(url, json = data)

  return API_response

def get_beats(API_response):
  # GET heartbeats
  json_data = API_response.json()
  beats_json = json_data['beats']

  # GET beats and convert beats_json to list
  beats = beats_str_to_list(beats_json)

  return beats
# process and add new heartbeats to existing dataset based on annotationi
def process_and_add(beats, annotation):    
  # choose dataset path according to annotation
  if annotation == 0:
      path = NORMAL
  else: 
      path = ABNORMAL

  # Create a DataFrame from the list of NumPy arrays
  # remove the first and the last beat as they tend to remain incomplete
  new_beats = pd.DataFrame(beats[1: (len(beats)-1)])

  # Load the existing CSV file
  existing_dataset = pd.read_csv(path, header=None)

  # Add the DataFrame to the existing DataFrame
  new_df = pd.concat([existing_dataset, new_beats], axis=0)

  # Write the new DataFrame to the CSV file
  new_df.to_csv(path, index=False)

# def main() -> None:
#   # load a complete ecg signal of a patient
#   signal_data = load_signal()

#   # if the signal is verified by a doctor or not
#   is_verified = 0

#   # we assume the annotation for the current signal is 1 (Abnormal)
#   annotation = 1

#   API_response = post_signal(signal_data, is_verified, annotation)

#   beats = get_beats(API_response)

#   process_and_add(beats, annotation)

# if __name__ == "__main__":
#     main()
