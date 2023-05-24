# API Instructions
![alt text](simplified_diagram.png)

## Setting Up
### Create a conda environment with the requirements specified in the requirements.txt file
conda create --name <env-name> --file requirements.txt

### Activate the conda environment
conda activate <env-name>

### Navigate to ECG Processing API 
cd ECG_Processing_API

### Start the server
uvicorn main:app --reload

## POST ECG Signal
POST ECG Signal as list in the following URL, "http://127.0.0.1:8000/signals/"
(List length must be at least 500)
In response it'll provide a beats string

The API saves the signal in the "signals" table.
Extracted heartbeats gets saved in the "items" table.

### convert signal to list
signal_data = signal.values.tolist()

### load the ECG-Process API URL
url = "http://127.0.0.1:8000/signals/"

### The data to post
data = {
  "signal_data": signal_data,
   "is_verified": 0
}

### POST signal and get response
API_response = requests.post(url, json = data)

## GET all ECG Signals
http://127.0.0.1:8000/signals/

## GET a particular ECG Signal
"/signals/{signal_id}"

## GET beats which have been extracted from a particular signal
'http://127.0.0.1:8000/signals/'+ str(signal_id) +'/beats/'

## GET all the beats
http://127.0.0.1:8000/beats/

