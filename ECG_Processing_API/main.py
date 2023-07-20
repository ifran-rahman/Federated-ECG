from typing import List
import json
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
import requests
import pandas as pd
import crud, models, schemas
from database import SessionLocal, engine
from data_process.dataProcessor import signal_to_beats
from blockchain_crud import save_data

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

def str_to_signal(signal_str):
    return pd.Series(signal_str.split('\n'))

def str_to_list(signal_str):
    return eval(signal_str)

def beats_to_json(beats):

    # Convert each numpy array in beats into a list
    beats_list = [beat.tolist() for beat in beats]

    # Convert beats_list into a JSON string
    beats_json = json.dumps(beats_list)
    return beats_json

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# upload a signal
@app.post("/signals/", response_model=schemas.BeatCreate)
def upload_signal(signal: schemas.SignalCreate, db: Session = Depends(get_db)):

    created_signal = crud.create_signal(db=db, signal=signal)
    
    signal = signal.signal_data

    # save signal
    signal_data = {
                    "id": "ownerId1_signal1",
                    "type": "patient",
                    "data": signal,
                    "isVerified": "true",
                    "ownerId": "ownerId1"
                    }
    save_data(signal_data)
    
    print("length of signal", len(signal))
    beats = signal_to_beats(signal)
    beats_json = beats_to_json(beats)
    signal_id = created_signal.id

    url = 'http://127.0.0.1:8000/signals/'+ str(signal_id) +'/beats/'
    
    print(len(beats))
    
    myobj = {"beats": beats_json} 
    x = requests.post(url, json = myobj)

    print("signal posted")
    # save beats
    beats_data = {
                    "id": "signal1_beats1",
                    "type": "patient",
                    "data": beats_json,
                    "isVerified": "true",
                    "ownerId": "ownerId1"
                    }
    save_data(beats_data)
    return myobj

# GET signals
@app.get("/signals/", response_model=List[schemas.Signal])
def read_signals(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    signals = crud.get_signals(db, skip=skip, limit=limit)
    return signals

# GET a particular signal
@app.get("/signals/{signal_id}", response_model=schemas.Signal)
def read_signal(signal_id: int, db: Session = Depends(get_db)):
    db_signal = crud.get_signal(db, signal_id=signal_id)
    if db_signal is None:
        raise HTTPException(status_code=404, detail="signal not found")
    return db_signal

# GET beats which have been extracted from a particular signal
@app.post("/signals/{signal_id}/beats/", response_model=schemas.Beat)
def create_beats_for_signal(
    signal_id: int, item: schemas.BeatCreate, db: Session = Depends(get_db)
):
    return crud.create_beats(db=db, item=item, signal_id=signal_id)

# GET all heartbeats
@app.get("/beats/", response_model=List[schemas.Beat])
def read_beats(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    beats = crud.get_beats(db, skip=skip, limit=limit)
    return beats


