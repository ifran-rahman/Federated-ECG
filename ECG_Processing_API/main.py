from typing import List
import json
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
import requests
import pandas as pd
import crud, models, schemas
from database import SessionLocal, engine
from data_process.dataProcessor import signal_to_beats

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

# upload signal
@app.post("/signals/", response_model=schemas.ItemCreate)
def upload_signal(user: schemas.UserCreate, db: Session = Depends(get_db)):

    created_signal = crud.create_user(db=db, user=user)
    
    signal = user.signal_data

    print("length of signal", len(signal))
    # signal = str_to_signal(user.signal_data)
    beats = signal_to_beats(signal)
    beats_json = beats_to_json(beats)
    signal_id = created_signal.id

    url = 'http://127.0.0.1:8000/signals/'+ str(signal_id) +'/beats/'
    
    print(len(beats))
    
    myobj = {"beats": beats_json} 
    x = requests.post(url, json = myobj)

    return myobj

@app.get("/signals/", response_model=List[schemas.Signal])
def read_signals(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    signals = crud.get_users(db, skip=skip, limit=limit)
    return signals

@app.get("/signals/{user_id}", response_model=schemas.Signal)
def read_signal(user_id: int, db: Session = Depends(get_db)):
    db_signal = crud.get_user(db, user_id=user_id)
    if db_signal is None:
        raise HTTPException(status_code=404, detail="signal not found")
    return db_signal

@app.post("/signals/{user_id}/beats/", response_model=schemas.Item)
def create_beats_for_user(
    user_id: int, item: schemas.ItemCreate, db: Session = Depends(get_db)
):
    return crud.create_user_item(db=db, item=item, user_id=user_id)

@app.get("/beats/", response_model=List[schemas.Item])
def read_beats(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    beats = crud.get_items(db, skip=skip, limit=limit)
    return beats



    