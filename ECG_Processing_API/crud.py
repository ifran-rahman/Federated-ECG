from sqlalchemy.orm import Session

import models, schemas

def process(signal):
    return [signal, signal, signal]

def get_signal(db: Session, signal_id: int):
    return db.query(models.Signal).filter(models.Signal.id == signal_id).first()

def get_signals(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Signal).offset(skip).limit(limit).all()

def create_signal(db: Session, signal: schemas.SignalCreate):
    db_signal= models.Signal(signal_data=str(signal.signal_data), is_verified=signal.is_verified) 
    db.add(db_signal)
    db.commit()
    db.refresh(db_signal)
    return db_signal

def get_beats(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Beat).offset(skip).limit(limit).all()

def create_beats(db: Session, item: schemas.BeatCreate, signal_id: int):
    db_beat = models.Beat(**item.dict(), owner_id=signal_id)
    db.add(db_beat)
    db.commit()
    db.refresh(db_beat)
    return db_beat