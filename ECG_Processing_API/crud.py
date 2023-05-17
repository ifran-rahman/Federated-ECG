from sqlalchemy.orm import Session

import models, schemas

def process(signal):
    return [signal, signal, signal]

def get_signal(db: Session, user_id: int):
    return db.query(models.Signal).filter(models.Signal.id == user_id).first()

# def get_user_by_email(db: Session, email: str):
#     return db.query(models.Signal).filter(models.Signal.email == email).first()

def get_signals(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Signal).offset(skip).limit(limit).all()

def create_signal(db: Session, user: schemas.UserCreate):
    db_user = models.Signal(signal_data=str(user.signal_data), is_verified=user.is_verified) 
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_beats(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Item).offset(skip).limit(limit).all()

def create_beats(db: Session, item: schemas.ItemCreate, user_id: int):
    db_item = models.Item(**item.dict(), owner_id=user_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item