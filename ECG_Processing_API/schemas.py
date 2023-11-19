from typing import List, Union
from pydantic import BaseModel


class BeatBase(BaseModel):
    beats: str
    # description: Union[str, None] = None
    
class BeatCreate(BeatBase):
    pass

class Beat(BeatBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True

class SignalBase(BaseModel):
    signal_data: str

class SignalCreate(SignalBase):
    signal_data: List[float]
    is_verified: bool
    # password: str

class Signal(SignalBase):
    id: int
    is_verified: bool
    beats: List[Beat] = []

    class Config:
        orm_mode = True

# class Beats_from_Signal(SignalBase):
#     id: int
#     is_verified: bool
#     beats: List[Beat] = []

#     class Config:
#         orm_mode = True

class Beats_from_Signal(SignalBase):
    signal_data: List[float]
