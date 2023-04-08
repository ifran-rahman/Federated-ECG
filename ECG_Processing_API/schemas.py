from typing import List, Union

from pydantic import BaseModel


class ItemBase(BaseModel):
    beats: str
    # description: Union[str, None] = None


class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    signal_data: str


class UserCreate(UserBase):
    signal_data: List[float]
    is_verified: bool
    # password: str


class Signal(UserBase):
    id: int
    is_verified: bool
    items: List[Item] = []

    class Config:
        orm_mode = True