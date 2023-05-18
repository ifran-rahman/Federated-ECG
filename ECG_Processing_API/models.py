from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from database import Base


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    signal_data = Column(String, index=True)
    is_verified = Column(Boolean, default=True)
    beats = relationship("Beat", back_populates="owner")


class Beat(Base):
    __tablename__ = "beats"

    id = Column(Integer, primary_key=True, index=True)
    beats = Column(String, index=True)
    # description = Column(String, index=True)
    owner_id = Column(Integer, ForeignKey("signals.id"))
    owner = relationship("Signal", back_populates="beats")
    