"""
ORM model for the users table.
Stores user credentials and role-based access control (RBAC) roles.
"""
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func

from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="viewer", nullable=False)  # admin / data_scientist / viewer

    created_at = Column(DateTime(timezone=True), server_default=func.now())
