import os
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine,URL, Column, Integer, String, Text, ForeignKey, Date
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.exc import SQLAlchemyError

#Database setup
#DATABASE_URL=URL.create("sqlite",host="localhost",database="chat_history.db")
# DATABASE_URL="sqlite:////home/cyberjroid/Documents/Projects/CyberJroid/Tutorials/Divverse/Team Kappa/kappa-ai-finquiry-backend/chathistory.db"
DATABASE_URL="sqlite:///chathistory.db"
engine = create_engine(DATABASE_URL,connect_args={"check_same_thread": False})
SessionLocal=sessionmaker(bind=engine, autoflush=False,autocommit=False)

#SQLAlchemy ORM model
Base= declarative_base()


class User(Base):
    __tablename__ = "user"
    
    id = Column(Integer, primary_key=True, index=True)
    username= Column(String,unique=True, index=True)
    first_name= Column(String)
    last_name= Column(String)
    image=Column(String)
    hashed_password= Column(String)

class AdviseHistory(Base):
    __tablename__ = "chat_history" 
    
    id = Column(Integer,primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"))
    stock= Column(String)
    message = Column(Text)
        
    user= relationship("User")

  
        
#Create Tables
Base.metadata.create_all(bind=engine)

#OAuth2 for authentication
oauth2_scheme =OAuth2PasswordBearer(tokenUrl="Login")

#pydantic models
class CreateUser(BaseModel):
    username: str
    password: str
    first_name: str
    last_name: str
    image: str| None

class Message(BaseModel):
    username: str
    password: str
    message: str
    stock: str
    
    
class AiAdviseHistory(BaseModel):
    password: str | None
    username: str |None
    stock: str | None

class Stock(BaseModel):
    stock:str 