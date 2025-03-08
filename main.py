import base64
import os
import io
import datetime
import json
from pathlib import Path

import bcrypt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Response
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pinecone import Pinecone, ServerlessSpec
import requests
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from model import *



# Load environment variables
load_dotenv(override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RAPID_API_KEY= os.getenv("RAPID_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")



headers_params={
    "headers" : {
	"x-rapidapi-key":RAPID_API_KEY ,
	"x-rapidapi-host": "yahoo-finance166.p.rapidapi.com" }
}
#initialist Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
# Initialize Groq API
llm = ChatGroq(api_key=GROQ_API_KEY,
               model="llama-3.3-70b-versatile",
               temperature=0.2,
               max_retries=2)

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name= ["stock-insight-index", "stock-sentiment-index"]
try:
    for name in index_name:
        if name not in pc.list_indexes():
            pc.create_index(
                            name,
                            dimension=384,
                            spec=ServerlessSpec(
                            cloud="aws",region="us-east-1")
            )
except Exception as e:
    pass

# Initialize Indexes (only if they exist)
insight_index = pc.Index("stock-insight-index")
    
sentiment_index = pc.Index("stock-sentiment-index")
    
#Initialize HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI instruction template
# Default initiate conversation

    
#stock price related info
def stock_price(stock:str):
    """ Get current price quote of stock"""
    url = "https://yahoo-finance166.p.rapidapi.com/api/stock/get-price"
    querystring = {"region":"US","symbol":stock}
    response =requests.get(url, headers=headers_params["headers"], params=querystring)
    return response.json()


def stock_chart(stock:str):
    """Get price chart for the query stock"""
    url="https://yahoo-finance166.p.rapidapi.com/api/stock/get-chart"
    querystring = {"region":"US","range":"3d","symbol":stock,"interval":"1h"}
    response =requests.get(url, headers=headers_params["headers"], params=querystring)
    return response.json()


prompt ="""
    Instructions:
    - Provide valid web address to be use for further information.
    - Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'.
    - Utilize the context provided for accurate and specific information.
    - Incorporate your preexisting knowledge to enhance the depth and relevance of your response.
    - Clearly present advantages and disadvantages of buying a stock based on retrieved information.
    - Cite your sources.
    - Format the response such that it can be displayed in an html page and.
    - Provide a brief summary of the stock's performance and future outlook.

    Context: {context}

    Question: This is for stock {stock}. {query}
"""


# Instantiate prompt
# function to handle prompt input
def use_prompt(context:str|None,query:str |None, stock:str|None):
    prompt_template = ChatPromptTemplate.from_template(
    prompt ).format_messages(context=context, query=query, stock=stock
        )
    return prompt_template


#stock price chart
@app.post("/stock-chart")
def get_stock_chart(request:Stock):
    #make api request to fetch price stamps
    data = stock_chart(request.stock)
    try:
        timestamps = data["chart"]["result"][0]["timestamp"]
        closing_prices = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        # Convert timestamps to readable dates
        dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
        # Create the plot
        plt.figure(figsize=(10, 5))
        plt.plot(dates, closing_prices, marker="o", linestyle="-", color="b", label="Close Price")
        plt.xlabel("Time")
        plt.ylabel("Closing Price (USD)")
        plt.title(f"Stock Prices for {data['chart']['result'][0]['meta']['symbol']}")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        # Save the plot to a bytes buffer
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format="png")
        plt.close()
        img_bytes.seek(0)

        # Encode the image in Base64
        encoded = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        # Build the Data URL
        data_url = f"data:image/png;base64,{encoded}"
        
        # Return the Data URL in a JSON response
        return JSONResponse(content={"image": data_url})
    except KeyError:
        return {"error": "Invalid data format"}

#Stock price endpoint
@app.post("/stock-price")
def get_stock_price(request:Stock):
    #make api request to fetch price
    data= stock_price(request.stock)
    price_data = data["quoteSummary"]["result"][0]["price"]
    df = pd.DataFrame([{
    'Symbol': price_data['symbol'],
    'Short Name': price_data['shortName'],
    'Currency': price_data['currency'],
    'Market Price': price_data['regularMarketPrice']['raw'],
    'Market Change': price_data['regularMarketChange']['raw'],
    'Change Percent': price_data['regularMarketChangePercent']['fmt'],
    'Day High': price_data['regularMarketDayHigh']['raw'],
    'Day Low': price_data['regularMarketDayLow']['raw'],
    'Market Volume': price_data['regularMarketVolume']['raw'],
    'Previous Close': price_data['regularMarketPreviousClose']['raw'],
    'Market Cap': price_data['marketCap']['raw'],
    'Market State': price_data['marketState']}])
    
    # Convert DataFrame to JSON response
    return{"price_data": df.to_dict(orient="records")}
        
    
     
    
#Function to create new user
@app.post("/register")
def register_user(request:CreateUser):
    db = SessionLocal()
    hash_password = bcrypt.hashpw(request.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    db_user = db.query(User).filter(User.username== request.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    new_user = User(username=request.username, hashed_password=hash_password, image=request.image,
                    first_name=request.first_name,last_name=request.last_name)
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    db.close()
    
    return {"message": "User registered successfully"}

#function to authenticate user
@app.post("/login")
def login_user(form_data:OAuth2PasswordRequestForm =Depends()):
    #get user from database
    db = SessionLocal()
    user = db.query(User).filter(User.username == form_data.username).first()
    
    if not user or not bcrypt.checkpw(form_data.password.encode("utf-8"), user.hashed_password.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token_payload = {
        "username": user.username,
        "image": user.first_name,
        "user_id": user.id
        #"exp": datetime.datetime() + datetime.timedelta(hours=1)  # Token expires in 1 hour
    }
    #jwt can be use to manage data validity her
    
    return {"response": token_payload}
    
#function to store chat message
def store_advice_message(username: int, message: str, ticker: str):
    db = SessionLocal()
    try:
        # Create and add new chat entry
        chat_entry = AdviseHistory(user_id=username, message=message, stock=ticker)
        db.add(chat_entry)
        db.commit()
    
    except Exception as e:
        db.rollback()  # Rollback transaction if there's an error
        print(f"Error storing advice message: {e}")
    
    finally:
        db.close()  # Ensure session is always closed

"""
def store_advice_message(user_id: int, message:str, ticker:str):
    db = SessionLocal()
    chat_entry=AdviseHistory(user_id=user_id,message=message,stock=ticker)
    db.add(chat_entry)
    db.commit()
    db.close()"""

def get_user(password:str,username:str):
    db= SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    if not user or not bcrypt.checkpw(password.encode("utf-8"), user.hashed_password.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {"user_id":user.id,"username":user.username}


#Retrieve previous conversation
def retrieve_previous_advice(user_id: int, ticker: str):
    db = SessionLocal()
    try:
        # Check if the user has previous advice
        messages = db.query(AdviseHistory).filter(
            (AdviseHistory.user_id == user_id) & (AdviseHistory.stock == ticker)
        ).order_by(AdviseHistory.id.asc()).all()

        # Return empty response if no messages found
        if not messages:
            return {"response": ""}

        # Format and return history
        return {
            "user_id": user_id,"ticker":ticker,
            "history": [{"id": msg.id, "message": msg.message} for msg in messages]
        }
    
    finally:
        db.close()  # Ensure the session is closed


# get AI expert advice
def get_ai_advice(user_id:str, prompt:str,stock:str, query:str):
    """This function embed user query to get advice on particular stocks"""
    #for better optimized similarity search attach stock ticker to query
    new_query = query +' '+'?'+ stock
    query_embedding =embedding_model.embed_query(new_query)
    #query as many index table to get information
    #join all query together to use single prompt
    vector_response_list=[]
    results=insight_index.query(vector=query_embedding,
                    top_k=10,
                    include_metadata=True,
                    filter={"ticker": stock}
                    )
    retrieve_content=[(result["metadata"][stock], result["metadata"]["content"]) for result in results["matches"]]
    vector_response_list.extend([f"{meta}: {content}" for meta, content in retrieve_content])
    #get sentiments news
    _results=sentiment_index.query(vector=query_embedding,
                    top_k=10,
                    include_metadata=True,
                    filter={"ticker": stock}
                    )
    retrievecontent=[(result["metadata"][stock], result["metadata"]["content"]) for result in _results["matches"]]
    vector_response_list.extend([f"{meta}: {content}" for meta, content in retrievecontent])
    
    # Join all retrieved content into a single string
    vector_response = " ".join(vector_response_list)
    prompt = use_prompt(context=vector_response,query=query, stock=stock)
    try:
        response= llm.invoke(
            prompt
        )
        for event in response:
            #store_advice_message(user_id=user_id, message=event[-1],ticker=stock)
            return {"response":event[-1]}  # Return the text response
    except Exception as e:
        return {"error": str(e)}


#Retrieve previous chat history for particular AI
@app.post("/history")
def get_advice_history(request:AiAdviseHistory):
    #fetch history with username
    user= get_user(request.password, request.username)
    history = retrieve_previous_advice(user_id=user["user_id"], ticker=request.stock)
    return history
        
@app.post("/financial/advice")
async def chat_text(request:Message):
    #get user state
    prompt_user = get_user(request.password, request.username)
    
    #get ai advice
    response = get_ai_advice(user_id=prompt_user["user_id"],prompt=prompt,
                             query=request.message, stock=request.stock)
    
    store_advice_message(username=request.username, message=response["response"],ticker=request.stock)
    return response
