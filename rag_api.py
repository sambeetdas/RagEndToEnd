from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
import os   

class RequestModel(BaseModel):
    question: str

app = FastAPI()

@app.post("/")
async def create_item(model: RequestModel):
    MONGODB_URI = os.getenv("MONGODB_URI")
    DATABASE_NAME = os.getenv("DATABASE_NAME") 
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")  
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    rag = RAGPipeline(
        mongodb_uri=MONGODB_URI,
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME,
        embedding_model = EMBEDDING_MODEL
    )
    response = await rag.search_similar_documents(model.question, limit=3)
    return response

