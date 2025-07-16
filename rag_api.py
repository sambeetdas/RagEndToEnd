from fastapi import FastAPI
from pydantic import BaseModel


class RequestModel(BaseModel):
    question: str

app = FastAPI()

@app.post("/")
async def create_item(model: RequestModel):
    return model

