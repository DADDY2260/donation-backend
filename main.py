from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model
class Donation(BaseModel):
    itemName: str
    category: str
    quantity: int
    location: str

@app.post("/donate")
async def receive_donation(donation: Donation):
    print("✅ Received donation:", donation.dict())
    return {
        "status": "success",
        "message": "Thank you for your donation!",
        "data": donation.dict()
    }
