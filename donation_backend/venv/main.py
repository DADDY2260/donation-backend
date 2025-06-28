from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/donate")
async def receive_donation(request: Request):
    try:
        data = await request.json()
        print("Received donation:", data)
        return {"status": "success", "message": "Donation received!"}
    except Exception as e:
        print("Error parsing donation:", e)
        return {"status": "error", "message": "Failed to parse donation"}