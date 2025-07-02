from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List
import logging
from sqlmodel import SQLModel, Field as ORMField, create_engine, Session, select
import os
import csv
from fastapi.responses import StreamingResponse
from io import StringIO
import requests
import asyncio

# Example NGOs with their needs and locations
NGOS = [
    {"name": "EduRelief", "category": "Education", "location": "Delhi"},
    {"name": "WarmClothes", "category": "Clothing", "location": "Mumbai"},
    {"name": "Food4All", "category": "Food", "location": "Delhi"},
    {"name": "HealthAid", "category": "Medical", "location": "Chennai"},
    # Add more as needed
]

# Define Donation model
class Donation(BaseModel):
    name: str = Field(..., example="John Doe")
    amount: float = Field(..., gt=0, example=100.0)
    itemName: str = Field(..., example="Blankets")
    category: str = Field(..., example="Clothing")
    quantity: int = Field(..., gt=0, example=10)
    location: str = Field(..., example="New York")

# Define DonationDB model for the database
class DonationDB(SQLModel, table=True):
    id: int | None = ORMField(default=None, primary_key=True)
    name: str
    amount: float
    itemName: str
    category: str
    quantity: int
    location: str

# Create FastAPI app
app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define response model
class DonationResponse(BaseModel):
    status: str
    message: str
    data: Dict[str, Any]

# Set up SQLite database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///donations.db")
engine = create_engine(DATABASE_URL, echo=True)

# Create tables on startup
@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

# POST endpoint for donations
@app.post("/donate", response_model=DonationResponse)
async def receive_donation(donation: Donation):
    """
    Receives a validated donation and saves it to the database.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Received donation: {donation.dict()}")
    # Save to DB
    donation_db = DonationDB(**donation.dict())
    with Session(engine) as session:
        session.add(donation_db)
        session.commit()
        session.refresh(donation_db)
    return {
        "status": "success",
        "message": "Thank you for your donation!",
        "data": donation.dict()
    }

# GET endpoint to list all donations
@app.get("/donations", response_model=List[DonationDB])
def list_donations(
    category: Optional[str] = Query(None),
    location: Optional[str] = Query(None)
):
    with Session(engine) as session:
        query = select(DonationDB)
        if category:
            query = query.where(DonationDB.category == category)
        if location:
            query = query.where(DonationDB.location == location)
        donations = session.exec(query).all()
        return donations

@app.get("/donations/export")
def export_donations():
    with Session(engine) as session:
        donations = session.exec(select(DonationDB)).all()
        output = StringIO()
        writer = csv.writer(output)
        # Write header
        writer.writerow(["id", "name", "amount", "itemName", "category", "quantity", "location"])
        # Write data
        for d in donations:
            writer.writerow([d.id, d.name, d.amount, d.itemName, d.category, d.quantity, d.location])
        output.seek(0)
        return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=donations.csv"})

@app.delete("/donations/{donation_id}")
def delete_donation(donation_id: int):
    with Session(engine) as session:
        donation = session.get(DonationDB, donation_id)
        if not donation:
            raise HTTPException(status_code=404, detail="Donation not found")
        session.delete(donation)
        session.commit()
        return {"status": "success", "message": "Donation deleted"}

@app.put("/donations/{donation_id}", response_model=DonationDB)
def update_donation(donation_id: int, updated: Donation = Body(...)):
    with Session(engine) as session:
        donation = session.get(DonationDB, donation_id)
        if not donation:
            raise HTTPException(status_code=404, detail="Donation not found")
        for field, value in updated.dict().items():
            setattr(donation, field, value)
        session.add(donation)
        session.commit()
        session.refresh(donation)
        return donation

@app.post("/match_ngo")
def match_ngo(donation: Donation):
    """
    Suggests the best-matching NGO for a donation based on category and location.
    """
    # First, try to match both category and location
    matches = [
        ngo for ngo in NGOS
        if ngo["category"].lower() == donation.category.lower()
        and ngo["location"].lower() == donation.location.lower()
    ]
    if matches:
        return {"matched_ngo": matches[0]}
    # If no exact match, try to match by category only
    matches = [ngo for ngo in NGOS if ngo["category"].lower() == donation.category.lower()]
    if matches:
        return {"matched_ngo": matches[0]}
    # If still no match, return None
    return {"matched_ngo": None, "message": "No suitable NGO found"}

NEWS_API_KEY = "fc1c145f579e4eddb5a09b01f9532d81"  # NewsAPI key

class CrisisAlertsResponse(BaseModel):
    crisis_headlines: List[str]

@app.get("/crisis_alerts", response_model=CrisisAlertsResponse)
async def crisis_alerts():
    print("Crisis endpoint called")
    url = f"https://newsapi.org/v2/top-headlines?apiKey={NEWS_API_KEY}"
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, requests.get, url)
    if response.status_code != 200:
        return {"error": "Failed to fetch news"}
    print(response.json())
    headlines = [article["title"] for article in response.json().get("articles", [])]
    print(headlines)
    crisis_keywords = [
        "flood", "earthquake", "fire", "drought", "epidemic", "outbreak", "disaster",
        "cyclone", "storm", "covid", "pandemic", "landslide", "tsunami", "heatwave",
        "violence", "protest", "shortage", "accident", "collapse", "explosion",
        "emergency", "aid", "relief", "evacuation", "the", "india", "indian"
    ]
    flagged = [h for h in headlines if any(k in h.lower() for k in crisis_keywords)]
    return {"crisis_headlines": flagged}

OPENROUTESERVICE_API_KEY = "5b3ce3597851110001cf62483ac438cb7cdb43a59c85dd6fa9e0075e"

def get_best_feature(features):
    # Prefer features with country == 'India' and highest confidence
    india_features = [f for f in features if f['properties'].get('country') == 'India']
    if india_features:
        return max(india_features, key=lambda f: f['properties'].get('confidence', 0))
    # Fallback to highest confidence overall
    if features:
        return max(features, key=lambda f: f['properties'].get('confidence', 0))
    return None

@app.get("/optimize_route")
async def optimize_route(start: str, end: str):
    """
    Returns the optimal route between start and end locations using OpenRouteService.
    """
    # Geocode start and end locations
    geocode_url = "https://api.openrouteservice.org/geocode/search"
    def geocode_location(location):
        if "india" not in location.lower():
            location += ", India"
        resp = requests.get(
            geocode_url,
            headers={"Authorization": OPENROUTESERVICE_API_KEY},
            params={"api_key": OPENROUTESERVICE_API_KEY, "text": location}
        )
        return resp

    loop = asyncio.get_event_loop()
    start_resp = await loop.run_in_executor(None, geocode_location, start)
    end_resp = await loop.run_in_executor(None, geocode_location, end)
    start_json = start_resp.json()
    end_json = end_resp.json()
    print(start_json)
    print(end_json)
    start_feature = get_best_feature(start_json.get("features", []))
    end_feature = get_best_feature(end_json.get("features", []))
    if not start_feature or not end_feature:
        return {"error": "Could not geocode start or end location"}
    start_coords = start_feature["geometry"]["coordinates"]
    end_coords = end_feature["geometry"]["coordinates"]
    # Get directions
    directions_url = "https://api.openrouteservice.org/v2/directions/driving-car"
    body = {
        "coordinates": [start_coords, end_coords]
    }
    import json as pyjson
    directions_resp = await loop.run_in_executor(
        None,
        lambda: requests.post(
            directions_url,
            headers={"Authorization": OPENROUTESERVICE_API_KEY, "Content-Type": "application/json"},
            data=pyjson.dumps(body)
        )
    )
    directions_json = directions_resp.json()
    print(directions_json)
    try:
        route = directions_json["routes"][0]
        summary = route["summary"]
        steps = route["segments"][0]["steps"]
        return {
            "distance_km": summary["distance"] / 1000,
            "duration_min": summary["duration"] / 60,
            "steps": [step["instruction"] for step in steps]
        }
    except (KeyError, IndexError):
        return {"error": "Could not find route"}