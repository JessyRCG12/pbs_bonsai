# main.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from app.orchestrator.chat_orchestrator import ChatOrchestrator, BuyerProfile, ListingRanker

app = FastAPI(title="BonsAI API")

orch = ChatOrchestrator(
    species_csv=str(Path(__file__).resolve().parents[1] / "data" / "bonsai_species.csv"),
    marketplace_csv=str(Path(__file__).resolve().parents[1] / "data" / "bonsai_marketplace.csv")
)

class ChatRequest(BaseModel):
    message: str
    top_k: Optional[int] = 5

class RecommendRequest(BaseModel):
    buyer_profile: Dict[str, Any]
    top_k: Optional[int] = 5
    species: Optional[str] = None

@app.post("/chat")
def chat(req: ChatRequest):
    result = orch.handle_message(req.message, k=req.top_k or 5)
    return result

@app.post("/recommend")
def recommend(req: RecommendRequest):
    bp = BuyerProfile(
        experience_level=req.buyer_profile["experience_level"],
        budget_eur=float(req.buyer_profile["budget_eur"]),
        preferred_environment=req.buyer_profile["preferred_environment"],
        available_light=req.buyer_profile["available_light"],
        maintenance_time=req.buyer_profile.get("maintenance_time","unsure"),
        location=req.buyer_profile.get("location","unknown"),
        min_temperature_c=req.buyer_profile.get("min_temperature_c"),
        notes=req.buyer_profile.get("notes"),
    )
    lr = ListingRanker(str(Path(__file__).resolve().parents[1] / "data" / "bonsai_marketplace.csv"))
    items = lr.rank(bp, k=req.top_k or 5, species=req.species)
    return {
        "buyer_profile": req.buyer_profile,
        "items": [i.__dict__ for i in items]
    }
