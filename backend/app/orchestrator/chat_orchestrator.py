# chat_orchestrator.py
from __future__ import annotations

import json
import os
import re
import time
import concurrent.futures
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple

import mlflow
import pandas as pd

# Use Ollama client when available; tolerate missing lib
try:
    from ollama import Client as OllamaClient  # type: ignore
except Exception:
    OllamaClient = None  # type: ignore


# --------- Data Contracts ---------

@dataclass
class BuyerProfile:
    experience_level: str
    budget_eur: float
    preferred_environment: str
    available_light: str
    maintenance_time: str
    location: str
    min_temperature_c: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class RecommendationItem:
    listing_id: str
    match_score: float
    rationale: str


# --------- Helpers ---------

def _norm(s: str) -> str:
    return str(s).strip().lower()


def _budget_band(budget_eur: float) -> str:
    if budget_eur <= 100:
        return "low"
    if budget_eur <= 250:
        return "medium"
    return "high"


def _water_map(s: str) -> str:
    m = _norm(s)
    if m.startswith("low"):
        return "low"
    if m.startswith("moderate"):
        return "medium"
    return "high"


def _env_from_placement(s: str) -> str:
    m = _norm(s)
    if "indoor" in m and "outdoor" in m:
        return "either"
    if "indoor" in m:
        return "indoor"
    return "outdoor"


def _load_system_prompt() -> str:
    """Load the buyer intake system prompt from the prompts folder."""
    base = Path(__file__).resolve().parents[1]  # /app/app
    path = base / "prompts" / "buyer_intake_system_prompt.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    # Safe fallback so we never 500 if the file is missing
    return (
        "You are a bonsai buyer intake assistant. "
        "Return ONLY a JSON object with keys: "
        "experience_level, budget_eur, preferred_environment, available_light, "
        "maintenance_time, location, min_temperature_c, notes."
    )


# --------- Species Scorer (deterministic) ---------

class SpeciesScorer:
    def __init__(self, species_csv: str):
        self.df = pd.read_csv(species_csv)
        self.df.columns = [c.strip().lower() for c in self.df.columns]

    def score_species(self, profile: BuyerProfile, top_k: int = 3) -> List[Dict]:
        rows = self.df.to_dict(orient="records")
        scored = []
        for r in rows:
            score = 1.0
            rationale_bits = []

            # environment
            env_pref = profile.preferred_environment
            env_species = _env_from_placement(r["main_placement"])
            if env_pref in ("indoor", "outdoor") and env_species != env_pref:
                score -= 0.25
            else:
                rationale_bits.append("environment match")

            # light (heuristic: assume species need ~ medium; penalize distance)
            light_map = {"low": 0, "medium": 1, "high": 2}
            user_light = profile.available_light
            diff = abs(light_map.get(user_light, 1) - light_map.get("medium", 1))
            score -= 0.1 * diff
            if diff == 0:
                rationale_bits.append("light match")

            # difficulty
            order = {"beginner": 0, "intermediate": 1, "advanced": 2}
            species_level = _norm(r["level"])
            diff = max(0, order.get(species_level, 1) - order.get(_norm(profile.experience_level), 0))
            score -= 0.2 * diff
            if diff == 0:
                rationale_bits.append("difficulty match")

            # budget
            band = _budget_band(profile.budget_eur)
            species_budget = _norm(r["budget"]).replace("/high", "")
            if band == "low" and "low" not in species_budget:
                score -= 0.2
            elif band == "medium" and "medium" not in species_budget and "low" not in species_budget:
                score -= 0.1
            else:
                rationale_bits.append("within budget")

            # climate (outdoor + min temp)
            if env_pref == "outdoor" and profile.min_temperature_c is not None:
                hardiness = _norm(r["hardiness"])
                if "sensitive" in hardiness and profile.min_temperature_c < 5:
                    score -= 0.2
                if "very hardy" in hardiness:
                    rationale_bits.append("hardiness match")

            score = max(0.0, min(1.0, score))
            scored.append({
                "species": r["species"],
                "species_score": round(score, 3),
                "rationale": ", ".join(rationale_bits) if rationale_bits else "best fit given constraints",
            })

        scored.sort(key=lambda x: x["species_score"], reverse=True)
        return scored[:top_k]


# --------- Listing Ranker (deterministic) ---------

class ListingRanker:
    def __init__(self, marketplace_csv: str):
        self.df = pd.read_csv(marketplace_csv)
        self.df.columns = [c.strip().lower() for c in self.df.columns]

    @staticmethod
    def _compatibility_penalty(row: Dict, profile: BuyerProfile, species_hint: Optional[str]) -> float:
        penalty = 0.0

        # species mismatch (if user has a target species)
        if species_hint and _norm(row["species"]) != _norm(species_hint):
            penalty += 0.4

        # availability
        if _norm(str(row.get("availability", ""))) != "in_stock":
            penalty += 0.5

        # budget overrun
        budget = profile.budget_eur or 0.0
        price = float(row.get("price", 0.0))
        if budget > 0 and price > budget:
            over = min((price - budget) / max(1.0, budget), 0.5)
            penalty += 0.4 * over

        return min(1.0, max(0.0, penalty))

    def rank(self, profile: BuyerProfile, k: int = 5, species: Optional[str] = None) -> List[RecommendationItem]:
        records = self.df.to_dict(orient="records")
        scored: List[Tuple[float, Dict]] = []
        for r in records:
            p = self._compatibility_penalty(r, profile, species)
            score = max(0.0, 1.0 - p)

            rationale_bits = []
            if species and _norm(r["species"]) == _norm(species):
                rationale_bits.append("species match")
            if r.get("price", 0) <= profile.budget_eur:
                rationale_bits.append("within budget")
            if _norm(str(r.get("availability", ""))) == "in_stock":
                rationale_bits.append("in stock")

            scored.append((score, {
                "listing_id": r["prod_id"],
                "match_score": round(score, 3),
                "rationale": ", ".join(rationale_bits) if rationale_bits else "best overall match",
            }))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [RecommendationItem(**x[1]) for x in scored[:k]]
        return top


# --------- Quick Heuristic Intake ---------

def extract_quick_profile_from_text(message: str) -> Dict:
    """Heuristic extraction for local testing without an LLM."""
    msg = message.lower()

    exp = None
    if "beginner" in msg:
        exp = "beginner"
    elif "advanced" in msg:
        exp = "advanced"
    elif "intermediate" in msg:
        exp = "intermediate"

    env = None
    if "indoor" in msg:
        env = "indoor"
    elif "outdoor" in msg:
        env = "outdoor"

    # light (handle "medium light" / "moderate light" / "medium")
    light = None
    if re.search(r"\b(high\s+light|full\s*sun)\b", msg):
        light = "high"
    elif re.search(r"\b(low\s+light|shade)\b", msg):
        light = "low"
    elif re.search(r"\b(medium\s+light|moderate\s+light|medium\b)\b", msg):
        light = "medium"

    m = re.search(r"(\d+)\s?(?:eur|€)?", msg)
    budget = float(m.group(1)) if m else None

    species = None
    for s in [
        "ficus", "ulmus", "zelkova", "ligustrum", "juniperus", "podocarpus",
        "sageretia", "carmona", "serissa", "portulacaria", "crassula",
        "cotoneaster", "pyracantha", "acer palmatum", "acer"
    ]:
        if s in msg:
            species = "Acer palmatum" if "acer" in s else s.title()
            break

    profile = {}
    if exp:
        profile["experience_level"] = exp
    if env:
        profile["preferred_environment"] = env
    if light:
        profile["available_light"] = light
    if budget is not None:
        profile["budget_eur"] = budget
    if species:
        profile["species_hint"] = species
    return profile


# --------- LLM Stub ---------

def simulate_llm_intake(message: str) -> BuyerProfile:
    """Fallback intake used when no provider is configured/available."""
    hints = extract_quick_profile_from_text(message)
    return BuyerProfile(
        experience_level=hints.get("experience_level", "beginner"),
        budget_eur=float(hints.get("budget_eur", 150)),
        preferred_environment=hints.get("preferred_environment", "either"),
        available_light=hints.get("available_light", "medium"),
        maintenance_time="low",
        location="Lisboa, PT",
        min_temperature_c=8.0,
        notes="auto-generated by simulate_llm_intake",
    )


# --------- LLM Intake (Ollama) ---------

def llm_intake_ollama(user_message: str) -> BuyerProfile:
    """
    Use Ollama to convert a vague user message into a structured BuyerProfile.
    Latency reductions:
      - num_predict=128 (shorter output)
      - temperature=0.0, top_p=0.9 (stable/shorter)
      - keep_alive="10m" (keep model warm)
      - 60s watchdog timeout with graceful fallback
    """
    system_prompt = _load_system_prompt()
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    timeout_s = int(os.getenv("OLLAMA_TIMEOUT_S", "60"))
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "128"))

    with mlflow.start_run(run_name="llm-intake", nested=True):
        mlflow.set_tag("provider", "ollama")
        mlflow.log_param("llm_provider", "ollama")
        mlflow.log_param("llm_model", model)
        mlflow.log_param("prompt_version", "v1")

        out_dir = Path("artifacts")
        out_dir.mkdir(exist_ok=True)
        (out_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
        mlflow.log_artifact(str(out_dir / "system_prompt.txt"))

        if OllamaClient is None:
            (out_dir / "llm_error.txt").write_text("Python 'ollama' client not installed.", encoding="utf-8")
            mlflow.log_artifact(str(out_dir / "llm_error.txt"))
            mlflow.set_tag("fallback", "simulate_llm_intake(no_client)")
            profile = simulate_llm_intake(user_message)
            mlflow.log_metric("llm_latency_ms", 0)
            mlflow.log_metric("llm_timeout", 0)
            return profile

        client = OllamaClient(host=host)

        def _call() -> str:
            resp = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                options={
                    "temperature": 0.0,
                    "top_p": 0.9,
                    "num_predict": num_predict,
                },
                format="json",
                keep_alive="10m",
            )
            return resp.get("message", {}).get("content", "")

        t0 = time.time()
        raw: Optional[str] = None
        timed_out = 0

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_call)
                raw = fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            timed_out = 1
        except Exception as e:
            (out_dir / "llm_error.txt").write_text(str(e), encoding="utf-8")
            mlflow.log_artifact(str(out_dir / "llm_error.txt"))

        latency_ms = int((time.time() - t0) * 1000)
        mlflow.log_metric("llm_latency_ms", latency_ms)
        if timed_out:
            mlflow.log_metric("llm_timeout", 1)

        if not raw:
            mlflow.set_tag("fallback", "simulate_llm_intake(timeout_or_error)")
            profile = simulate_llm_intake(user_message)
            return profile

        # Persist raw output for audit/debug
        (out_dir / "raw_llm_output.txt").write_text(raw, encoding="utf-8")
        mlflow.log_artifact(str(out_dir / "raw_llm_output.txt"))

        # Parse JSON safely
        try:
            data: Dict[str, Any] = json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0)) if m else {}

        profile = BuyerProfile(
            experience_level=data.get("experience_level", "beginner"),
            budget_eur=float(data.get("budget_eur", 150)),
            preferred_environment=data.get("preferred_environment", "either"),
            available_light=data.get("available_light", "unsure"),
            maintenance_time=data.get("maintenance_time", "unsure"),
            location=data.get("location", "unknown"),
            min_temperature_c=data.get("min_temperature_c"),
            notes=data.get("notes"),
        )
        return profile


def build_profile_via_provider(user_message: str) -> BuyerProfile:
    """Select which intake to use based on LLM_PROVIDER env var."""
    provider = (os.getenv("LLM_PROVIDER", "stub") or "stub").lower()
    if provider == "ollama":
        return llm_intake_ollama(user_message)
    # (future-ready) elif provider == "openai": return llm_intake_openai(user_message)
    return simulate_llm_intake(user_message)


# --------- Orchestrator ---------

class ChatOrchestrator:
    def __init__(self, species_csv: str, marketplace_csv: str):
        self.species_csv = species_csv
        self.marketplace_csv = marketplace_csv
        self.species_scorer = SpeciesScorer(species_csv)
        self.listing_ranker = ListingRanker(marketplace_csv)

    def _has_minimum_direct_fields(self, msg_hints: Dict) -> bool:
        required = {"experience_level", "budget_eur", "preferred_environment", "available_light"}
        return required.issubset(set(msg_hints.keys()))

    def handle_message(self, user_message: str, k: int = 5) -> Dict:
        t0 = time.time()
        mlflow.set_experiment("bonsai-chat-orchestrator")

        with mlflow.start_run(run_name="chat-session"):
            mlflow.set_tag("service", "bonsai-backend")
            mlflow.log_param("k", k)

            hints = extract_quick_profile_from_text(user_message)
            mode = "direct" if self._has_minimum_direct_fields(hints) else "guided"
            mlflow.log_param("mode", mode)
            mlflow.set_tag("scenario", mode)

            if mode == "direct":
                profile = BuyerProfile(
                    experience_level=hints["experience_level"],
                    budget_eur=float(hints["budget_eur"]),
                    preferred_environment=hints["preferred_environment"],
                    available_light=hints["available_light"],
                    maintenance_time="unsure",
                    location="unknown",
                )
                species_hint = hints.get("species_hint")
                recs = self.listing_ranker.rank(profile, k=k, species=species_hint)
                result = {
                    "mode": mode,
                    "buyer_profile": asdict(profile),
                    "species_candidates": (
                        [{"species": species_hint, "species_score": 1.0, "rationale": "user provided"}]
                        if species_hint else []
                    ),
                    "items": [asdict(x) for x in recs],
                }

            else:
                profile = build_profile_via_provider(user_message)
                species_candidates = self.species_scorer.score_species(profile, top_k=3)
                top_species = species_candidates[0]["species"] if species_candidates else None
                recs = self.listing_ranker.rank(profile, k=k, species=top_species)

                result = {
                    "mode": mode,
                    "buyer_profile": asdict(profile),
                    "species_candidates": species_candidates,
                    "items": [asdict(x) for x in recs],
                }

            latency_ms = int((time.time() - t0) * 1000)
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_metric("results_count", len(result["items"]))

            out_dir = Path("artifacts")
            out_dir.mkdir(exist_ok=True)
            (out_dir / "buyer_profile.json").write_text(
                json.dumps(result["buyer_profile"], indent=2), encoding="utf-8"
            )
            (out_dir / "species_candidates.json").write_text(
                json.dumps(result["species_candidates"], indent=2), encoding="utf-8"
            )
            (out_dir / "recommendations.json").write_text(
                json.dumps(result["items"], indent=2), encoding="utf-8"
            )

            mlflow.log_artifact(str(out_dir / "buyer_profile.json"))
            mlflow.log_artifact(str(out_dir / "species_candidates.json"))
            mlflow.log_artifact(str(out_dir / "recommendations.json"))

            return result


# --------- Local CLI demo ---------

if __name__ == "__main__":
    orch = ChatOrchestrator(
        species_csv=str(Path(__file__).resolve().parents[1] / "data" / "bonsai_species.csv"),
        marketplace_csv=str(Path(__file__).resolve().parents[1] / "data" / "bonsai_marketplace.csv"),
    )
    demo = orch.handle_message(
        "I am a beginner with 120 EUR, I prefer indoor and I have medium light. I like Ficus.",
        k=5,
    )
    print(json.dumps(demo, indent=2))
