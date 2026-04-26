from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from cache_manager import get_or_fetch_graph
from route_engine import find_route

app = FastAPI(title="RouteRunner")
templates = Jinja2Templates(directory="templates")


class RouteRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    distance_km: float = Field(..., ge=1, le=50)
    avoid_paved: float = Field(0.5, ge=0, le=1)
    prefer_nature: float = Field(0.5, ge=0, le=1)
    avoid_traffic: float = Field(0.5, ge=0, le=1)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/route")
async def calculate_route(req: RouteRequest):
    target_m = req.distance_km * 1000
    # Fetch radius: slightly larger than half the route so midpoints are reachable
    radius = min(target_m * 0.65, 10_000)

    try:
        G = await run_in_threadpool(get_or_fetch_graph, req.lat, req.lon, radius)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch map data: {exc}")

    prefs = {
        "avoid_paved": req.avoid_paved,
        "prefer_nature": req.prefer_nature,
        "avoid_traffic": req.avoid_traffic,
    }

    result = await run_in_threadpool(find_route, req.lat, req.lon, target_m, prefs, G)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No circular route found. Try a shorter distance or different location.",
        )

    coords, actual_m = result
    return {
        "coordinates": coords,
        "actual_distance_km": round(actual_m / 1000, 2),
        "target_distance_km": req.distance_km,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}
