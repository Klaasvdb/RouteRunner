from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from cache_manager import get_or_fetch_graph
from route_engine import find_route

app = FastAPI(title="RouteRunner")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


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

    try:
        result = await run_in_threadpool(find_route, req.lat, req.lon, target_m, prefs, G)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Route calculation failed: {exc}")

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No circular route found. Try a shorter distance or different location.",
        )

    return {
        "coordinates": result["coords"],
        "actual_distance_km": round(result["length"] / 1000, 2),
        "target_distance_km": req.distance_km,
        "candidates": result["candidates"],
        "evaluated": result["evaluated"],
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}
