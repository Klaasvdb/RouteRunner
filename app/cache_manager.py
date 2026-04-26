import os
import pickle
import time
import hashlib
from typing import Optional

import osmnx as ox
import networkx as nx

CACHE_DIR = "/app/cache"
CACHE_TTL = 7 * 24 * 3600  # 7 days


def _cache_key(lat: float, lon: float, radius: float) -> str:
    # Round to ~1km grid to maximise cache hits for nearby start points
    lat_r = round(lat * 100) / 100
    lon_r = round(lon * 100) / 100
    radius_r = round(radius / 500) * 500
    raw = f"{lat_r}_{lon_r}_{radius_r}"
    return hashlib.md5(raw.encode()).hexdigest()


def _paths(key: str) -> tuple[str, str]:
    return (
        os.path.join(CACHE_DIR, f"{key}.pkl"),
        os.path.join(CACHE_DIR, f"{key}.meta"),
    )


def _load(key: str) -> Optional[nx.MultiDiGraph]:
    pkl_path, meta_path = _paths(key)
    if not os.path.exists(pkl_path):
        return None

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            ts = float(f.read())
        if time.time() - ts > CACHE_TTL:
            os.remove(pkl_path)
            os.remove(meta_path)
            return None

    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _save(G: nx.MultiDiGraph, key: str) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    pkl_path, meta_path = _paths(key)
    with open(pkl_path, "wb") as f:
        pickle.dump(G, f)
    with open(meta_path, "w") as f:
        f.write(str(time.time()))


def get_or_fetch_graph(lat: float, lon: float, radius: float) -> nx.MultiDiGraph:
    key = _cache_key(lat, lon, radius)
    G = _load(key)
    if G is None:
        G = ox.graph_from_point(
            (lat, lon),
            dist=int(radius),
            network_type="walk",
            simplify=True,
        )
        _save(G, key)
    return G
