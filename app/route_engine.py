import random
from typing import Optional

import networkx as nx
import numpy as np
import osmnx as ox

# Lower multiplier = more desirable for running
_HIGHWAY_COST: dict[str, float] = {
    "footway": 1.0,
    "path": 1.0,
    "track": 1.2,
    "bridleway": 1.3,
    "pedestrian": 1.4,
    "cycleway": 1.5,
    "steps": 2.5,
    "living_street": 2.5,
    "service": 3.5,
    "residential": 4.0,
    "unclassified": 4.5,
    "tertiary": 7.0,
    "tertiary_link": 7.0,
    "secondary": 15.0,
    "secondary_link": 15.0,
    "primary": 25.0,
    "primary_link": 25.0,
    "trunk": 100.0,
    "trunk_link": 100.0,
    "motorway": 10_000.0,
    "motorway_link": 10_000.0,
}

_SOFT_SURFACES = {
    "ground", "dirt", "grass", "gravel", "unpaved", "compacted",
    "fine_gravel", "sand", "earth", "woodchips", "mud", "pebblestone",
}
_HARD_SURFACES = {
    "asphalt", "concrete", "paved", "cobblestone", "sett",
    "paving_stones", "metal",
}

_NATURE_HIGHWAYS = {"footway", "path", "track", "bridleway"}
_TRAFFIC_HIGHWAYS = {
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "trunk", "trunk_link", "motorway", "motorway_link",
}


def _edge_weight(data: dict, prefs: dict) -> float:
    highway = data.get("highway", "unclassified")
    surface = data.get("surface", None)
    length = float(data.get("length", 1.0))

    if isinstance(highway, list):
        highway = highway[0]
    highway = str(highway)

    cost = _HIGHWAY_COST.get(highway, 5.0)

    avoid_traffic = prefs.get("avoid_traffic", 0.5)
    if highway in _TRAFFIC_HIGHWAYS:
        cost *= 1.0 + avoid_traffic * 4.0

    avoid_paved = prefs.get("avoid_paved", 0.5)
    if isinstance(surface, list):
        surface = surface[0]
    if surface in _SOFT_SURFACES:
        cost *= max(0.3, 1.0 - avoid_paved * 0.6)
    elif surface in _HARD_SURFACES:
        cost *= 1.0 + avoid_paved * 1.0

    prefer_nature = prefs.get("prefer_nature", 0.5)
    if highway in _NATURE_HIGHWAYS:
        cost *= max(0.2, 1.0 - prefer_nature * 0.7)

    return cost * length


def _apply_weights(G: nx.MultiDiGraph, prefs: dict) -> None:
    for u, v, k, data in G.edges(keys=True, data=True):
        G[u][v][k]["w"] = _edge_weight(data, prefs)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _path_stats(G: nx.MultiDiGraph, path: list) -> tuple[float, float]:
    """Return (total_length_m, total_weight)."""
    total_len = 0.0
    total_w = 0.0
    for i in range(len(path) - 1):
        edge = G.get_edge_data(path[i], path[i + 1])
        if not edge:
            continue
        lengths = [d.get("length", 0) for d in edge.values()]
        weights = [d.get("w", 1e9) for d in edge.values()]
        total_len += min(lengths)
        total_w += min(weights)
    return total_len, total_w


def _diverse_return(
    G: nx.MultiDiGraph, mid: int, start: int, outbound: list
) -> list:
    """Find return path that avoids heavily reusing outbound edges."""
    touched: list[tuple[int, int, int, float]] = []
    for i in range(len(outbound) - 1):
        u, v = outbound[i], outbound[i + 1]
        if not G.has_edge(u, v):
            continue
        for k in G[u][v]:
            orig = G[u][v][k].get("w", 1.0)
            G[u][v][k]["w"] = orig * 5.0
            touched.append((u, v, k, orig))

    try:
        path = nx.shortest_path(G, mid, start, weight="w")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        path = []
    finally:
        for u, v, k, orig in touched:
            G[u][v][k]["w"] = orig

    return path


def find_route(
    lat: float,
    lon: float,
    target_distance_m: float,
    prefs: dict,
    G: nx.MultiDiGraph,
) -> Optional[dict]:
    _apply_weights(G, prefs)

    start = ox.distance.nearest_nodes(G, lon, lat)
    s_lat = G.nodes[start]["y"]
    s_lon = G.nodes[start]["x"]

    half = target_distance_m / 2.0

    # Bucket candidates into 8 directional sectors for route diversity
    n_sectors = 8
    sectors: list[list[int]] = [[] for _ in range(n_sectors)]

    for node, d in G.nodes(data=True):
        if node == start:
            continue
        dist = _haversine(s_lat, s_lon, d["y"], d["x"])
        if half * 0.25 < dist < half * 1.3:
            bearing = np.degrees(np.arctan2(d["x"] - s_lon, d["y"] - s_lat)) % 360
            sector = int(bearing / (360 / n_sectors)) % n_sectors
            sectors[sector].append(node)

    candidates: list[int] = []
    for sector in sectors:
        if sector:
            candidates.extend(random.sample(sector, min(6, len(sector))))

    if not candidates:
        return None

    candidate_coords = [
        [G.nodes[n]["y"], G.nodes[n]["x"]] for n in candidates
    ]

    best_path: Optional[list] = None
    best_score = float("inf")
    evaluated: list[dict] = []

    for mid in candidates:
        try:
            path_out = nx.shortest_path(G, start, mid, weight="w")
            path_back = _diverse_return(G, mid, start, path_out)
            if not path_back:
                continue

            full = path_out + path_back[1:]
            actual_len, total_w = _path_stats(G, full)

            if actual_len == 0:
                continue

            deviation = abs(actual_len / target_distance_m - 1.0) * 3.0
            avg_cost = total_w / actual_len / 10.0
            score = deviation + avg_cost

            evaluated.append({
                "coords": [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in full],
                "score": round(score, 4),
                "length_km": round(actual_len / 1000, 2),
            })

            if score < best_score:
                best_score = score
                best_path = full

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    if best_path is None:
        return None

    # Sort worst→best so the animation reveals the winner last
    evaluated.sort(key=lambda x: x["score"], reverse=True)

    best_coords = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in best_path]
    best_len, _ = _path_stats(G, best_path)

    return {
        "coords": best_coords,
        "length": best_len,
        "candidates": candidate_coords,
        "evaluated": evaluated,
    }
