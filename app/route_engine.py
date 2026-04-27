from typing import Optional
import networkx as nx
import numpy as np
import osmnx as ox

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

# Multiplier applied to edges already used in a previous segment of the
# triangle (start->A, A->B, B->start). Forces subsequent shortest-path
# searches to find genuinely different routes instead of overlapping the
# same edges in reverse.
_REUSE_PENALTY = 50.0


def _edge_weight(data: dict, prefs: dict) -> float:
    highway = data.get("highway", "unclassified")
    surface = data.get("surface", None)
    length = float(data.get("length", 1.0))

    if isinstance(highway, list):
        highway = highway[0]
    highway = str(highway)
    if isinstance(surface, list):
        surface = surface[0]

    cost = _HIGHWAY_COST.get(highway, 5.0)

    # --- Traffic avoidance ---
    avoid_traffic = prefs.get("avoid_traffic", 0.5)
    if highway in _TRAFFIC_HIGHWAYS:
        cost *= 1.0 + avoid_traffic * 4.0

    # --- Surface preference ---
    # When surface tag is missing on a nature highway, assume soft.
    # (Statistically reliable: untagged footway/path/track is rarely asphalt.)
    avoid_paved = prefs.get("avoid_paved", 0.5)
    effective_surface = surface
    if effective_surface is None and highway in _NATURE_HIGHWAYS:
        effective_surface = "ground"

    if effective_surface in _SOFT_SURFACES:
        cost *= max(0.3, 1.0 - avoid_paved * 0.6)
    elif effective_surface in _HARD_SURFACES:
        cost *= 1.0 + avoid_paved * 1.0

    # --- Nature preference ---
    # Aggressive: at prefer_nature=1.0, nature paths get 20x discount,
    # AND non-nature paths get 3x penalty. Combined effect ~60x preference.
    prefer_nature = prefs.get("prefer_nature", 0.5)
    if highway in _NATURE_HIGHWAYS:
        cost *= max(0.05, 1.0 - prefer_nature * 0.95)
    else:
        cost *= 1.0 + prefer_nature * 2.0

    return cost * length


def _apply_weights(G: nx.MultiDiGraph, prefs: dict) -> None:
    for u, v, k, data in G.edges(keys=True, data=True):
        G[u][v][k]["w"] = _edge_weight(data, prefs)


def _path_stats(G: nx.MultiDiGraph, path: list) -> tuple[float, float]:
    total_len = total_w = 0.0
    for i in range(len(path) - 1):
        edge = G.get_edge_data(path[i], path[i + 1])
        if not edge:
            continue
        total_len += min(d.get("length", 0) for d in edge.values())
        total_w   += min(d.get("w", 1e9)    for d in edge.values())
    return total_len, total_w


def _penalize_path(G: nx.MultiDiGraph, path: list, factor: float = _REUSE_PENALTY) -> None:
    """Multiply weights on edges used by `path` to discourage their reuse."""
    for u, v in zip(path[:-1], path[1:]):
        if G.has_edge(u, v):
            for k in G[u][v]:
                G[u][v][k]["w"] *= factor


def _restore_path(G: nx.MultiDiGraph, path: list, factor: float = _REUSE_PENALTY) -> None:
    """Undo `_penalize_path`."""
    for u, v in zip(path[:-1], path[1:]):
        if G.has_edge(u, v):
            for k in G[u][v]:
                G[u][v][k]["w"] /= factor


def _node_at_bearing(
    G: nx.MultiDiGraph,
    s_lat: float, s_lon: float,
    bearing_deg: float, distance_m: float,
) -> Optional[int]:
    """Nearest graph node to the point at `distance_m` in compass direction `bearing_deg`."""
    R = 6_371_000.0
    b = np.radians(bearing_deg)
    t_lat = s_lat + np.degrees(distance_m / R * np.cos(b))
    t_lon = s_lon + np.degrees(distance_m / R * np.sin(b) / np.cos(np.radians(s_lat)))
    try:
        return ox.distance.nearest_nodes(G, t_lon, t_lat)
    except Exception:
        return None


def _loop_penalty(coords: list) -> float:
    """
    Bounding-box aspect-ratio penalty.
    0 = square bounding box (proper loop).
    1 = infinitely elongated (straight out-and-back).
    """
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    lat_r = max(lats) - min(lats)
    lon_r = max(lons) - min(lons)
    if lat_r == 0 or lon_r == 0:
        return 1.0
    aspect = max(lat_r, lon_r) / min(lat_r, lon_r)
    return (aspect - 1.0) / (aspect + 1.0)


def _triangle_path(
    G: nx.MultiDiGraph,
    start: int,
    node_a: int,
    node_b: int,
) -> Optional[list]:
    """
    Find a triangle start -> A -> B -> start, penalizing edges between
    segments so that the three legs don't overlap. Restores the graph
    afterwards regardless of success.
    """
    p1 = p2 = None
    try:
        p1 = nx.shortest_path(G, start, node_a, weight="w")
        _penalize_path(G, p1)

        p2 = nx.shortest_path(G, node_a, node_b, weight="w")
        _penalize_path(G, p2)

        p3 = nx.shortest_path(G, node_b, start, weight="w")
        return p1 + p2[1:] + p3[1:]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    finally:
        if p1 is not None:
            _restore_path(G, p1)
        if p2 is not None:
            _restore_path(G, p2)


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

    best_path: Optional[list] = None
    best_score = float("inf")
    evaluated: list[dict] = []
    seen_candidates: set[int] = set()
    candidate_coords: list = []

    # Three leg sizes bracket detour ratios of ~1.2, ~1.5, ~2.0.
    # Actual trail paths are rarely straight, so Euclidean leg < real path leg.
    leg_factors = (0.27, 0.22, 0.17)

    for leg_factor in leg_factors:
        leg = target_distance_m * leg_factor
        for base in range(0, 360, 45):
            for spread in (+90, -90, +60, -60):
                node_a = _node_at_bearing(G, s_lat, s_lon, base % 360, leg)
                node_b = _node_at_bearing(G, s_lat, s_lon, (base + spread) % 360, leg)
                if node_a is None or node_b is None:
                    continue
                if len({start, node_a, node_b}) < 3:
                    continue

                for n in (node_a, node_b):
                    if n not in seen_candidates:
                        seen_candidates.add(n)
                        candidate_coords.append([G.nodes[n]["y"], G.nodes[n]["x"]])

                full = _triangle_path(G, start, node_a, node_b)
                if full is None:
                    continue

                actual_len, total_w = _path_stats(G, full)
                if actual_len == 0:
                    continue

                coords = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in full]
                deviation = abs(actual_len / target_distance_m - 1.0) * 3.0
                avg_cost  = total_w / actual_len / 10.0
                loop_pen  = _loop_penalty(coords) * 2.0
                score = deviation + avg_cost + loop_pen

                evaluated.append({
                    "coords": coords,
                    "score": round(score, 4),
                    "length_km": round(actual_len / 1000, 2),
                })

                if score < best_score:
                    best_score = score
                    best_path = full

    if best_path is None:
        return None

    # Sort worst → best so the animation reveals the winner last
    evaluated.sort(key=lambda x: x["score"], reverse=True)
    best_coords = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in best_path]
    best_len, _ = _path_stats(G, best_path)

    return {
        "coords": best_coords,
        "length": best_len,
        "candidates": candidate_coords,
        "evaluated": evaluated,
    }