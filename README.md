# RouteRunner

A self-hosted web app that calculates circular running routes optimised for soft surfaces, forests, and nature trails — using live OpenStreetMap data.

## Features

- Click-to-set start location on an interactive map
- Target distance slider (1–50 km)
- Three preference sliders:
  - **Soft surfaces** — prefer dirt, grass, gravel over asphalt
  - **Forests & nature trails** — boost footpaths, bridleways, and tracks
  - **Avoid traffic** — penalise busy primary/secondary roads
- OSM graph cache (7-day TTL) stored in a Docker volume — fast repeat requests
- Circular route algorithm samples midpoints in all compass directions for variety

## Quick Start (Docker Compose)

```bash
git clone https://github.com/your-username/RouteRunner.git
cd RouteRunner
docker compose up -d --build
```

Open `http://localhost:8765` in your browser.

The `cache/` folder is created automatically and persists OSM data between restarts.

## Unraid Setup

### Option A — Docker Compose (recommended)

1. Install the **Compose Manager** plugin from Unraid Community Applications.
2. Clone or upload this repo to a share (e.g. `/mnt/user/appdata/routerunner`).
3. In Compose Manager, point it at the folder and click **Start**.

### Option B — Manual Docker

```bash
docker build -t routerunner .
docker run -d \
  --name routerunner \
  -p 8765:8000 \
  -v /mnt/user/appdata/routerunner/cache:/app/cache \
  --restart unless-stopped \
  routerunner
```

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `TZ` | `Europe/Amsterdam` | Timezone for logs |

To change the host port, edit `docker-compose.yml`:

```yaml
ports:
  - "YOUR_PORT:8000"
```

## Project Structure

```
RouteRunner/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── app/
    ├── main.py            # FastAPI endpoints
    ├── route_engine.py    # OSM graph scoring & circular route algorithm
    ├── cache_manager.py   # Pickle-based OSM graph cache
    └── templates/
        └── index.html     # Leaflet map UI
```

## How It Works

1. On first request for an area, `osmnx` fetches the pedestrian network from OpenStreetMap and caches it as a pickle file.
2. Every edge in the graph is scored based on highway type, surface material, and the user's preference sliders — lower score = more desirable.
3. The engine samples candidate midpoints in 8 compass directions at roughly half the target distance away.
4. For each candidate: find the weighted-shortest path out, then a *diverse* return path (outbound edges are penalised to encourage a loop rather than an out-and-back).
5. The loop with the best combination of route quality and distance accuracy is returned.
