#!/usr/bin/env python3
"""
Autonomous Flight Sentinel — Dashboard Server
==============================================
Serves the dashboard UI on http://localhost:8765

Endpoints:
  GET /              → index.html
  GET /api/status    → mission state JSON
  GET /api/photos    → categorised photo list
  GET /api/events    → SSE stream (log lines + state changes)
  GET /photos/...    → static image files from output/
"""

import http.server
import json
import os
import re
import glob
import time
import threading
import socketserver
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

# ── Paths ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
FLIGHT_DIR = BASE_DIR.parent / "autonomousflight"
OUTPUT_DIR = FLIGHT_DIR / "output"
LOG_FILE   = OUTPUT_DIR / "dashboard.log"
PORT       = 8765

# ── Mission state ──────────────────────────────────────
_state = {
    "state": "IDLE",          # IDLE | MONITORING | ALERT | FLYING | MISSION_COMPLETE
    "bearing": None,
    "fire_location": None,
    "drone_position": None,
    "alert_message": None,
    "mission_start": None,
    "elapsed": 0,
    "photo_count": 0,
    "last_update": time.time(),
}
_state_lock = threading.Lock()

# ── SSE clients ────────────────────────────────────────
_sse_clients = []
_sse_lock    = threading.Lock()

# ─────────────────────────────────────────────────────
# LOG WATCHER
# ─────────────────────────────────────────────────────
_log_pos = 0

def _parse_line(line: str):
    """Update state from a single log line."""
    with _state_lock:
        s = _state

        # State transitions
        if "CONTINUOUS MONITORING" in line:
            s["state"] = "MONITORING"
        
        if re.search(r"ALARM|SMOKE ALERT|🚨|SMOKE DETECTED", line, re.I):
            s["state"]         = "ALERT"
            s["alert_message"] = line.strip()
        
        if re.search(r"DISPATCHING DRONE|PHASE A — BEARING|Taking off|Takeoff successful", line, re.I):
            if s["state"] != "MISSION_COMPLETE":
                s["state"] = "FLYING"
                if not s["mission_start"]:
                    s["mission_start"] = time.time()
        
        # Mission completion patterns
        if re.search(r"MISSION REPORT|Mission complete|Landed safely|MISSION FULLY COMPLETE|🏁 Mission complete", line, re.I):
            s["state"] = "MISSION_COMPLETE"
        
        if re.search(r"fire_confirmed.*True|FIRE CONFIRMED|🔥✅ FIRE CONFIRMED", line, re.I):
            s["state"] = "MISSION_COMPLETE"

        # Bearing
        m = re.search(r"[Bb]earing[:\s]+([0-9.]+)", line)
        if m:
            s["bearing"] = float(m.group(1))

        # Drone position
        m = re.search(r"Pos:\s*\(([0-9.\-]+),\s*([0-9.\-]+)\)", line)
        if m:
            s["drone_position"] = {"x": float(m.group(1)), "y": float(m.group(2))}

        # Fire location
        m = re.search(r"Fire.*?\(([0-9.\-]+),\s*([0-9.\-]+)\)", line, re.IGNORECASE)
        if m:
            s["fire_location"] = {"x": float(m.group(1)), "y": float(m.group(2))}

        # Update elapsed
        if s["mission_start"]:
            s["elapsed"] = time.time() - s["mission_start"]
        s["last_update"] = time.time()


def _watch_log():
    global _log_pos
    while True:
        try:
            if LOG_FILE.exists():
                with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(_log_pos)
                    lines = f.readlines()
                    _log_pos = f.tell()
                for ln in lines:
                    ln = ln.rstrip()
                    if ln:
                        _parse_line(ln)
                        _broadcast({"type": "log", "message": ln, "ts": time.time()})
        except Exception:
            pass
        time.sleep(0.4)


def _watch_output():
    """Detect new PNGs in output/ → escalate state."""
    known: set = set()
    while True:
        try:
            current = set(OUTPUT_DIR.glob("*.png")) | set(OUTPUT_DIR.glob("*.jpg"))
            new = current - known
            if new:
                with _state_lock:
                    if _state["state"] == "IDLE":
                        _state["state"] = "MONITORING"
                    alarm = [f for f in new if "ALARM" in f.name.upper()]
                    if alarm:
                        _state["state"] = "ALERT"
                        _state["alert_message"] = f"{len(alarm)} alarm photo(s) detected"
                    _state["photo_count"] = len(current)
                for f in new:
                    _broadcast({"type": "new_photo", "name": f.name, "ts": time.time()})
            known = current
        except Exception:
            pass
        time.sleep(2)


def _check_mission_reports():
    """Read last mission report JSON for fire location."""
    reports = sorted(glob.glob(str(OUTPUT_DIR / "mission" / "mission_report_*.json")))
    if not reports:
        return
    try:
        with open(reports[-1]) as f:
            r = json.load(f)
        if r.get("fire_confirmed"):
            fp = r.get("final_position")
            if fp and len(fp) >= 2:
                with _state_lock:
                    _state["fire_location"] = {"x": fp[0], "y": fp[1]}
                    _state["state"] = "MISSION_COMPLETE"
    except Exception:
        pass


def _broadcast(data: dict):
    msg = f"data: {json.dumps(data)}\n\n".encode()
    with _sse_lock:
        dead = []
        for c in _sse_clients:
            try:
                c.wfile.write(msg)
                c.wfile.flush()
            except Exception:
                dead.append(c)
        for c in dead:
            _sse_clients.remove(c)


# ─────────────────────────────────────────────────────
# PHOTO CATALOGUE
# ─────────────────────────────────────────────────────
def _get_photos() -> dict:
    out = {"tower": [], "drone": [], "mission": []}
    try:
        for pat in ("MONITOR_*.png", "BASELINE_*.png", "ALARM_*.png"):
            for f in sorted(OUTPUT_DIR.glob(pat))[-20:]:
                out["tower"].append({"name": f.name, "url": f"/photos/output/{f.name}"})

        bdir = OUTPUT_DIR / "bearing_nav"
        if bdir.exists():
            for f in sorted(bdir.glob("*.png"))[-20:]:
                out["drone"].append({"name": f.name, "url": f"/photos/bearing_nav/{f.name}"})

        mdir = OUTPUT_DIR / "mission"
        if mdir.exists():
            for f in sorted(mdir.glob("*.png"))[-10:]:
                out["mission"].append({"name": f.name, "url": f"/photos/mission/{f.name}", "kind": "image"})
    except Exception:
        pass
    return out


# ─────────────────────────────────────────────────────
# HTTP HANDLER
# ─────────────────────────────────────────────────────
class _Handler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            self._serve_file(BASE_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/api/status":
            _check_mission_reports()
            with _state_lock:
                self._json(_state.copy())
        elif path == "/api/photos":
            self._json(_get_photos())
        elif path == "/api/events":
            self._sse()
        elif path.startswith("/photos/"):
            self._serve_photo(path)
        else:
            # Serve static files from Dashboard directory (css, js, html chunks)
            local_path = BASE_DIR / path.lstrip("/")
            if local_path.is_file():
                ct = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
                self._serve_file(local_path, ct)
            else:
                self.send_error(404)

    # ── helpers ────────────────────────────────────────
    def _serve_file(self, p: Path, ct: str):
        try:
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", len(data))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_error(404)

    def _serve_photo(self, url_path: str):
        parts = url_path.lstrip("/").split("/", 2)   # ["photos", "output"|"bearing_nav"|"mission", "filename"]
        if len(parts) < 3:
            self.send_error(404); return
        _, section, name = parts
        if section == "output":
            fp = OUTPUT_DIR / name
        elif section == "bearing_nav":
            fp = OUTPUT_DIR / "bearing_nav" / name
        elif section == "mission":
            fp = OUTPUT_DIR / "mission" / name
        else:
            self.send_error(404); return
        if not fp.exists():
            self.send_error(404); return
        ct = mimetypes.guess_type(str(fp))[0] or "image/png"
        self._serve_file(fp, ct)

    def _json(self, data):
        body = json.dumps(data, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            with _state_lock:
                init = {"type": "init", "state": _state["state"]}
            self.wfile.write(f"data: {json.dumps(init)}\n\n".encode())
            self.wfile.flush()
        except Exception:
            return
        with _sse_lock:
            _sse_clients.append(self)
        try:
            while True:
                time.sleep(15)
                self.wfile.write(b"data: {\"type\":\"heartbeat\"}\n\n")
                self.wfile.flush()
        except Exception:
            pass
        finally:
            with _sse_lock:
                if self in _sse_clients:
                    _sse_clients.remove(self)

    def log_message(self, *_):
        pass   # silence access log


# ─────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("╔══════════════════════════════════════════╗")
    print("║  🔥 FIRE SENTINEL — DASHBOARD SERVER     ║")
    print(f"║  http://localhost:{PORT}                   ║")
    print(f"║  Watching: {OUTPUT_DIR.name}/               ║")
    print("╚══════════════════════════════════════════╝")

    threading.Thread(target=_watch_log,    daemon=True).start()
    threading.Thread(target=_watch_output, daemon=True).start()

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", PORT), _Handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n⛔ Dashboard server stopped.")


if __name__ == "__main__":
    main()
