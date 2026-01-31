#!/usr/bin/env python3
"""
Simple aiohttp web server for the fursuit scanner PoC.
Serves static files from the static/ directory.
"""

import os
from pathlib import Path

from aiohttp import web

HOST = os.getenv("WEB_HOST", "0.0.0.0")
PORT = int(os.getenv("WEB_PORT", "8080"))

STATIC_DIR = Path(__file__).parent / "static"


async def index_handler(request: web.Request) -> web.FileResponse:
    """Serve index.html for the root path."""
    return web.FileResponse(STATIC_DIR / "index.html")


def create_app() -> web.Application:
    """Create and configure the aiohttp application."""
    app = web.Application()

    # Serve index.html at root
    app.router.add_get("/", index_handler)

    # Serve static files
    app.router.add_static("/static/", STATIC_DIR, name="static")

    return app


def main() -> None:
    """Run the web server."""
    if not STATIC_DIR.exists():
        STATIC_DIR.mkdir(parents=True)
        print(f"Created static directory: {STATIC_DIR}")

    app = create_app()
    print(f"Starting web server at http://{HOST}:{PORT}")
    print(f"Serving static files from: {STATIC_DIR}")
    web.run_app(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
