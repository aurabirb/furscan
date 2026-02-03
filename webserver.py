"""Aiohttp web server for serving static files."""

import os
from pathlib import Path

from aiohttp import web

# Configuration (can be overridden by environment variables)
HOST = os.getenv("WEB_HOST", "0.0.0.0")
PORT = int(os.getenv("WEB_PORT", "8080"))
STATIC_DIR = Path(__file__).parent / "static"


async def index_handler(request: web.Request) -> web.FileResponse:
    """Serve index.html for the root path."""
    return web.FileResponse(STATIC_DIR / "index.html")


def create_app(static_dir: Path = None) -> web.Application:
    """Create and configure the aiohttp web application.

    Args:
        static_dir: Optional path to static files directory. Defaults to ./static/
    """
    static = static_dir or STATIC_DIR
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_static("/static/", static, name="static")
    return app


async def start_server(runner: web.AppRunner, host: str = None, port: int = None) -> None:
    """Start the web server asynchronously.

    Args:
        runner: The aiohttp AppRunner to use
        host: Host to bind to. Defaults to WEB_HOST env var or 0.0.0.0
        port: Port to bind to. Defaults to WEB_PORT env var or 8080
    """
    await runner.setup()
    site = web.TCPSite(runner, host or HOST, port or PORT)
    await site.start()
    print(f"Web server running at http://{host or HOST}:{port or PORT}")


def run_standalone() -> None:
    """Run the web server in standalone mode (blocking)."""
    if not STATIC_DIR.exists():
        STATIC_DIR.mkdir(parents=True)
        print(f"Created static directory: {STATIC_DIR}")

    app = create_app()
    print(f"Starting web server at http://{HOST}:{PORT}")
    print(f"Serving static files from: {STATIC_DIR}")
    web.run_app(app, host=HOST, port=PORT)


if __name__ == "__main__":
    run_standalone()
