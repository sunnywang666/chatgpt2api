from __future__ import annotations

from contextlib import asynccontextmanager
import os
from threading import Event

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api import accounts, ai, external_images, image_tasks, owned_accounts, system
from api.errors import install_exception_handlers
from api.support import resolve_web_asset, start_limited_account_watcher
from services.backup_service import backup_service
from services.config import config
from services.image_service import start_image_cleanup_scheduler


def create_app() -> FastAPI:
    app_version = config.app_version

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        stop_event = Event()
        thread = start_limited_account_watcher(stop_event)
        cleanup_thread = start_image_cleanup_scheduler(stop_event)
        backup_service.start()
        config.cleanup_old_images()
        try:
            yield
        finally:
            stop_event.set()
            thread.join(timeout=1)
            cleanup_thread.join(timeout=1)
            backup_service.stop()

    app = FastAPI(title="chatgpt2api", version=app_version, lifespan=lifespan)
    install_exception_handlers(app)
    # Keep the image boundary outside the routers so a public marker can never
    # reach an unallowlisted route. Register it before CORS; Starlette builds
    # middleware from the reverse registration order, making CORS handle
    # browser preflights before the boundary evaluates the actual request.
    app.middleware("http")(external_images.external_image_boundary)

    configured_origins = os.getenv("CHATGPT2API_CORS_ORIGINS")
    if configured_origins is None:
        configured_origins = config.data.get("cors_origins", [])
    if isinstance(configured_origins, str):
        cors_origins = [item.strip() for item in configured_origins.split(",") if item.strip()]
    elif isinstance(configured_origins, (list, tuple)):
        cors_origins = [str(item).strip() for item in configured_origins if str(item).strip()]
    else:
        cors_origins = []
    cors_origins = [origin for origin in cors_origins if origin != "*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "HEAD", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Workbench-Image-Client", "X-Workbench-Account-Owner"],
    )
    app.include_router(ai.create_router())
    app.include_router(accounts.create_router())
    app.include_router(image_tasks.create_router())
    app.include_router(owned_accounts.create_router())
    app.include_router(system.create_router(app_version))

    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def serve_web(full_path: str):
        asset = resolve_web_asset(full_path)
        if asset is not None:
            return FileResponse(asset)
        if full_path.strip("/").startswith("_next/"):
            raise HTTPException(status_code=404, detail="Not Found")
        fallback = resolve_web_asset("")
        if fallback is None:
            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(fallback)

    return app
