"""Lightweight activation data server for mechinterp experiments.

Run locally to share activation data across multiple experiment runners,
avoiding redundant 4-5 minute load times per experiment.

Usage:
    uvicorn splatnlp.mechinterp.server.activation_server:app --port 8765

Or via CLI:
    python -m splatnlp.mechinterp.server.activation_server --port 8765
"""

import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Global state - loaded once, shared across requests
_state: dict = {}


class ActivationRequest(BaseModel):
    """Request for feature activations."""

    feature_id: int
    limit: int | None = None


class ActivationResponse(BaseModel):
    """Response with activation data."""

    feature_id: int
    n_rows: int
    columns: list[str]
    data: list[dict]
    load_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_type: str
    n_features: int
    uptime_seconds: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load database on startup, cleanup on shutdown."""
    start = time.time()
    logger.info("Loading activation database...")

    try:
        from splatnlp.dashboard.efficient_fs_database import EfficientFSDatabase
        from splatnlp.mechinterp.skill_helpers.context_loader import (
            ULTRA_MODEL_PATHS,
            load_context,
        )

        # Load the database
        db = EfficientFSDatabase(
            data_dir=ULTRA_MODEL_PATHS["data_dir"],
            examples_dir=ULTRA_MODEL_PATHS["examples_dir"],
        )

        # Load context for vocab access
        ctx = load_context("ultra")

        _state["db"] = db
        _state["ctx"] = ctx
        _state["model_type"] = "ultra"
        _state["start_time"] = time.time()
        _state["feature_cache"] = {}  # LRU-ish cache for recent features

        load_time = time.time() - start
        logger.info(f"Database loaded in {load_time:.1f}s")
        logger.info(f"Available features: {len(db.get_all_feature_ids())}")

    except Exception as e:
        logger.error(f"Failed to load database: {e}")
        raise

    yield

    # Cleanup
    _state.clear()
    logger.info("Server shutdown, state cleared")


app = FastAPI(
    title="Activation Server",
    description="Local server for sharing SAE activation data across experiments",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if "db" not in _state:
        raise HTTPException(503, "Database not loaded")

    return HealthResponse(
        status="healthy",
        model_type=_state["model_type"],
        n_features=len(_state["db"].get_all_feature_ids()),
        uptime_seconds=time.time() - _state["start_time"],
    )


@app.get("/features")
async def list_features():
    """List all available feature IDs."""
    if "db" not in _state:
        raise HTTPException(503, "Database not loaded")

    return {"feature_ids": _state["db"].get_all_feature_ids()}


@app.get("/activations/{feature_id}", response_model=ActivationResponse)
async def get_activations(
    feature_id: int,
    limit: int | None = None,
    sample_frac: float | None = None,
    include_abilities: bool = True,
):
    """Get activation data for a feature as JSON.

    Args:
        feature_id: The SAE feature ID
        limit: Optional limit on number of rows returned
        sample_frac: Optional random fraction (0-1) to downsample rows
        include_abilities: If False, skip ability_input_tokens for faster scans

    For larger responses, prefer /activations/{feature_id}/arrow.
    """
    if "db" not in _state:
        raise HTTPException(503, "Database not loaded")

    db = _state["db"]
    cache = _state["feature_cache"]

    # Check cache first
    cache_key = f"{feature_id}_{limit}_{sample_frac}_{include_abilities}"
    if cache_key in cache:
        cached = cache[cache_key]
        return ActivationResponse(
            feature_id=feature_id,
            n_rows=len(cached["data"]),
            columns=cached["columns"],
            data=cached["data"],
            load_time_ms=0.0,  # Cache hit
        )

    start = time.time()
    try:
        df = db.get_all_feature_activations_for_pagerank(
            feature_id,
            include_negative=True,
            limit=limit,
            sample_frac=sample_frac,
            include_abilities=include_abilities,
        )

        data = df.to_dicts()
        columns = list(df.columns)

        # Cache result (simple LRU - keep last 10 features)
        if len(cache) >= 10:
            oldest = next(iter(cache))
            del cache[oldest]
        cache[cache_key] = {"data": data, "columns": columns}

        load_time_ms = (time.time() - start) * 1000

        return ActivationResponse(
            feature_id=feature_id,
            n_rows=len(data),
            columns=columns,
            data=data,
            load_time_ms=load_time_ms,
        )

    except Exception as e:
        raise HTTPException(404, f"Feature {feature_id} not found: {e}")


@app.get("/activations/{feature_id}/arrow")
async def get_activations_arrow(
    feature_id: int,
    limit: int | None = None,
    sample_frac: float | None = None,
    include_abilities: bool = True,
):
    """Get activation data for a feature as Arrow IPC format.

    Args:
        feature_id: The SAE feature ID
        limit: Optional limit on number of rows returned
        sample_frac: Optional random fraction (0-1) to downsample rows
        include_abilities: If False, skip ability_input_tokens for faster scans

    Returns:
        Arrow IPC binary data with custom headers for metadata.
    """
    if "db" not in _state:
        raise HTTPException(503, "Database not loaded")

    db = _state["db"]
    arrow_cache = _state.get("arrow_cache", {})
    if "arrow_cache" not in _state:
        _state["arrow_cache"] = arrow_cache

    # Check Arrow cache first
    cache_key = f"{feature_id}_{limit}_{sample_frac}_{include_abilities}"
    if cache_key in arrow_cache:
        cached = arrow_cache[cache_key]
        return Response(
            content=cached["bytes"],
            media_type="application/vnd.apache.arrow.stream",
            headers={
                "X-Feature-Id": str(feature_id),
                "X-Num-Rows": str(cached["n_rows"]),
                "X-Load-Time-Ms": "0.0",
                "X-Cache-Hit": "true",
            },
        )

    start = time.time()
    try:
        df = db.get_all_feature_activations_for_pagerank(
            feature_id,
            include_negative=True,
            limit=limit,
            sample_frac=sample_frac,
            include_abilities=include_abilities,
        )

        # Serialize to Arrow IPC format
        buf = io.BytesIO()
        df.write_ipc(buf)
        arrow_bytes = buf.getvalue()
        n_rows = len(df)

        # Cache result (keep last 20 features in Arrow format)
        if len(arrow_cache) >= 20:
            oldest = next(iter(arrow_cache))
            del arrow_cache[oldest]
        arrow_cache[cache_key] = {"bytes": arrow_bytes, "n_rows": n_rows}

        load_time_ms = (time.time() - start) * 1000

        return Response(
            content=arrow_bytes,
            media_type="application/vnd.apache.arrow.stream",
            headers={
                "X-Feature-Id": str(feature_id),
                "X-Num-Rows": str(n_rows),
                "X-Load-Time-Ms": f"{load_time_ms:.1f}",
                "X-Cache-Hit": "false",
            },
        )

    except Exception as e:
        raise HTTPException(404, f"Feature {feature_id} not found: {e}")


@app.get("/context/vocab")
async def get_vocab():
    """Get the token vocabulary."""
    if "ctx" not in _state:
        raise HTTPException(503, "Context not loaded")

    ctx = _state["ctx"]
    return {
        "vocab": ctx.vocab,
        "weapon_vocab": ctx.weapon_vocab,
    }


@app.get("/context/inv_vocab")
async def get_inv_vocab():
    """Get the inverse vocabulary (ID -> token)."""
    if "ctx" not in _state:
        raise HTTPException(503, "Context not loaded")

    ctx = _state["ctx"]
    return {
        "inv_vocab": {str(k): v for k, v in ctx.inv_vocab.items()},
        "inv_weapon_vocab": {
            str(k): v for k, v in ctx.inv_weapon_vocab.items()
        },
    }


def main():
    """Run the server via CLI."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Activation data server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument(
        "--port", type=int, default=8765, help="Port to bind to"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print(f"Starting activation server on http://{args.host}:{args.port}")
    print("Endpoints:")
    print("  GET /health - Health check")
    print("  GET /features - List feature IDs")
    print("  GET /activations/{feature_id} - Get activation data")
    print("  GET /context/vocab - Get vocabularies")

    uvicorn.run(
        "splatnlp.mechinterp.server.activation_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
