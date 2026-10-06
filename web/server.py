#!/usr/bin/env python3
from __future__ import annotations

"""
Server entrypoint for the SafeLens backend.

Preferred usage (from repo root):
  uv run uvicorn web.server:app --reload
or
  uv run -m web.server

This file also supports being executed directly (e.g., `python web/server.py`)
by falling back to appending the repository root to sys.path if needed.
"""


import torch
# PyTorch 2.6+ compatibility for Pyannote Audio / WhisperX checkpoints
try:
    import omegaconf
    import omegaconf.listconfig
    import omegaconf.dictconfig
    import omegaconf.base
    import omegaconf.nodes
    torch.serialization.add_safe_globals([
        omegaconf.listconfig.ListConfig,
        omegaconf.dictconfig.DictConfig,
        omegaconf.base.ContainerMetadata,
        omegaconf.base.Metadata,
        omegaconf.nodes.AnyNode,
        omegaconf.nodes.ValueNode,
        omegaconf.nodes.StringNode,
        omegaconf.nodes.IntegerNode,
        omegaconf.nodes.FloatNode,
        omegaconf.nodes.BooleanNode,
    ])
except Exception:
    pass

_orig_torch_load = torch.load
def _compat_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _compat_torch_load

# Ensure ffmpeg binary is discoverable in PATH
try:
    import imageio_ffmpeg
    ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass


from pathlib import Path
import sys

# Ensure repository root is in sys.path
root_dir = str(Path(__file__).resolve().parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from web.logging_config import configure_logging
from web.api import app


if __name__ == "__main__":
    import uvicorn
    import logging
    import argparse
    import os

    configure_logging()
    logger = logging.getLogger(__name__)

    # Allow host/port to be provided via CLI or environment
    parser = argparse.ArgumentParser(description="Run SafeLens backend server")
    parser.add_argument(
        "--host", default=os.getenv("HOST", "0.0.0.0"), help="Bind address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Bind port",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.getenv("RELOAD", "false").lower() in {"1", "true", "yes"},
        help="Enable auto-reload (development)",
    )
    args = parser.parse_args()

    logger.info("Starting FastAPI video upload server...")
    upload_folder = Path("./videos")
    allowed_extensions = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".flv", ".wmv"}
    max_file_size = 500 * 1024 * 1024  # 500MB

    logger.info(f"Bind: {args.host}:{args.port} (reload={args.reload})")
    logger.info(f"Upload folder: {upload_folder.absolute()}")
    logger.info(f"Allowed extensions: {allowed_extensions}")
    logger.info(f"Max file size: {max_file_size // (1024 * 1024)}MB")

    uvicorn.run(
        app, host=args.host, port=args.port, reload=args.reload, log_config=None
    )
