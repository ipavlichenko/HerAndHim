"""Telegram-compatible copies of the packaged catalogue portraits."""
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image

from .characters import character


@lru_cache(maxsize=20)
def portrait_jpeg(character_id: str) -> bytes:
    companion = character(character_id)
    path = Path(__file__).parents[1] / "web/static/catalog" / companion.portrait
    with Image.open(path) as source, BytesIO() as output:
        source.convert("RGB").save(output, format="JPEG", quality=90)
        return output.getvalue()
