"""Extrator de imagens: uma peça 'image' por arquivo."""

import io
from pathlib import Path

from PIL import Image, ImageOps

from .base import BaseExtractor, ExtractError, Extraction, Piece, SkipFile

MIN_IMAGE_SIDE = 256
MAX_IMAGE_SIDE = 2048
PHOTO_BUDGET = 140
IMAGE_EXTS = frozenset("jpg jpeg png webp bmp gif tif tiff".split())


class ImageExtractor(BaseExtractor):
    extensions = IMAGE_EXTS

    def read(self, path: Path) -> Extraction:
        data = path.read_bytes()  # OSError propaga; erros do Pillow viram ExtractError
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.seek(0)
                im.load()
                taken = None
                try:
                    taken = im.getexif().get_ifd(0x8769).get(36867)
                except Exception:
                    pass
                img = ImageOps.exif_transpose(im)
                if img is im:
                    img = im.copy()
            w, h = img.size
            if min(w, h) < MIN_IMAGE_SIDE:
                raise SkipFile(f"imagem pequena {w}×{h}")
            if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
                rgba = img.convert("RGBA")
                bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, rgba).convert("RGB")
            else:
                img = img.convert("RGB")
            big = max(w, h)
            if big > MAX_IMAGE_SIDE:
                k = MAX_IMAGE_SIDE / big
                img = img.resize(
                    (max(1, round(w * k)), max(1, round(h * k))), Image.Resampling.BICUBIC
                )
            buf = io.BytesIO()
            img.save(buf, "PNG")
        except SkipFile:
            raise
        except (Image.DecompressionBombError, OSError, ValueError, SyntaxError, EOFError) as e:
            raise ExtractError(f"imagem ilegível: {e}") from None
        info: dict[str, object] = {"width": w, "height": h}
        if taken:
            info["taken"] = str(taken)
        piece = Piece("image", 1, None, None, None, "image", buf.getvalue(), PHOTO_BUDGET)
        return Extraction([piece], info)
