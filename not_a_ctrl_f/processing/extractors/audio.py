"""Extrator de áudio: janelas deslizantes mono em WAV."""

import io
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import BaseExtractor, ExtractError, Extraction, Piece, SkipFile

AUDIO_WINDOW = 30.0
AUDIO_HOP = 25.0
MAX_AUDIO_WINDOWS = 240
AUDIO_EXTS = frozenset("wav mp3 flac ogg".split())


class AudioExtractor(BaseExtractor):
    extensions = AUDIO_EXTS

    def read(self, path: Path) -> Extraction:
        with open(path, "rb") as fh:  # OSError propaga
            try:
                f = sf.SoundFile(fh)
            except (sf.LibsndfileError, RuntimeError) as e:
                raise ExtractError(f"áudio ilegível: {e}") from None
            with f:
                sr, ch, frames = f.samplerate, f.channels, f.frames
                duration = frames / sr if sr else 0.0
                if duration <= 0:
                    raise SkipFile("vazio")
                pieces: list[Piece] = []
                try:
                    for k in range(MAX_AUDIO_WINDOWS):
                        start = k * AUDIO_HOP
                        if start >= duration:
                            break
                        a = int(round(start * sr))
                        b = min(int(round((start + AUDIO_WINDOW) * sr)), frames)
                        f.seek(a)
                        data = f.read(b - a, dtype="float32", always_2d=True)
                        mono = np.asarray(data.mean(axis=1), dtype="float32")
                        out = io.BytesIO()
                        sf.write(out, mono, sr, format="WAV", subtype="PCM_16")
                        pieces.append(
                            Piece(
                                "audio", 1 + k, None, start,
                                min(start + AUDIO_WINDOW, duration), "audio", out.getvalue(),
                            )
                        )
                        if start + AUDIO_WINDOW >= duration:
                            break
                except (sf.LibsndfileError, RuntimeError) as e:
                    raise ExtractError(f"áudio ilegível: {e}") from None
        return Extraction(
            pieces, {"duration": round(duration, 3), "sample_rate": sr, "channels": ch}
        )
