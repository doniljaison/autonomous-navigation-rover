"""Dependency-free PNG encoding (the Pi runtime does not need Pillow/matplotlib)."""

import struct
import zlib

import numpy as np


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def png_bytes(img: np.ndarray, level: int = 3) -> bytes:
    """img: HxW uint8 (grey) or HxWx3 uint8 (RGB). Row 0 is the TOP of the image."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    if img.ndim == 2:
        h, w = img.shape
        ctype, rowbytes = 0, w
    else:
        h, w, _ = img.shape
        ctype, rowbytes = 2, w * 3
    raw = np.empty((h, rowbytes + 1), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = img.reshape(h, rowbytes)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) +
            _chunk(b"IDAT", zlib.compress(raw.tobytes(), level)) + _chunk(b"IEND", b""))


def map_rgb(log: np.ndarray, occ=0.6, free=-0.4, path_cells=None, flip=True) -> np.ndarray:
    """Grid (log[cy, cx], y up) -> RGB image with y up when flip=True."""
    img = np.full(log.shape + (3,), 128, dtype=np.uint8)
    img[log < free] = 255
    img[log > occ] = 0
    if path_cells is not None:
        cx, cy = path_cells
        ok = (cx >= 0) & (cy >= 0) & (cx < log.shape[1]) & (cy < log.shape[0])
        img[cy[ok], cx[ok]] = (230, 60, 120)
    return img[::-1] if flip else img
