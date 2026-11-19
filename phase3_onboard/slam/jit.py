"""numba shim: the same code runs un-jitted when numba is missing (correctness on
any PC), and jitted + cached on the Pi.  Set ROVER_NO_JIT=1 to force pure Python."""

import os

HAVE_NUMBA = False
if os.environ.get("ROVER_NO_JIT", "0") != "1":
    try:
        import numba  # noqa: F401
        HAVE_NUMBA = True
    except ImportError:
        HAVE_NUMBA = False


def njit(*args, **kwargs):
    if HAVE_NUMBA:
        import numba
        kwargs.setdefault("cache", True)
        kwargs.setdefault("nogil", True)
        return numba.njit(*args, **kwargs)

    def deco(f):
        return f
    if args and callable(args[0]):
        return args[0]
    return deco
