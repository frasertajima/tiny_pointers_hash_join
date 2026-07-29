"""
hash_join — Python interface to the reliable 2-level tiny-pointer GPU hash-join engine
(hash_join_api.cuf / libhashjoin.so), same ctypes + CuPy bridge pattern as
../../MPDOK/mpdok_ops.py's MPDOKSolver -- not a new convention.

Honest scope, stated up front (see README.md for the full writeup): this only wins over a
plain CPU/GPU dict/hash-map at HIGH load factor (>=~95%) on a memory-tight, build-once/
probe-many table. At low/moderate load, plain linear probing (or a standard library hash
map) is simpler and just as fast or faster. Use this when you specifically need a compact,
high-load-factor, GPU-resident dictionary with a guaranteed-zero build-failure rate — not as
a general-purpose hash-map replacement.

Usage:
    from hash_join import HashJoinTable
    import cupy as cp

    keys = cp.asarray([101, 102, 103], dtype=cp.int64)
    values = cp.asarray([0, 1, 2], dtype=cp.int32)          # e.g. row ids

    table = HashJoinTable(n_keys=len(keys), loadfactor=0.90)
    table.build(keys, values)                                # raises if any key failed to insert

    queries = cp.asarray([101, 999, 103], dtype=cp.int64)    # 999 does not exist
    result = table.probe(queries)                             # cp.int32 array, -1 = no match
    table.close()                                             # or use as a context manager
"""

import ctypes
import os

import cupy as cp
import numpy as np

_DEFAULT_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libhashjoin.so")


class HashJoinTable:
    """A single GPU-resident reliable 2-level tiny-pointer table.

    Parameters
    ----------
    n_keys : int
        Number of distinct build keys you plan to insert. Used to size the table.
    B : int
        Bucket size. 16 is the default used throughout this codebase's tiny-pointer family.
    loadfactor : float
        Target primary-bucket load (e.g. 0.90). Lower = more memory, fewer overflow spills.
    ovf_frac : float
        Overflow region size as a fraction of n_keys (default 0.15, matching
        spacedict.cuf/kvpage.cuf/stabledict.cuf's own default). If `build()` reports any
        failures, increase this and rebuild — 0.15 comfortably covers a random key
        distribution's expected ~6% worst-case spill rate with headroom.
    lib_path : str, optional
        Path to libhashjoin.so (defaults to the copy built alongside this file).
    """

    def __init__(self, n_keys, B=16, loadfactor=0.90, ovf_frac=0.15, lib_path=None):
        self._lib = ctypes.CDLL(lib_path or _DEFAULT_LIB)

        self._lib.hje_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.hje_build.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.hje_probe.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
        ]
        self._lib.hje_destroy.argtypes = [ctypes.c_int]

        handle = ctypes.c_int(-1)
        info = ctypes.c_int(-1)
        self._lib.hje_create(
            int(n_keys), int(B), float(loadfactor), float(ovf_frac),
            ctypes.byref(handle), ctypes.byref(info),
        )
        if info.value != 0:
            raise RuntimeError(
                "HashJoinTable: no free handle (max 8 tables open at once); "
                "close() an existing table first"
            )
        self._handle = handle.value
        self._built = False
        self._n_keys = int(n_keys)

    def build(self, keys, values):
        """Insert (keys, values) — both length-n arrays, keys int64, values int32.
        Raises RuntimeError if any key failed to insert (increase ovf_frac and retry)."""
        keys = cp.asarray(keys, dtype=cp.int64)
        values = cp.asarray(values, dtype=cp.int32)
        if keys.shape[0] != values.shape[0]:
            raise ValueError("keys and values must be the same length")
        n = keys.shape[0]

        nfail = ctypes.c_int(-1)
        self._lib.hje_build(
            self._handle,
            ctypes.c_void_p(keys.data.ptr),
            ctypes.c_void_p(values.data.ptr),
            int(n),
            ctypes.byref(nfail),
        )
        if nfail.value != 0:
            raise RuntimeError(
                f"HashJoinTable.build: {nfail.value}/{n} keys failed to insert "
                f"(table sized for n_keys={self._n_keys}; either this build has more keys "
                f"than that, or ovf_frac is too small for this key distribution — "
                f"see README.md's honest note on random vs. sequential key spill rates)"
            )
        self._built = True

    def probe(self, queries):
        """Look up query keys (int64 array). Returns an int32 CuPy array of the same
        length: the matched value, or -1 if the key was never inserted."""
        if not self._built:
            raise RuntimeError("HashJoinTable.probe: call build() first")
        queries = cp.asarray(queries, dtype=cp.int64)
        n = queries.shape[0]
        out = cp.empty(n, dtype=cp.int32)
        self._lib.hje_probe(
            self._handle,
            ctypes.c_void_p(queries.data.ptr),
            int(n),
            ctypes.c_void_p(out.data.ptr),
        )
        return out

    def close(self):
        if self._handle is not None:
            self._lib.hje_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        # best-effort; explicit close()/context-manager use is preferred
        try:
            self.close()
        except Exception:
            pass
