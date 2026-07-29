# Hash-join engine — GPU hash JOIN built on the tiny-pointers `tinymap` module

Moved into its own subdirectory 2026-07-29: this grew from a single worked example
(`joindemo.cuf`) into a reusable module + benchmark + correctness demo + notebook, so it earned
its own space rather than living loosely alongside `../hashbench.cuf`, `../kvpage.cuf`, and the
other one-off application demos in the parent `tiny_pointers/` directory. Depends on the shared
`../tinymap.cuf` module (device-callable `ttd_find`/`ttd_insert`, the reliable 2-level
`ttd_find_r`/`ttd_insert_r`, and the linear-probe baseline `lpd_find`/`lpd_insert`) — see
`../README.md` for that module's own documentation and every other application built on it.

## Build & run

```
make            # builds joindemo, joindemo_reliable, join_example_small, libhashjoin.so
make csv        # runs the CLI benchmarks with the `csv` flag, writes the CSVs
make notebook   # csv + regenerates/executes HASH_JOIN_ENGINE.ipynb (benchmark notebook)
make howto      # builds libhashjoin.so + regenerates/executes HOW_TO_USE.ipynb (the packaged API)
make run        # runs the small worked example with human-readable output
```

## The packaged Python API (`hash_join.py` / `libhashjoin.so`, added 2026-07-29)

The CLI demos above are benchmarks and a correctness proof, not something else's code can call.
`hash_join_api.cuf` wraps the reliable 2-level table in a small, stable `bind(c)` surface (create/
build/probe/destroy, up to 8 independent tables at once), compiled into `libhashjoin.so` — **the
exact same `bind(c)` + ctypes + CuPy bridge pattern `../../MPDOK/mpdok_ops.py` already uses** for
its own solvers, not a new convention invented for this. `hash_join.py`'s `HashJoinTable` class
wraps it:

```python
from hash_join import HashJoinTable
import cupy as cp

keys = cp.asarray([101, 102, 103], dtype=cp.int64)
values = cp.asarray([0, 1, 2], dtype=cp.int32)          # e.g. row ids

with HashJoinTable(n_keys=len(keys), loadfactor=0.90) as table:
    table.build(keys, values)                             # raises if any key failed to insert
    result = table.probe(cp.asarray([101, 999, 103], dtype=cp.int64))   # 999 doesn't exist
    # result = [0, -1, 2]  (cp.int32, -1 = no match)
```

**Verified, not just written**: a 200,000-random-key correctness + failure-detection check (real
matches correct, absent keys correctly return no-match, zero build failures at 90% load) and an
honest failure-mode demo (an undersized overflow region makes `build()` raise `RuntimeError`
rather than silently drop keys) both pass — see `HOW_TO_USE.ipynb` and `build_howto_notebook.py`.

**Language decision, recorded honestly**: CUDA Fortran kernel + Python host (matching
`MPDOK`'s own pattern) was chosen over a Rust wrapper for this first packaging pass, specifically
because the immediate ask was "a how-to-use notebook with more examples" — Python/CuPy is the
lowest-friction path to that, and notebooks in this codebase are Python-first throughout. A Rust
FFI wrapper (matching the `stash` family's portable, libc-only ethos) is the natural next step
**if** a real daily-driver/production use case for this specifically-niche, high-load-factor
engine identifies itself — not built speculatively ahead of one, per this codebase's own
established pattern (`gp_engine` sat in a "parking lot" note for weeks before `climate_cat_lab`
gave it a reason to exist).

## Files

- `joindemo.cuf` — the original GPU hash join benchmark: build on R=4.19M keys, probe with
  S=67.1M rows, single-level tiny-pointer table vs. linear probing at the same slot budget.
- `joindemo_reliable.cuf` — same benchmark, using `tinymap`'s reliable 2-level table
  (`ttd_insert_r`/`ttd_find_r`) instead — the real fix built 2026-07-29 (see below).
- `join_example_small.cuf` — **a real, human-readable worked example, not just a throughput
  benchmark**: a tiny `customers JOIN orders` query (8 customers, 12 orders, including 2 orders
  referencing customer IDs that don't exist), run through both the linear-probe reference and
  the reliable tiny-pointer table side by side, with actual matched output printed and checked
  for agreement — a correctness demonstration, not only a speed one.
- `hash_join_api.cuf` — the `bind(c)` API surface (create/build/probe/destroy) compiled into
  `libhashjoin.so`, the packaged library `hash_join.py` calls.
- `hash_join.py` — the Python-facing `HashJoinTable` class (ctypes + CuPy).
- `build_howto_notebook.py` / `HOW_TO_USE.ipynb` — the packaged-API notebook: three worked
  examples (minimal correctness, realistic 200K-key scale, an honest undersized-overflow failure
  demo), calling `hash_join.py` live, not reading pre-generated CSVs.
- `build_join_notebook.py` — generates `HASH_JOIN_ENGINE.ipynb` from the three CSV outputs above.
- `HASH_JOIN_ENGINE.ipynb` — the consolidated notebook (math, the worked example's actual output,
  the benchmark charts, the honest summary).
- `Makefile`

## The worked example — both engines, actual output, not just Mr/s

```
order_id  customer_id  lp_tier  tt_tier  agree?
    1001          101        3        3     yes
    1003          977 NO MATCH NO MATCH     yes   <- customer 977 was never inserted
    1007          988 NO MATCH NO MATCH     yes   <- customer 988 was never inserted
    ...
  12 / 12 orders: both engines agree.
  reliable-table build failures (should be 0): 0
```

Both engines — independently built and probed — agree on all 12 rows, including both genuine
no-match cases (returning `NO MATCH`, not garbage or a crash). This is the property that makes
the throughput numbers below meaningful to compare at all: both tables are computing the same
correct join, just at different speeds and reliability.

## The large-scale benchmark (R=4.19M build keys, S=67.1M probe keys, B=16)

```
load |  TINY: build Mr/s  probe Mr/s |  LP: build Mr/s  probe Mr/s | tt_fail% match%
50   |        175          855       |       440         1612      |  0.08    99.9
90   |        115          631       |       304          901      |  6.00    94.0
99   |        108          598       |        61          163      |  9.50    90.5   <- tiny ~3.7x faster
```

**Honest reading:** linear probing is faster at low/moderate load (simpler, fewer reads per
probe). Tiny pointers win where it matters for a memory-tight, build-once/probe-many table: at
**high load (crossover ~95%)** they beat linear probing on *both* build and probe (~3.7× probe at
99%), with flat predictable latency (no probe-length tail) and 4-bit references. The single-level
match dip (90.5% at 99%) is what the reliable 2-level table (below) drives back to 100%.

### The reliable variant (`joindemo_reliable.cuf`, added 2026-07-29) — banks the fail-rate fix, and it's faster too

The single-level match dip above is exactly what `tinymap`'s own reliable 2-level path
(`ttd_insert_r`/`ttd_find_r` — bucket + linear-probe overflow backup, the same mechanism
`../kvpage.cuf` already uses) fixes. Wiring it into the join benchmark — no new mechanism, just
reusing existing module functions in the same timing harness — gives a real, measured three-way
at the same shape:

```
              build Mr/s @99%   probe Mr/s @99%   fail% @99%
linear probe       61.0             164.5            —
single-level tt   107.8             596.3           9.5%
reliable 2-level  128.9             680.9           0.0%
```

**The reliable table is not just fail-free (0.00% vs. 9.5%) — it's also faster on both build and
probe, at every load factor tested (50-99%).** Likely mechanism, stated as a real, honest
hypothesis rather than asserted: the 15% overflow region (`ovf_frac=0.15`, the same default
`spacedict`/`kvpage`/`stabledict` already use) gives more total slots for the same key count,
lowering effective per-bucket load — so part of the speed gain is the extra ~15% memory budget,
not necessarily a pure algorithmic improvement over the single-level table at matched capacity.
Worth a controlled same-capacity comparison if this gets revisited, not claimed as a free lunch.

## Competitive benchmarking — what was tried, honestly

**BGHT** ([github.com/owensgroup/BGHT](https://github.com/owensgroup/BGHT), "Better GPU Hash
Tables," Awad et al.) is the closest real academic competitor — a bucketed cuckoo hash table
reporting 100% success to 0.991 load factor at ~1.43 average probes in its own published results.
It was cloned, `cmake`-configured (had to `pip install cmake` and force
`-allow-unsupported-compiler`, since our CUDA 13.3/gcc 16 toolchain postdates this 2021-era code
by several CUDA major versions), and compiled cleanly — but its `bcht` table's **constructor
itself hangs**, reproduced at both 4.19M- and 1,024-key scale, isolated to something beyond plain
`thrust::fill` (tested in isolation, works fine). Most likely a genuine CUDA-toolkit-version
incompatibility given the multi-version gap, not a claim BGHT's design is broken — but a true
same-hardware head-to-head was **not achieved** this session.

**[cuCollections](https://github.com/NVIDIA/cuCollections)** (`cuco`) is the sharper answer to
"what's actually the standard": NVIDIA's own actively-maintained header-only library, and the
real engine behind cuDF's production hash joins/groupbys — not an academic reference
implementation. Being part of the RAPIDS ecosystem means it's continuously CI-tested against
current CUDA toolkits, far likelier to build cleanly here than BGHT was. Offers `cuco::static_map`
(fixed-size, open addressing + linear probing — a direct analogue to our linear-probe baseline)
and `cuco::dynamic_map` (grows via linked static maps — the closer structural analogue to our
reliable 2-level table). **Not yet attempted** — the natural next candidate if this thread is
picked up again, ahead of further BGHT toolchain archaeology.
