"""Compare both Python audits with equivalent in-process DuckDB SQL."""

import argparse
import json
import platform
from collections import Counter
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from .window_audit import Finding, IndexedAudit, Query, ScanAudit, Transaction, supported
from .window_bench import QUERY_COUNTS, WIDTH, elapsed, workload


REPEATS = 8
KEYS = "SELECT DISTINCT account, win, cutoff FROM requests"
REPLAY = f"""
WITH keys AS ({KEYS}),
findings AS (
    SELECT k.account, k.win, k.cutoff,
           coalesce(sum(t.cents), 0) AS batch_sum, count(t.id) AS batch_count,
           coalesce(sum(t.cents) FILTER (WHERE t.available_at <= k.cutoff), 0) AS seen_sum,
           count(t.id) FILTER (WHERE t.available_at <= k.cutoff) AS seen_count,
           arg_max(t.id, (t.available_at, t.id)) FILTER (WHERE t.available_at > k.cutoff) AS witness
    FROM keys k LEFT JOIN transactions t USING (account, win)
    GROUP BY k.account, k.win, k.cutoff
)
SELECT q.pos, batch_sum, batch_count, seen_sum, seen_count, witness
FROM requests q JOIN findings USING (account, win, cutoff)
"""

BUILD_PREFIX = """
CREATE TEMP TABLE prefixes AS
SELECT account, win, available_at,
       sum(subtotal) OVER bucket AS seen_sum,
       sum(n) OVER bucket AS seen_count
FROM (
    SELECT account, win, available_at, sum(cents) AS subtotal, count(*) AS n
    FROM transactions GROUP BY account, win, available_at
)
WINDOW bucket AS (PARTITION BY account, win ORDER BY available_at ROWS UNBOUNDED PRECEDING)
"""
BUILD_TOTALS = """
CREATE TEMP TABLE totals AS
SELECT account, win, sum(cents) AS batch_sum, count(*) AS batch_count,
       arg_max(id, (available_at, id)) AS last_id
FROM transactions GROUP BY account, win
"""

PREFIX = f"""
WITH keys AS ({KEYS}),
visible AS (
    SELECT k.account, k.win, k.cutoff,
           coalesce(p.seen_sum, 0) AS seen_sum, coalesce(p.seen_count, 0) AS seen_count
    FROM keys k ASOF LEFT JOIN prefixes p
         ON k.account = p.account AND k.win = p.win AND k.cutoff >= p.available_at
)
SELECT q.pos, coalesce(t.batch_sum, 0), coalesce(t.batch_count, 0), v.seen_sum, v.seen_count,
       CASE WHEN v.seen_count < t.batch_count THEN t.last_id END
FROM requests q JOIN visible v USING (account, win, cutoff)
LEFT JOIN totals t ON q.account = t.account AND q.win = t.win
"""


def elapsed_sql(duckdb, rows, width, queries, *, prefix):
    start = perf_counter()
    with duckdb.connect(config={"threads": 1, "max_temp_directory_size": "0B"}) as db:
        source = np.asarray([(r.id, r.account, r.event_at, r.available_at, r.cents) for r in rows],
                            dtype=np.int64).reshape(-1, 5).T
        db.execute("""
            CREATE TEMP TABLE transactions AS
            SELECT column0 AS id, column1 AS account, column2 AS event_at,
                   column2 - ((column2 % $width + $width) % $width) AS win,
                   column3 AS available_at, column4 AS cents
            FROM source
        """, {"width": width})
        (valid,), = db.execute("""
            SELECT count(*) = count(DISTINCT id) AND count(*) FILTER (WHERE available_at < event_at) = 0
            FROM transactions
        """).fetchall()
        if not valid:
            raise ValueError("Expected unique immutable IDs and availability at or after the event")
        if prefix:
            db.execute(BUILD_PREFIX)
            db.execute(BUILD_TOTALS)
        built = perf_counter()

        request_columns = np.asarray([(i, q.account, q.start, q.cutoff) for i, q in enumerate(queries)
                                      if supported(q, width)], dtype=np.int64).reshape(-1, 4).T
        db.execute("""
            CREATE TEMP TABLE requests AS
            SELECT column0 AS pos, column1 AS account, column2 AS win, column3 AS cutoff
            FROM request_columns
        """)
        output = db.execute(PREFIX if prefix else REPLAY).fetchall()
        if len(output) != request_columns.shape[1]:
            raise RuntimeError("DuckDB did not return one finding per supported query")
        findings = [Finding("unknown")] * len(queries)
        for pos, batch_sum, batch_count, seen_sum, seen_count, witness in output:
            findings[pos] = Finding("pass" if witness is None else "leak",
                                    (batch_sum, batch_count), (seen_sum, seen_count), witness)
        end = perf_counter()
    return {"build_ms": (built - start) * 1000, "query_ms": (end - built) * 1000,
            "total_ms": (end - start) * 1000}, findings


def compare(methods, rows, width, queries, repeats):
    runs = {key: [] for key in methods}
    keys = list(methods)
    for repeat in range(repeats):
        offset = repeat % len(keys)
        outcomes = {}
        for key in keys[offset:] + keys[:offset]:
            timing, outcomes[key] = methods[key](rows, width, queries)
            runs[key].append(timing)
        for key in keys[1:]:
            if outcomes[key] != outcomes[keys[0]]:
                raise RuntimeError(f"{key} disagrees with Python scan for {len(queries)} queries")

    return {"queries": len(queries),
            "verdicts": dict(Counter(r.verdict for r in outcomes[keys[0]])),
            **{key: {metric: median(run[metric] for run in samples) for metric in samples[0]}
               for key, samples in runs.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/raw/window-duckdb-bench.json"))
    args = parser.parse_args(argv)
    if args.out.suffix.lower() != ".json":
        parser.error("--out must be a .json file")
    try:
        import duckdb
    except ImportError:
        parser.error("install duckdb==2.0.1 in a disposable environment for this benchmark")

    methods = {
        "python_scan": lambda rows, width, queries: elapsed(ScanAudit, rows, width, queries),
        "python_index": lambda rows, width, queries: elapsed(IndexedAudit, rows, width, queries),
        "duckdb_replay": lambda rows, width, queries: elapsed_sql(duckdb, rows, width, queries, prefix=False),
        "duckdb_prefix": lambda rows, width, queries: elapsed_sql(duckdb, rows, width, queries, prefix=True),
    }
    # Warm up the Python/SQL bridge and check ties, inclusive availability, and empty/unsupported windows.
    fixture = [Transaction(1, 7, -1, 5, 30), Transaction(2, 7, -1, 5, 10),
               Transaction(3, 7, -60, 0, -5), Transaction(4, 7, 0, 0, 100)]
    checks = [Query(7, -60, 0), Query(7, -60, 5), Query(0, 0, 60), Query(7, 0, 59)]
    compare(methods, fixture, WIDTH, checks, 1)

    rows, queries = workload(args.seed)
    report = {"schema": 1, "seed": args.seed, "rows": len(rows), "window_width": WIDTH,
              "repeats": REPEATS, "duckdb": duckdb.__version__, "duckdb_threads": 1,
              "python": platform.python_version(), "numpy": np.__version__, "platform": platform.system(),
              "plans": [compare(methods, rows, WIDTH, queries[:count], REPEATS) for count in QUERY_COUNTS]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
