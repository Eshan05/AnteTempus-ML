"""Measure equivalent scan and indexed audit pipelines on fixed synthetic windows."""

import argparse
import json
import tracemalloc
from collections import Counter
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

from .window_audit import IndexedAudit, Query, ScanAudit, Transaction


WIDTH = 60
ACCOUNTS = 160
WINDOWS = 8
PER_WINDOW = 64
QUERY_COUNTS = (100, 8000, 40_000)
REPEATS = 3


def workload(seed: int):
    rng = Random(seed)
    rows = []
    for account in range(ACCOUNTS):
        for window in range(WINDOWS):
            for _ in range(PER_WINDOW):
                event = window * WIDTH + rng.randrange(WIDTH)
                delay = rng.randrange(1, 80) if rng.random() < 0.08 else 0
                rows.append(Transaction(len(rows), account, event, event + delay, rng.randrange(1, 20_000)))
    rng.shuffle(rows)
    queries = []
    for _ in range(max(QUERY_COUNTS)):
        start = rng.randrange(WINDOWS) * WIDTH
        queries.append(Query(rng.randrange(ACCOUNTS), start, start + WIDTH + rng.choice((0, 10, 30, 90))))
    return rows, queries


def elapsed(factory, rows, width, queries):
    start = perf_counter()
    audit = factory(rows, width)
    built = perf_counter()
    findings = [audit.query(query) for query in queries]
    end = perf_counter()
    return {"build_ms": (built - start) * 1000, "query_ms": (end - built) * 1000,
            "total_ms": (end - start) * 1000}, findings


def peak_bytes(factory, rows, width, queries) -> int:
    tracemalloc.start()
    try:
        audit = factory(rows, width)
        findings = [audit.query(query) for query in queries]
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def compare(rows, width, queries, repeats: int = REPEATS) -> dict:
    if repeats < 1:
        raise ValueError("Benchmark requires at least one repeat")
    methods = {"scan": ScanAudit, "index": IndexedAudit}
    runs = {key: [] for key in methods}
    for repeat in range(repeats):
        outcomes = {}
        for key in (methods if repeat % 2 == 0 else reversed(methods)):
            timing, outcomes[key] = elapsed(methods[key], rows, width, queries)
            runs[key].append(timing)
        if outcomes["scan"] != outcomes["index"]:
            raise RuntimeError("Indexed verifier disagrees with replay")

    report = {"queries": len(queries), "verdicts": dict(Counter(row.verdict for row in outcomes["scan"]))}
    for key, factory in methods.items():
        report[key] = {metric: median(result[metric] for result in runs[key]) for metric in runs[key][0]}
        report[key]["peak_bytes"] = peak_bytes(factory, rows, width, queries)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/raw/window-bench.json"))
    args = parser.parse_args(argv)
    if args.out.suffix.lower() != ".json":
        parser.error("--out must be a .json file.")
    rows, queries = workload(args.seed)
    report = {"schema": 1, "seed": args.seed, "rows": len(rows), "window_width": WIDTH, "repeats": REPEATS,
              "plans": [compare(rows, WIDTH, queries[:count]) for count in QUERY_COUNTS]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
